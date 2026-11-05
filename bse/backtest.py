"""Backside short-scalp research engine.

For every signal event it builds up to five entries (timing variants), then
simulates every combination of stop rule x exit rule x maximum hold on the
same 1-minute path. All combinations share one vectorised pass.

Fill and cost model (all conservative where the bar data is ambiguous):
  * Market short at a bar close: fills at close - half spread - slippage.
  * Rule 201 (SSR) active: the short must rest on the offer at close + 1 tick
    and fills only if price trades back up through it within 2 bars; a
    break-of-low (sell-stop) entry is impossible and is skipped.
  * Stop orders fill at max(stop, bar open) + half spread + slippage + a share
    of the bar's range (fast markets fill badly). Halts show up as gaps and the
    stop fills at the reopen print.
  * Profit targets are limit buys that fill only if price trades at least one
    tick THROUGH the limit; a gap below the limit fills at the open.
  * If a stop and a target are both inside one bar, the stop is assumed first
    (unless the bar opened beyond the target).
  * Close-based stops exit at that bar's close plus costs.
  * Commission (per share with min/cap), locate fee and the estimated spread
    are charged on every trade. Positions never overlap within one ticker for
    the same configuration, and everything is flat by 15:55.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config, Costs, hm
from .indicators import SessionData
from .signals import SIGNALS

BIG = 10 ** 6
TRADE_SIGNALS = [k for k, v in SIGNALS.items() if v[1] != "baseline"]


@dataclass(frozen=True)
class Rule:
    code: str
    label: str
    kind: str
    param: object


def build_rules(cfg: Config):
    sc = cfg.strat
    stops = [Rule("ema9", "Stop order above the 9 EMA", "level", "ema9"),
             Rule("ema20", "Stop order above the 20 EMA", "level", "ema20"),
             Rule("vwap", "Stop order above VWAP", "level", "vwap"),
             Rule("lh", "Stop order above the latest lower high", "lh", None)]
    stops += [Rule(f"pct{x:g}", f"Fixed stop {x * 100:g}% above entry", "pct", x) for x in sc.stops_pct]
    stops += [Rule(f"atr{k:g}", f"Stop {k:g} ATR above entry", "atr", k) for k in sc.stops_atr]
    stops += [Rule("ema9c", "Exit on a 1-min close above the 9 EMA", "close", "ema9"),
              Rule("ema20c", "Exit on a 1-min close above the 20 EMA", "close", "ema20"),
              Rule("vwapc", "Exit on a 1-min close above VWAP", "close", "vwap")]
    targets = [Rule(f"tp{x:g}", f"Cover at {x * 100:g}% profit", "pct", x) for x in sc.targets_pct]
    targets += [Rule("nextlow", "Cover when price breaks the prior low", "nextlow", None)]
    targets += [Rule(f"tr{x:g}", f"{x * 100:g}% trailing stop", "trail_pct", x) for x in sc.trail_pct]
    targets += [Rule(f"tra{k:g}", f"{k:g} ATR trailing stop", "trail_atr", k) for k in sc.trail_atr]
    targets += [Rule("none", "No target (stop or time exit only)", "none", None)]
    holds = list(sc.holds_min)
    return stops, targets, holds


def tod_bucket(mod: np.ndarray, cfg: Config) -> np.ndarray:
    edges = np.array([hm(x) for x in cfg.strat.tod_edges])
    return np.clip(np.searchsorted(edges, mod, side="right") - 1, 0, len(edges) - 2)


def spread_frac(s: SessionData, costs: Costs) -> np.ndarray:
    A = s.arr
    floor = np.maximum(costs.spread_floor_ticks * A["tick"] / s.c, costs.spread_floor_bps / 1e4)
    sp = np.maximum(floor, A["spread_ar"]) * costs.spread_mult
    return np.clip(sp, 0.0, costs.spread_cap_bps / 1e4)


def commission_frac(price: np.ndarray, costs: Costs) -> np.ndarray:
    per_share = costs.commission_per_share / np.maximum(price, 1e-6)
    return np.minimum(np.maximum(per_share, costs.commission_min / costs.notional_usd), costs.commission_cap_pct)


# --------------------------------------------------------------------- entries
@dataclass
class Entries:
    sig_i: int
    tim_i: int
    ev_idx: np.ndarray     # signal bar
    info: np.ndarray       # bar whose close the order is based on
    fill: np.ndarray       # bar in which the entry fills
    price: np.ndarray      # entry price before spread/slippage
    start: np.ndarray      # first bar of the exit path
    t_in: np.ndarray       # ordering time for the no-overlap rule
    passive: np.ndarray    # SSR resting order (no spread paid on entry)
    stop_entry: np.ndarray # sell-stop entry (break of low)
    ssr: np.ndarray


def _first_in_window(mask, e, mod, epoch, minutes, n):
    jmax = int(np.searchsorted(mod, mod[e] + minutes, side="right") - 1)
    if jmax <= e:
        return -1
    seg = mask[e + 1:jmax + 1] & (epoch[e + 1:jmax + 1] == epoch[e])
    if not seg.any():
        return -1
    return e + 1 + int(np.argmax(seg))


def build_entries(s: SessionData, perbar: dict, events: list[dict], cfg: Config) -> list[Entries]:
    n = len(s)
    A = s.arr
    mod, c, o, h, l = s.mod, s.c, s.o, s.h, s.l
    epoch = perbar["epoch_id"]
    ssr = A["ssr"]
    tick = A["tick"]
    sc, cc = cfg.strat, cfg.costs
    t_lo, t_hi = hm(cfg.session.trade_start), hm(cfg.session.trade_last_entry)
    after_gap = A["after_gap"]
    sig_index = {k: i for i, k in enumerate(TRADE_SIGNALS)}
    groups: dict[tuple, list] = {}

    for ev in events:
        code = ev["sig"]
        if code not in sig_index:
            continue
        e = ev["idx"]
        ref = SIGNALS[code][2]
        for ti, timing in enumerate(sc.entry_timings):
            info = fill = -1
            price = np.nan
            stop_entry = False
            open_entry = False
            if timing == "signal_close":
                info = fill = e
                price = c[e]
            elif timing == "next_open":
                if e + 1 < n:
                    info, fill = e, e + 1
                    price = o[e + 1]
                    open_entry = True
            elif timing == "failed_reclaim":
                j = _first_in_window(perbar[f"fr_{ref}"], e, mod, epoch, sc.confirm_window_min, n)
                if j > 0:
                    info = fill = j
                    price = c[j]
            elif timing == "lower_low_break":
                j = _first_in_window(l < l[e], e, mod, epoch, sc.confirm_window_min, n)
                if j > 0:
                    info, fill = j - 1, j
                    lvl = l[e] - tick[e]
                    price = min(lvl, o[j])
                    stop_entry = True
            elif timing == "lower_high_below_ema20":
                mask = perbar["lh_flag"] & (perbar["lh_peak_idx"] > e)
                ok = np.zeros(n, bool)
                cand = np.flatnonzero(mask)
                if len(cand):
                    pk = perbar["lh_peak_idx"][cand]
                    good = h[pk] < A["ema20"][pk]
                    ok[cand[good]] = True
                j = _first_in_window(ok, e, mod, epoch, sc.lower_high_window_min, n)
                if j > 0:
                    info = fill = j
                    price = c[j]
            if fill < 0 or not np.isfinite(price):
                continue
            if not (t_lo <= mod[fill] <= t_hi):
                continue
            if price < cc.min_short_price:
                continue
            if cc.skip_after_halt_bars and after_gap[max(0, fill - cc.skip_after_halt_bars + 1):fill + 1].any():
                continue
            passive = False
            # a break-of-low entry can itself trigger Rule 201 on the way down
            is_ssr = bool(ssr[info] or (stop_entry and ssr[fill]))
            start = fill if (open_entry or stop_entry) else fill + 1
            t_in = fill + (0.0 if open_entry else (0.5 if stop_entry else 0.99))
            if is_ssr and cc.ssr_policy != "ignore":
                if cc.ssr_policy == "skip" or stop_entry:
                    continue
                lim = c[info] + tick[info]
                k = -1
                for jj in range(info + 1, min(n, info + 1 + cc.ssr_fill_bars)):
                    if h[jj] >= lim + tick[info]:
                        k = jj
                        break
                if k < 0:
                    continue
                fill, price, start, passive, t_in = k, lim, k, True, k + 0.5
            if start >= n:
                continue
            groups.setdefault((sig_index[code], ti), []).append(
                (e, info, fill, price, start, t_in, passive, stop_entry, is_ssr))

    out = []
    for (si, ti), rows in groups.items():
        rows.sort(key=lambda r: r[5])
        a = np.array(rows, dtype=float)
        out.append(Entries(si, ti, a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2].astype(int),
                           a[:, 3], a[:, 4].astype(int), a[:, 5], a[:, 6].astype(bool),
                           a[:, 7].astype(bool), a[:, 8].astype(bool)))
    return out


# ------------------------------------------------------------------ simulation
def _first(mask):
    anyv = mask.any(axis=1)
    return np.where(anyv, mask.argmax(axis=1), BIG)


@dataclass
class SimResult:
    j_exit: np.ndarray    # (m,S,G,H) absolute bar index of exit
    p_exit: np.ndarray    # exit price before costs
    reason: np.ndarray    # 0 stop, 1 target, 2 trailing, 3 time
    t_out: np.ndarray
    risk: np.ndarray      # (m,S) initial risk as a fraction of entry
    stop_market: np.ndarray


def simulate(s: SessionData, ent: Entries, stops, targets, holds, cfg: Config,
             j_flat: int) -> SimResult:
    A = s.arr
    m = len(ent.price)
    P = ent.price
    info = ent.info
    tick = A["tick"][info]
    buf = cfg.strat.stop_buffer_pct
    L = int(j_flat - ent.start.min() + 1)
    L = max(L, 1)
    idx = ent.start[:, None] + np.arange(L)[None, :]
    valid = idx <= j_flat
    ic = np.minimum(idx, j_flat)
    H, Lo, C, O, M = s.h[ic], s.l[ic], s.c[ic], s.o[ic], s.mod[ic]
    Hm = np.where(valid, H, -np.inf)
    Lm = np.where(valid, Lo, np.inf)
    rows = np.arange(m)

    nS, nG, nH = len(stops), len(targets), len(holds)
    js = np.full((m, nS), BIG)
    ps = np.full((m, nS), np.nan)
    skind = np.zeros(nS, int)          # 0 stop order (intrabar), 1 close-based
    risk = np.zeros((m, nS))
    for k, r in enumerate(stops):
        if r.kind == "close":
            lv = A[r.param][ic]
            hit = (C > lv) & valid
            j = _first(hit)
            js[:, k] = j
            jj = np.minimum(j, L - 1)
            ps[:, k] = C[rows, jj]
            skind[k] = 1
            risk[:, k] = np.maximum(A[r.param][info] / P - 1.0, buf)
            continue
        if r.kind == "level":
            S = np.maximum(A[r.param][info], P) * (1 + buf)
        elif r.kind == "lh":
            lh = A["hod"][info].copy()
            lp = s.arr.get("_lh_price")
            if lp is not None:
                v = lp[info]
                lh = np.where(np.isfinite(v), v, lh)
            S = np.maximum(lh, P) * (1 + buf)
        elif r.kind == "pct":
            S = P * (1 + r.param)
        else:  # atr
            S = P + r.param * A["atr"][info]
        S = np.maximum(S, P + tick)
        hit = Hm >= S[:, None]
        j = _first(hit)
        js[:, k] = j
        jj = np.minimum(j, L - 1)
        ps[:, k] = np.maximum(S, O[rows, jj])
        risk[:, k] = S / P - 1.0

    jt = np.full((m, nG), BIG)
    pt = np.full((m, nG), np.nan)
    tkind = np.zeros(nG, int)          # 0 limit, 1 trailing stop, 2 none
    tgap = np.zeros((m, nG), bool)
    prev_min = None
    for k, r in enumerate(targets):
        if r.kind == "none":
            tkind[k] = 2
            continue
        if r.kind in ("pct", "nextlow"):
            if r.kind == "pct":
                T = P * (1 - r.param)
            else:
                T = np.minimum(A["move_low"][info], s.l[info])
            T = np.minimum(T, P - tick)
            hit = Lm <= (T - tick)[:, None]
            j = _first(hit)
            jj = np.minimum(j, L - 1)
            jt[:, k] = j
            pt[:, k] = np.minimum(T, O[rows, jj])
            tgap[:, k] = O[rows, jj] <= T
            continue
        if prev_min is None:
            run = np.minimum.accumulate(Lm, axis=1)
            prev_min = np.minimum(np.concatenate([P[:, None], run[:, :-1]], axis=1), P[:, None])
        if r.kind == "trail_pct":
            TL = prev_min * (1 + r.param)
        else:
            TL = prev_min + r.param * A["atr"][info][:, None]
        hit = Hm >= TL
        j = _first(hit)
        jj = np.minimum(j, L - 1)
        jt[:, k] = j
        pt[:, k] = np.maximum(TL[rows, jj], O[rows, jj])
        tkind[k] = 1

    jh = np.zeros((m, nH), int)
    m_in = s.mod[ent.fill]
    for k, hold in enumerate(holds):
        lim = m_in + hold
        cnt = (valid & (M <= lim[:, None])).sum(axis=1) - 1
        jh[:, k] = np.maximum(cnt, 0)
    ph = C[rows[:, None], jh]

    # combine stop x target ---------------------------------------------------
    Aj = js[:, :, None]
    Bj = jt[:, None, :]
    st_static = (skind == 0)[None, :, None]
    t_limit = (tkind == 0)[None, None, :]
    t_trail = (tkind == 1)[None, None, :]
    tie = (Aj == Bj) & (Aj < BIG)
    tgt_tie = ((~st_static) & (t_limit | t_trail)) | (st_static & t_limit & tgap[:, None, :]) | \
              (st_static & t_trail & (pt[:, None, :] <= ps[:, :, None]))
    tfirst = (Bj < Aj) | (tie & tgt_tie)
    j1 = np.where(tfirst, Bj, Aj)
    p1 = np.where(tfirst, pt[:, None, :], ps[:, :, None])
    r1 = np.where(tfirst, np.where(t_trail, 2, 1), 0)
    sm1 = np.where(tfirst, t_trail, st_static)
    # vs time exit -------------------------------------------------------------
    j1e, p1e, r1e, sm1e = j1[..., None], p1[..., None], r1[..., None], sm1[..., None]
    jhb = jh[:, None, None, :]
    use_h = jhb < j1e
    jrel = np.where(use_h, jhb, j1e)
    pex = np.where(use_h, ph[:, None, None, :], p1e)
    reason = np.where(use_h, 3, r1e)
    smk = np.where(use_h, False, sm1e)
    j_abs = ent.start[:, None, None, None] + jrel
    close_based = (reason == 3) | ((reason == 0) & (skind[None, :, None, None] == 1))
    t_out = j_abs + np.where(close_based, 0.99, 0.5)
    return SimResult(j_abs, pex, reason, t_out, risk, smk)


def net_returns(s: SessionData, ent: Entries, sim: SimResult, costs: Costs) -> np.ndarray:
    sp = spread_frac(s, costs)
    rng = (s.h - s.l) / np.maximum(s.c, 1e-9)
    P = ent.price[:, None, None, None]
    slip = costs.slippage_bps / 1e4
    half_in = np.where(ent.passive, 0.0, sp[ent.fill] / 2 + slip)
    half_in = half_in + np.where(ent.stop_entry, costs.stop_slip_range_frac * rng[ent.fill], 0.0)
    p_in = P * (1 - half_in[:, None, None, None])
    j = sim.j_exit
    limit_exit = (sim.reason == 1)
    extra = sp[j] / 2 + slip + np.where(sim.stop_market, costs.stop_slip_range_frac * rng[j], 0.0)
    p_out = np.where(limit_exit, sim.p_exit, sim.p_exit * (1 + extra))
    gross = (p_in - p_out) / p_in
    fee_in = commission_frac(ent.price, costs)[:, None, None, None]
    fee_out = commission_frac(sim.p_exit, costs) * sim.p_exit / P
    return gross - fee_in - fee_out - costs.locate_pct


def no_overlap(ent: Entries, sim: SimResult) -> np.ndarray:
    m = len(ent.price)
    keep = np.zeros(sim.t_out.shape, bool)
    last = np.full(sim.t_out.shape[1:], -1.0)
    for k in range(m):
        ok = ent.t_in[k] > last
        keep[k] = ok
        last = np.where(ok, sim.t_out[k], last)
    return keep


def session_j_flat(s: SessionData, cfg: Config) -> int:
    return int(np.searchsorted(s.mod, hm(cfg.session.flat_by), side="right") - 1)


# --------------------------------------------------------------------- the grid
class Grid:
    """Accumulates every configuration's trades without storing them."""

    def __init__(self, cfg: Config, days: list[str]):
        self.cfg = cfg
        self.stops, self.targets, self.holds = build_rules(cfg)
        self.nSig, self.nT = len(TRADE_SIGNALS), len(cfg.strat.entry_timings)
        self.nS, self.nG, self.nH = len(self.stops), len(self.targets), len(self.holds)
        self.block = self.nS * self.nG * self.nH
        self.ncfg = self.nSig * self.nT * self.block
        self.days = days
        self.day_i = {d: i for i, d in enumerate(days)}
        import datetime as _dt
        d0 = _dt.date.fromisoformat(days[0])
        monday = d0 - _dt.timedelta(days=d0.weekday())
        self.week_of_day = np.array([(_dt.date.fromisoformat(d) - monday).days // 7 for d in days])
        self.nW = int(self.week_of_day.max()) + 1
        self.nB = len(cfg.strat.tod_edges) - 1
        nc, nd = self.ncfg, len(days)
        f4 = np.float32
        # per-WEEK sums of squares (per-day arrays would not fit for a 2-year window)
        self.w_sq = np.zeros((nc, self.nW), f4)
        self.wb_sum = np.zeros((nc, self.nW, self.nB), f4)
        # trade counts per (config, week, bucket) stay far below 65,535
        self.wb_n = np.zeros((nc, self.nW, self.nB), np.uint16)
        self.wb_win = np.zeros((nc, self.nW, self.nB), np.uint16)
        self.pos = np.zeros(nc)
        self.neg = np.zeros(nc)
        self.rmult = np.zeros(nc)
        self.hold_min = np.zeros(nc)
        self.reasons = np.zeros((nc, 4), np.int64)
        self.scen = {k: {"sum": np.zeros((nc, self.nB)), "n": np.zeros((nc, self.nB), np.int64),
                         "win": np.zeros((nc, self.nB), np.int64)} for k in ("low", "high")}
        self.cost_base = cfg.costs
        self.cost_scen = {k: cfg.costs_for(k) for k in ("low", "high")}

    @staticmethod
    def estimate_bytes(cfg: Config, n_weeks: int) -> int:
        stops, targets, holds = build_rules(cfg)
        ncfg = len(TRADE_SIGNALS) * len(cfg.strat.entry_timings) * len(stops) * len(targets) * len(holds)
        nb = len(cfg.strat.tod_edges) - 1
        return ncfg * n_weeks * (nb * (4 + 2 + 2) + 4)

    def cfg_id(self, si, ti, sk, gk, hk):
        return (((si * self.nT + ti) * self.nS + sk) * self.nG + gk) * self.nH + hk

    def decode(self, cid: int) -> dict:
        hk = cid % self.nH
        r = cid // self.nH
        gk = r % self.nG
        r //= self.nG
        sk = r % self.nS
        r //= self.nS
        ti = r % self.nT
        si = r // self.nT
        return {"sig": TRADE_SIGNALS[si], "timing": self.cfg.strat.entry_timings[ti],
                "stop": self.stops[sk].code, "target": self.targets[gk].code,
                "hold": int(self.holds[hk]), "si": si, "ti": ti, "sk": sk, "gk": gk, "hk": hk}

    def add_session(self, s: SessionData, perbar: dict, events: list[dict]):
        if s.date not in self.day_i:
            return
        s.arr["_lh_price"] = perbar["lh_price"]
        di = self.day_i[s.date]
        wi = self.week_of_day[di]
        j_flat = session_j_flat(s, self.cfg)
        if j_flat < 1:
            return
        for ent in build_entries(s, perbar, events, self.cfg):
            ent_ok = ent.start <= j_flat
            if not ent_ok.any():
                continue
            if not ent_ok.all():
                ent = _subset(ent, ent_ok)
            sim = simulate(s, ent, self.stops, self.targets, self.holds, self.cfg, j_flat)
            r = net_returns(s, ent, sim, self.cost_base)
            keep = no_overlap(ent, sim)
            base = self.cfg_id(ent.sig_i, ent.tim_i, 0, 0, 0)
            sl = slice(base, base + self.block)
            rk = np.where(keep, r, 0.0)
            self.w_sq[sl, wi] += (rk * rk).sum(0).ravel()
            self.pos[sl] += np.where(rk > 0, rk, 0).sum(0).ravel()
            self.neg[sl] += np.where(rk < 0, rk, 0).sum(0).ravel()
            rm = r / np.maximum(sim.risk[:, :, None, None], 1e-3)
            self.rmult[sl] += np.where(keep, rm, 0).sum(0).ravel()
            dur = s.mod[np.minimum(sim.j_exit, len(s) - 1)] - s.mod[ent.fill][:, None, None, None]
            self.hold_min[sl] += np.where(keep, dur, 0).sum(0).ravel()
            for q in range(4):
                self.reasons[sl, q] += (keep & (sim.reason == q)).sum(0).ravel()
            bk = tod_bucket(s.mod[ent.fill], self.cfg)
            rl = {k: net_returns(s, ent, sim, c) for k, c in self.cost_scen.items()}
            for b in np.unique(bk):
                mb = (bk == b)[:, None, None, None] & keep
                self.wb_sum[sl, wi, b] += np.where(mb, r, 0).sum(0).ravel()
                self.wb_n[sl, wi, b] += mb.sum(0).ravel().astype(np.uint16)
                self.wb_win[sl, wi, b] += (mb & (r > 0)).sum(0).ravel().astype(np.uint16)
                for k, rr in rl.items():
                    self.scen[k]["sum"][sl, b] += np.where(mb, rr, 0).sum(0).ravel()
                    self.scen[k]["n"][sl, b] += mb.sum(0).ravel()
                    self.scen[k]["win"][sl, b] += (mb & (rr > 0)).sum(0).ravel()

    def merge(self, other: "Grid"):
        for name in ("w_sq", "wb_sum", "wb_n", "wb_win", "pos", "neg",
                     "rmult", "hold_min", "reasons"):
            getattr(self, name).__iadd__(getattr(other, name))
        for k in self.scen:
            for f in self.scen[k]:
                self.scen[k][f] += other.scen[k][f]


def _subset(ent: Entries, mask: np.ndarray) -> Entries:
    return Entries(ent.sig_i, ent.tim_i, *(getattr(ent, f)[mask] for f in
                   ("ev_idx", "info", "fill", "price", "start", "t_in", "passive", "stop_entry", "ssr")))


# ------------------------------------------------------- detailed trade records
def trades_for(s: SessionData, perbar: dict, events: list[dict], cfg: Config,
               wanted: dict[tuple, list[tuple]], grid: Grid, flatten_at: int | None = None) -> list[dict]:
    """Individual trades for selected configs. wanted: (si, ti) -> [(sk, gk, hk, cid)].
    flatten_at: optional minute-of-day at which open positions are covered."""
    out = []
    s.arr["_lh_price"] = perbar["lh_price"]
    j_flat = session_j_flat(s, cfg)
    if flatten_at is not None:
        j_flat = min(j_flat, int(np.searchsorted(s.mod, flatten_at, side="right") - 1))
    if j_flat < 1:
        return out
    scen = {"base": cfg.costs, "low": cfg.costs_for("low"), "high": cfg.costs_for("high")}
    for ent in build_entries(s, perbar, events, cfg):
        key = (ent.sig_i, ent.tim_i)
        if key not in wanted:
            continue
        ok = ent.start <= j_flat
        if not ok.any():
            continue
        if not ok.all():
            ent = _subset(ent, ok)
        sim = simulate(s, ent, grid.stops, grid.targets, grid.holds, cfg, j_flat)
        rs = {k: net_returns(s, ent, sim, c) for k, c in scen.items()}
        keep = no_overlap(ent, sim)
        for sk, gk, hk, cid in wanted[key]:
            for k in np.flatnonzero(keep[:, sk, gk, hk]):
                j = int(sim.j_exit[k, sk, gk, hk])
                out.append({
                    "cid": int(cid), "sid": s.sid, "ticker": s.ticker, "date": s.date,
                    "sig_idx": int(ent.ev_idx[k]), "entry_idx": int(ent.fill[k]),
                    "entry_mod": int(s.mod[ent.fill[k]]), "entry_px": float(ent.price[k]),
                    "exit_idx": j, "exit_mod": int(s.mod[min(j, len(s) - 1)]),
                    "exit_px": float(sim.p_exit[k, sk, gk, hk]),
                    "reason": int(sim.reason[k, sk, gk, hk]),
                    "r": float(rs["base"][k, sk, gk, hk]), "r_low": float(rs["low"][k, sk, gk, hk]),
                    "r_high": float(rs["high"][k, sk, gk, hk]),
                    "risk": float(sim.risk[k, sk]), "ssr": bool(ent.ssr[k]), "passive": bool(ent.passive[k]),
                    "halt_in_trade": bool(s.arr["after_gap"][ent.fill[k] + 1:j + 1].any()),
                })
    return out
