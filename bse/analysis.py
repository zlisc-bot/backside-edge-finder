"""Statistics behind every number on the dashboard.

Conventions
  * A signal "worked" if price fell 1 ATR below the signal close before it rose
    1 ATR above it, within 30 minutes. It "failed" if the +1 ATR came first (a
    bar touching both counts as failed). "Edge" = P(worked) - P(failed); a
    coin-flip market gives ~0.
  * Lift = a signal's value minus the CONTROL sample's value at the same times
    of day (control = every 5th bar after a high, no signal required). A signal
    with no lift is no better than shorting at a random moment.
  * Intervals are 95% bootstrap intervals resampling whole sessions (signals)
    or whole days (trades), because observations inside one are correlated.
  * Horizon metrics are only computed when the full horizon fits before the
    close, so late-day signals are not flattered or penalised by a short window.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from . import calendar as cal
from .config import Config, fmt_hm, hm
from .signals import SIGNALS
from .stages import STAGES
from .validation import cluster_bootstrap

warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")

COARSE = [("Pre-market", 0, 570), ("Before 10:00", 570, 600), ("10:00-11:00", 600, 660),
          ("11:00-12:00", 660, 720), ("Afternoon", 720, 1440)]
HZS = (5, 15, 30, 60)


def tod15(mod):
    mod = np.asarray(mod)
    return np.where(mod < 570, -1, np.minimum((mod - 570) // 15, 25))


def tod15_label(i: int) -> str:
    return "Pre-market" if i < 0 else f"{fmt_hm(570 + 15 * i)}"


def coarse(mod):
    mod = np.asarray(mod)
    out = np.empty(len(mod), dtype=object)
    for name, a, b in COARSE:
        out[(mod >= a) & (mod < b)] = name
    return out


def prep_events(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev[ev["has_future"]].copy()
    rth_end = ev["date"].map(lambda d: cal.rth_end_minute(d)).to_numpy()
    for H in HZS:
        short = ev["mod"].to_numpy() + H > rth_end
        cols = [c for c in ev.columns if c.endswith(f"_{H}") and not c.startswith("dn")]
        cols += [f"dn{x}_{H}" for x in (5, 10, 20) if f"dn{x}_{H}" in ev.columns]
        for c in cols:
            ev.loc[short, c] = np.nan
        if H == 30:
            for c in ("fp_atr", "fp_pct"):
                ev.loc[short, c] = np.nan
    ev["full30"] = ev["fp_atr"].notna() | (ev["mod"].to_numpy() + 30 <= rth_end)
    fp = ev["fp_atr"]
    ev["worked"] = np.where(ev["full30"], (fp == 0).astype(float), np.nan)
    ev["failed"] = np.where(ev["full30"], (fp == 1).astype(float), np.nan)
    ev["unres"] = np.where(ev["full30"], fp.isna().astype(float), np.nan)
    ev["edge"] = ev["worked"] - ev["failed"]
    fpp = ev["fp_pct"]
    ev["worked_pct"] = np.where(ev["full30"], (fpp == 0).astype(float), np.nan)
    ev["failed_pct"] = np.where(ev["full30"], (fpp == 1).astype(float), np.nan)
    ev["tod15"] = tod15(ev["mod"])
    ev["coarse"] = coarse(ev["mod"])
    ev["eid"] = ev["sid"] + "|" + ev["sig"] + "|" + ev["idx"].astype(str)
    return ev


METRICS = [("n", "idx", "count"), ("sessions", "sid", "nunique"), ("tickers", "ticker", "nunique"),
           ("days", "date", "nunique")]
METRICS += [(f"p_nml_{h}", f"nml_{h}", "mean") for h in ("5", "15", "30", "60", "eod")]
METRICS += [(f"p_nlod_{h}", f"nlod_{h}", "mean") for h in ("15", "60", "eod")]
METRICS += [(f"p_dn{x}_{h}", f"dn{x}_{h}", "mean") for h in ("60", "eod") for x in (5, 10, 20)]
_AG = {"mean": "mean", "median": "med"}
METRICS += [(f"mfe_{h}_{_AG[a]}", f"mfe_{h}", a) for h in ("15", "30", "60", "eod") for a in ("mean", "median")]
METRICS += [(f"mae_{h}_{_AG[a]}", f"mae_{h}", a) for h in ("15", "30", "60", "eod") for a in ("mean", "median")]
METRICS += [("t_nml_med", "t_nml", "median"), ("t_low60_med", "t_low60", "median"),
            ("p_worked", "worked", "mean"), ("p_failed", "failed", "mean"), ("p_unres", "unres", "mean"),
            ("edge", "edge", "mean"), ("p_worked_pct", "worked_pct", "mean"),
            ("p_failed_pct", "failed_pct", "mean")]
METRICS += [(f"rec_{l}_{h}", f"rec_{l}_{h}", "mean") for l in ("ema9", "ema20", "vwap") for h in ("15", "30", "eod")]
METRICS += [(f"p_nhod_{h}", f"nhod_{h}", "mean") for h in ("30", "60", "eod")]
METRICS += [("fwd_15_mean", "fwd_15", "mean"), ("fwd_30_mean", "fwd_30", "mean"),
            ("hod_age_med", "mins_since_hod", "median"), ("pullback_med", "pullback", "median"),
            ("time_med", "mod", "median"), ("gain_pc_med", "gain_pc", "median"),
            ("price_med", "price", "median")]


def summarize(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    agg = {name: pd.NamedAgg(column=col, aggfunc=fn) for name, col, fn in METRICS if col in df.columns}
    out = df.groupby(keys, observed=True).agg(**agg).reset_index()
    return out


class Lift:
    """Time-matched comparison with the control sample, with session bootstrap."""

    def __init__(self, ev: pd.DataFrame, metrics: list[str], reps: int, seed: int):
        self.codes, self.uniq = pd.factorize(ev["sid"])
        self.nS = len(self.uniq)
        self.nb = 27
        self.b = (ev["tod15"].to_numpy() + 1).astype(int)
        self.ev = ev
        rng = np.random.default_rng(seed)
        self.W = rng.poisson(1.0, size=(reps, self.nS)).astype(np.float32)
        ctl = (ev["sig"] == "B00").to_numpy()
        self.metrics = metrics
        self.ctl_rate, self.ctl_boot = {}, {}
        for m in metrics:
            cnt, sm = self._mat(ctl, m)
            with np.errstate(invalid="ignore", divide="ignore"):
                self.ctl_rate[m] = sm.sum(0) / cnt.sum(0)
                self.ctl_boot[m] = (self.W @ sm) / (self.W @ cnt)

    def _mat(self, mask, m):
        x = self.ev[m].to_numpy(float)
        ok = mask & np.isfinite(x)
        flat = self.codes[ok] * self.nb + self.b[ok]
        cnt = np.bincount(flat, minlength=self.nS * self.nb).reshape(self.nS, self.nb).astype(np.float32)
        sm = np.bincount(flat, weights=x[ok], minlength=self.nS * self.nb).reshape(self.nS, self.nb).astype(np.float32)
        return cnt, sm

    def overall(self, mask: np.ndarray, m: str) -> dict:
        cnt, sm = self._mat(mask, m)
        tot = cnt.sum()
        if tot < 5:
            return {"obs": math.nan, "exp": math.nan, "lift": math.nan, "lo": math.nan, "hi": math.nan}
        cr = np.nan_to_num(self.ctl_rate[m], nan=0.0)
        has = np.isfinite(self.ctl_rate[m])
        w = cnt.sum(0) * has
        obs = sm.sum() / tot
        exp = (w * cr).sum() / max(w.sum(), 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            SC, SS = self.W @ cnt, self.W @ sm
            ob = SS.sum(1) / SC.sum(1)
            wb = SC * has
            eb = (wb * np.nan_to_num(self.ctl_boot[m], nan=0.0)).sum(1) / np.maximum(wb.sum(1), 1)
            lb = ob - eb
            olo, ohi = np.nanpercentile(ob, [2.5, 97.5])
            llo, lhi = np.nanpercentile(lb, [2.5, 97.5])
        return {"obs": float(obs), "obs_lo": float(olo), "obs_hi": float(ohi), "exp": float(exp),
                "lift": float(obs - exp), "lo": float(llo), "hi": float(lhi)}

    def by_bucket(self, mask: np.ndarray, m: str) -> pd.DataFrame:
        cnt, sm = self._mat(mask, m)
        with np.errstate(invalid="ignore", divide="ignore"):
            obs = sm.sum(0) / cnt.sum(0)
            SC, SS = self.W @ cnt, self.W @ sm
            ob = SS / SC
            lb = ob - self.ctl_boot[m]
            lo = np.nanpercentile(lb, 2.5, axis=0)
            hi = np.nanpercentile(lb, 97.5, axis=0)
        return pd.DataFrame({"tod15": np.arange(self.nb) - 1, "n": cnt.sum(0), "obs": obs,
                             "ctl": self.ctl_rate[m], "lift": obs - self.ctl_rate[m], "lo": lo, "hi": hi})


def signal_tables(ev: pd.DataFrame, cfg: Config) -> dict:
    keys = ["edge", "nml_15", "mfe_30", "failed"]
    lift = Lift(ev, keys, cfg.val.bootstrap_reps // 2, cfg.val.seed)
    stats = summarize(ev, ["sig"])
    rows = []
    for code in stats["sig"]:
        mask = (ev["sig"] == code).to_numpy()
        r = {"sig": code}
        for m in keys:
            o = lift.overall(mask, m)
            r.update({f"{m}_lift": o["lift"], f"{m}_lift_lo": o["lo"], f"{m}_lift_hi": o["hi"],
                      f"{m}_ctl": o["exp"], f"{m}_lo": o.get("obs_lo"), f"{m}_hi": o.get("obs_hi")})
        rows.append(r)
    stats = stats.merge(pd.DataFrame(rows), on="sig")
    # recurrence: minutes between consecutive same-type signals in a session
    e2 = ev[ev["sig"] != "B00"].sort_values(["sid", "sig", "mod"])
    gap = e2.groupby(["sid", "sig"])["mod"].diff()
    rec = gap.groupby(e2["sig"]).median().rename("recur_med_min")
    stats = stats.merge(rec, left_on="sig", right_index=True, how="left")
    stats["name"] = stats["sig"].map(lambda k: SIGNALS[k][0])
    stats["group"] = stats["sig"].map(lambda k: SIGNALS[k][1])

    # by 15-minute bucket
    tod = summarize(ev, ["sig", "tod15"])
    parts = []
    for code in tod["sig"].unique():
        mask = (ev["sig"] == code).to_numpy()
        for m in ("edge", "nml_15"):
            bb = lift.by_bucket(mask, m)[["tod15", "lift", "lo", "hi", "ctl"]]
            bb.columns = ["tod15"] + [f"{m}_{x}" for x in ("lift", "lift_lo", "lift_hi", "ctl")]
            bb["sig"] = code
            parts.append(bb)
    bl = pd.concat(parts).groupby(["sig", "tod15"]).first().reset_index()
    tod = tod.merge(bl, on=["sig", "tod15"], how="left")
    tod["bucket"] = tod["tod15"].map(tod15_label)

    coarse_t = summarize(ev, ["sig", "coarse"])
    return {"signals": stats, "signal_tod": tod, "signal_coarse": coarse_t, "lift": lift}


def _bins(x, edges, labels):
    return pd.cut(x, edges, labels=labels, right=False)


def context_tables(ev: pd.DataFrame, lift: Lift) -> pd.DataFrame:
    """Each signal split by what was observable at the time (plus one
    clearly-labelled hindsight split)."""
    e = ev.copy()
    e["c_gain"] = _bins(e["gain_pc"] * 100, [-1e9, 25, 50, 100, 200, 1e9],
                        ["<25%", "25-50%", "50-100%", "100-200%", "200%+"])
    e["c_age"] = _bins(e["mins_since_hod"], [0, 5, 15, 30, 60, 1e9], ["0-5 min", "5-15", "15-30", "30-60", "60+"])
    e["c_pull"] = _bins(e["pullback"] * 100, [-1, 5, 10, 20, 35, 1e9], ["<5%", "5-10%", "10-20%", "20-35%", "35%+"])
    e["c_price"] = _bins(e["price"], [0, 0.25, 0.5, 0.75, 1.0, 1e9], ["<$0.25", "$0.25-0.50", "$0.50-0.75", "$0.75-1.00", ">=$1.00"])
    e["c_cross"] = np.where(e["e9_above_e20"], "Before 9/20 cross", "After 9/20 cross")
    e["c_vwap"] = np.where(e["below_vwap"], "Below VWAP", "Above VWAP")
    e["c_ssr"] = np.where(e["ssr"], "SSR active", "No SSR")
    e["c_rt"] = np.where(e["known_sub1"] & (e["cumvol"] >= 1_000_000), "Observable live", "Needed hindsight")
    e["c_true"] = np.where(e["after_true_hod"], "After the day's high (hindsight)", "Before the day's high (hindsight)")
    dims = {"c_gain": "Move into the high (vs prior close)", "c_age": "Minutes since the high",
            "c_pull": "Pullback from the high", "c_price": "Price", "c_cross": "9 EMA vs 20 EMA",
            "c_vwap": "Price vs VWAP", "c_ssr": "Short-sale restriction", "c_rt": "Universe known in real time",
            "c_true": "Relative to the day's eventual high"}
    out = []
    for dim, title in dims.items():
        t = summarize(e, ["sig", dim]).rename(columns={dim: "slice"})
        t["slice"] = t["slice"].astype(str)
        t["dimension"] = title
        t["dim"] = dim
        # control within the same slice
        ctl = summarize(e[e["sig"] == "B00"], [dim]).rename(columns={dim: "slice"})
        ctl["slice"] = ctl["slice"].astype(str)
        t = t.merge(ctl[["slice", "edge", "p_nml_15"]].rename(columns={"edge": "ctl_edge", "p_nml_15": "ctl_nml_15"}),
                    on="slice", how="left")
        out.append(t)
    return pd.concat(out, ignore_index=True)


def tod_profile(ev: pd.DataFrame) -> pd.DataFrame:
    """Section 21 metrics by 15-minute bucket, all backside signals pooled
    (one row per signal bar) and the control sample for comparison."""
    sig = ev[ev["sig"] != "B00"].drop_duplicates(["sid", "idx"]).sort_values(["sid", "mod"]).copy()
    sig["gap"] = sig.groupby("sid")["mod"].diff()
    out = []
    for name, df in (("signals", sig), ("control", ev[ev["sig"] == "B00"])):
        g = df.groupby("tod15")
        t = pd.DataFrame({
            "n": g.size(),
            "p_continue_15": g["nml_15"].mean(),
            "p_worked": g["worked"].mean(),
            "p_failed": g["failed"].mean(),
            "edge": g["edge"].mean(),
            "dn_exc_30": g["mfe_30"].mean(),
            "up_exc_30": g["mae_30"].mean(),
            "dn_exc_30_med": g["mfe_30"].median(),
            "up_exc_30_med": g["mae_30"].median(),
            "p_rec_vwap_30": g["rec_vwap_30"].mean(),
            "p_rec_ema20_30": g["rec_ema20_30"].mean(),
            "p_new_hod_30": g["nhod_30"].mean(),
            "p_new_hod_eod": g["nhod_eod"].mean(),
            "gap_min": g["gap"].mean() if "gap" in df else np.nan,
        }).reset_index()
        t["sample"] = name
        out.append(t)
    t = pd.concat(out, ignore_index=True)
    t["bucket"] = t["tod15"].map(tod15_label)
    return t


# ------------------------------------------------------------------ strategies
def daily_drawdown(tr: pd.DataFrame) -> float:
    if not len(tr):
        return math.nan
    x = tr.sort_values(["date", "entry_mod"])["r"].to_numpy()
    eq = np.cumsum(x)
    return float((eq - np.maximum.accumulate(np.concatenate([[0], eq]))[1:]).min())


def trade_summary(tr: pd.DataFrame, reps: int, seed: int) -> dict:
    if not len(tr):
        return {"n": 0}
    r = tr["r"].to_numpy()
    lo, hi = cluster_bootstrap(r, tr["date"].to_numpy(), reps, seed)
    wins = r > 0
    day = tr.groupby("date")["r"].sum()
    return {"n": int(len(r)), "ev": float(r.mean()), "ev_lo": lo, "ev_hi": hi,
            "median": float(np.median(r)), "win": float(wins.mean()),
            "pf": float(r[wins].sum() / max(-r[~wins].sum(), 1e-12)),
            "avg_win": float(r[wins].mean()) if wins.any() else math.nan,
            "avg_loss": float(r[~wins].mean()) if (~wins).any() else math.nan,
            "total": float(r.sum()), "days": int(day.size), "pos_days": float((day > 0).mean()),
            "day_sharpe": float(day.mean() / day.std(ddof=1)) if day.size > 2 and day.std(ddof=1) > 0 else math.nan,
            "max_dd": daily_drawdown(tr), "ev_low": float(tr["r_low"].mean()), "ev_high": float(tr["r_high"].mean()),
            "ssr_share": float(tr["ssr"].mean()), "halt_share": float(tr["halt_in_trade"].mean()),
            "ev_halt": float(tr.loc[tr["halt_in_trade"], "r"].mean()) if tr["halt_in_trade"].any() else math.nan,
            "sessions": int(tr["sid"].nunique()), "tickers": int(tr["ticker"].nunique()),
            "hold_med": float((tr["exit_mod"] - tr["entry_mod"]).median())}


def cutoff_tables(normal: pd.DataFrame, flats: dict[str, pd.DataFrame], cids: list[int], cfg: Config,
                  test_from: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reps, seed = cfg.val.bootstrap_reps // 2, cfg.val.seed
    rows, heat, marg = [], [], []
    for cid in cids:
        tr = normal[normal["cid"] == cid]
        if not len(tr):
            continue
        for cut in cfg.strat.cutoffs:
            cm = hm(cut)
            a = tr[tr["entry_mod"] < cm]
            for period, sub in (("all", a), ("train", a[a["date"] < test_from]), ("test", a[a["date"] >= test_from])):
                s = trade_summary(sub, reps, seed)
                rows.append({"cid": cid, "cutoff": cut, "mode": "stop new entries", "period": period, **s})
            if cut != "16:00":
                f = flats.get(f"flat_{cut}")
                if f is not None and len(f):
                    b = f[(f["cid"] == cid) & (f["entry_mod"] < cm)]
                    for period, sub in (("all", b), ("train", b[b["date"] < test_from]), ("test", b[b["date"] >= test_from])):
                        s = trade_summary(sub, reps, seed)
                        rows.append({"cid": cid, "cutoff": cut, "mode": "stop entries + cover at cutoff",
                                     "period": period, **s})
        for st in cfg.strat.start_times:
            for cut in cfg.strat.cutoffs:
                sub = tr[(tr["entry_mod"] >= hm(st)) & (tr["entry_mod"] < hm(cut))]
                if hm(st) >= hm(cut):
                    continue
                heat.append({"cid": cid, "start": st, "cutoff": cut, "n": len(sub),
                             "ev": float(sub["r"].mean()) if len(sub) else math.nan,
                             "win": float((sub["r"] > 0).mean()) if len(sub) else math.nan,
                             "total": float(sub["r"].sum())})
        b15 = tod15(tr["entry_mod"].to_numpy())
        for b in sorted(set(b15)):
            sub = tr[b15 == b]
            lo, hi = cluster_bootstrap(sub["r"].to_numpy(), sub["date"].to_numpy(), reps, seed)
            marg.append({"cid": cid, "tod15": int(b), "bucket": tod15_label(int(b)), "n": len(sub),
                         "ev": float(sub["r"].mean()), "ev_lo": lo, "ev_hi": hi,
                         "win": float((sub["r"] > 0).mean())})
    return pd.DataFrame(rows), pd.DataFrame(heat), pd.DataFrame(marg)


def marginals(ct: pd.DataFrame, grid) -> pd.DataFrame:
    """Average EV of every parameter value across all other choices - shows
    which choices matter regardless of how the rest is tuned."""
    ok = ct[ct["n"] >= grid.cfg.strat.min_trades].copy()
    if not len(ok):
        return pd.DataFrame()
    dec = pd.DataFrame([grid.decode(int(c)) for c in ok["cid"]])
    ok = pd.concat([ok.reset_index(drop=True), dec], axis=1)
    out = []
    for dim in ("sig", "timing", "stop", "target", "hold"):
        g = ok.groupby(dim)
        t = pd.DataFrame({"mean_ev": g["ev"].mean(), "median_ev": g["ev"].median(),
                          "share_positive": g["ev"].apply(lambda x: float((x > 0).mean())),
                          "best_ev": g["ev"].max(), "configs": g.size(),
                          "mean_win": g["win"].mean()}).reset_index().rename(columns={dim: "value"})
        t["dimension"] = dim
        out.append(t)
    return pd.concat(out, ignore_index=True)


def stage_tables(segs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not len(segs):
        return pd.DataFrame(), pd.DataFrame()
    s = segs[segs["stage"] > 0]
    rows = []
    for k, g in s.groupby("stage"):
        nxt = g["next"]
        has_next = nxt >= 0
        if k <= 3:
            rev = nxt.isin([kk for kk in (1, 2, 3) if kk < k])
        elif k == 4:
            rev = nxt.isin([1, 2, 3])
        elif k in (5, 6, 7):
            rev = nxt == 8
        else:
            rev = nxt.isin([4, 5])
        rows.append({
            "stage": int(k), "name": STAGES[int(k)], "segments": int(len(g)), "sessions": int(g["sid"].nunique()),
            "start_med": float(g["start_mod"].median()), "end_med": float(g["end_mod"].median()),
            "dur_med": float(g["dur_min"].median()), "dur_mean": float(g["dur_min"].mean()),
            "range_med": float(g["range_pct"].median()), "ret_med": float(g["ret_pct"].median()),
            "rel_vol_med": float(g["rel_vol"].median()), "rg_vol_med": float(g["rg_vol"].median()),
            "below_vwap": float(g["start_below_vwap"].mean()), "below_e20": float(g["start_below_e20"].mean()),
            "e9_below_e20": float(g["start_e9_below_e20"].mean()), "e20_falling": float(g["start_e20_falling"].mean()),
            "p_next": float((nxt[has_next] == k + 1).mean()) if k < 8 and has_next.any() else math.nan,
            "p_reverse": float(rev[has_next].mean()) if has_next.any() else math.nan,
            "p_end_of_day": float((~has_next).mean()),
            "fwd_nml_15": float(g["nml_15"].mean()), "fwd_30": float(g["fwd_30"].mean()),
            "fwd_mfe_30": float(g["mfe_30"].mean()), "fwd_mae_30": float(g["mae_30"].mean()),
            "fwd_edge": float(((g["fp_atr"] == 0).astype(float) - (g["fp_atr"] == 1).astype(float)).mean()),
            "fwd_nhod_eod": float(g["nhod_eod"].mean()), "reach5_30": float(g["reach5_30"].mean()),
            "reach8_eod": float(g["reach8_eod"].mean()),
        })
    tab = pd.DataFrame(rows)
    tm = pd.crosstab(s["stage"], s["next"], normalize="index")
    tm = tm.reset_index().melt(id_vars="stage", var_name="next", value_name="p")
    return tab, tm


def pooled_by(ev: pd.DataFrame, col: str, metric: str, reps: int, seed: int) -> pd.DataFrame:
    """All backside signals pooled (one row per signal bar) vs the control
    sample, split by `col`, with a joint session bootstrap for the lift."""
    sig = ev[ev["sig"] != "B00"].drop_duplicates(["sid", "idx"])
    ctl = ev[ev["sig"] == "B00"]
    codes, uniq = pd.factorize(pd.concat([sig["sid"], ctl["sid"]]))
    cs, cc = codes[:len(sig)], codes[len(sig):]
    rng = np.random.default_rng(seed)
    W = rng.poisson(1.0, size=(reps, len(uniq))).astype(np.float32)
    rows = []
    for val in sorted(set(sig[col].dropna().astype(str)) | set(ctl[col].dropna().astype(str))):
        ms = (sig[col].astype(str) == val).to_numpy() & sig[metric].notna().to_numpy()
        mc = (ctl[col].astype(str) == val).to_numpy() & ctl[metric].notna().to_numpy()
        if ms.sum() < 5 or mc.sum() < 5:
            continue
        def agg(codes_, x):
            cnt = np.bincount(codes_, minlength=len(uniq)).astype(np.float32)
            sm = np.bincount(codes_, weights=x, minlength=len(uniq)).astype(np.float32)
            return cnt, sm
        c1, s1 = agg(cs[ms], sig[metric].to_numpy(float)[ms])
        c2, s2 = agg(cc[mc], ctl[metric].to_numpy(float)[mc])
        with np.errstate(invalid="ignore", divide="ignore"):
            b = (W @ s1) / (W @ c1) - (W @ s2) / (W @ c2)
        lo, hi = np.nanpercentile(b, [2.5, 97.5])
        rows.append({"slice": val, "n": int(ms.sum()), "sessions": int(sig.loc[ms, "sid"].nunique()),
                     "value": float(s1.sum() / c1.sum()), "control": float(s2.sum() / c2.sum()),
                     "lift": float(s1.sum() / c1.sum() - s2.sum() / c2.sum()), "lo": float(lo), "hi": float(hi)})
    return pd.DataFrame(rows)
