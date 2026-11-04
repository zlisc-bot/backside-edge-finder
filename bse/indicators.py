"""Turn one session's raw minute bars into causal indicator arrays.

Every array element t uses only bars 0..t (bar t's close is the latest
information), except pivots, which are stored at the bar where they become
KNOWN (pivot bar + `pivot_right`). Nothing here peeks at later bars.

Index 0 is the first bar of the session at/after 04:00 ET; the prior day's
bars are used only to warm up the EMAs and ATR and are then dropped.
The arrays stop at the regular-session close (16:00, or 13:00 on early closes).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .calendar import rth_end_minute
from .config import TZ, Config, hm


@dataclass
class SessionData:
    sid: str
    ticker: str
    date: str
    meta: dict
    ts: np.ndarray
    mod: np.ndarray          # minute of day, ET
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    vw: np.ndarray
    arr: dict = field(default_factory=dict)   # indicator arrays, same length

    def __len__(self):
        return len(self.c)

    def __getattr__(self, k):
        arr = self.__dict__.get("arr")
        if arr is not None and k in arr:
            return arr[k]
        raise AttributeError(k)


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def _rolling_sum_prev(x: np.ndarray, w: int) -> np.ndarray:
    """sum of x[t-w .. t-1] (excludes t)."""
    cs = np.concatenate([[0.0], np.cumsum(x, dtype=float)])
    t = np.arange(len(x))
    lo = np.maximum(t - w, 0)
    return cs[t] - cs[lo]


def _rolling_sum_incl(x: np.ndarray, w: int) -> np.ndarray:
    """sum of x[t-w+1 .. t]."""
    cs = np.concatenate([[0.0], np.cumsum(x, dtype=float)])
    t = np.arange(len(x))
    lo = np.maximum(t - w + 1, 0)
    return cs[t + 1] - cs[lo]


def _pivots(h: np.ndarray, l: np.ndarray, L: int, R: int):
    """Fractal swing points. Returns arrays indexed by CONFIRMATION bar:
    ph_conf[j] = high of pivot bar j-R if bar j-R is a swing high, else nan."""
    n = len(h)
    ph_conf = np.full(n, np.nan)
    pl_conf = np.full(n, np.nan)
    if n < L + R + 1:
        return ph_conf, pl_conf
    from numpy.lib.stride_tricks import sliding_window_view as swv
    idx = np.arange(L, n - R)
    lw_h = swv(h, L)[idx - L].max(axis=1)          # h[i-L .. i-1]
    rw_h = swv(h, R)[idx + 1].max(axis=1)          # h[i+1 .. i+R]
    lw_l = swv(l, L)[idx - L].min(axis=1)
    rw_l = swv(l, R)[idx + 1].min(axis=1)
    is_ph = (h[idx] > lw_h) & (h[idx] >= rw_h)
    is_pl = (l[idx] < lw_l) & (l[idx] <= rw_l)
    ph_conf[idx[is_ph] + R] = h[idx[is_ph]]
    pl_conf[idx[is_pl] + R] = l[idx[is_pl]]
    return ph_conf, pl_conf


def prepare_session(bars: pd.DataFrame, meta: dict, cfg: Config) -> SessionData | None:
    """bars: minute bars of the session date and (optionally) the prior trading
    day, columns ts,o,h,l,c,v,vw. meta: one row of the sessions table."""
    if bars is None or not len(bars):
        return None
    bars = bars.sort_values("ts")
    et = pd.to_datetime(bars["ts"].to_numpy(), unit="ms", utc=True).tz_convert(TZ)
    dstr = et.strftime("%Y-%m-%d").to_numpy()
    mod_all = (et.hour * 60 + et.minute).to_numpy()
    d = meta["date"]
    ext0 = hm(cfg.session.ext_start)
    rth0 = hm(cfg.session.rth_start)
    rth1 = rth_end_minute(d)

    warm = dstr < d
    day = (dstr == d) & (mod_all >= ext0) & (mod_all < rth1)
    use = warm | day
    if day.sum() < 30:
        return None
    b = bars[use]
    c_all = b["c"].to_numpy(float)
    h_all = b["h"].to_numpy(float)
    l_all = b["l"].to_numpy(float)
    ic = cfg.ind
    ema9_all = _ema(c_all, ic.ema_fast)
    ema20_all = _ema(c_all, ic.ema_slow)
    prev_c = np.concatenate([[c_all[0]], c_all[:-1]])
    tr = np.maximum(h_all - l_all, np.maximum(np.abs(h_all - prev_c), np.abs(l_all - prev_c)))
    atr_all = pd.Series(tr).ewm(alpha=1.0 / ic.atr_len, adjust=False).mean().to_numpy()

    k = int(warm[use].sum())       # first day bar
    sl = slice(k, None)
    s = SessionData(
        sid=meta["session_id"], ticker=meta["ticker"], date=d, meta=dict(meta),
        ts=b["ts"].to_numpy()[sl], mod=mod_all[use][sl].astype(int),
        o=b["o"].to_numpy(float)[sl], h=h_all[sl], l=l_all[sl], c=c_all[sl],
        v=b["v"].to_numpy(float)[sl], vw=b["vw"].to_numpy(float)[sl])
    n = len(s.c)
    A = s.arr
    A["ema9"] = ema9_all[sl]
    A["ema20"] = ema20_all[sl]
    A["atr"] = np.maximum(atr_all[sl], 1e-9)
    A["warm_bars"] = np.full(n, k)

    # VWAP -------------------------------------------------------------------
    pv = np.where(np.isfinite(s.vw), s.vw, (s.h + s.l + s.c) / 3.0) * s.v
    cv = np.cumsum(s.v)
    A["cumvol"] = cv
    vwap_ext = np.cumsum(pv) / np.maximum(cv, 1e-9)
    rth = s.mod >= rth0
    pv_r = np.where(rth, pv, 0.0)
    v_r = np.where(rth, s.v, 0.0)
    vwap_rth = np.where(rth, np.cumsum(pv_r) / np.maximum(np.cumsum(v_r), 1e-9), np.nan)
    A["vwap_ext"] = vwap_ext
    A["vwap_rth"] = vwap_rth
    if cfg.session.vwap_anchor == "rth":
        # before 09:30 fall back to the extended-hours VWAP
        A["vwap"] = np.where(rth, vwap_rth, vwap_ext)
    else:
        A["vwap"] = vwap_ext
    A["is_rth"] = rth

    # running high / low of day ---------------------------------------------
    hod = np.maximum.accumulate(s.h)
    prev_hod = np.concatenate([[-np.inf], hod[:-1]])
    new_hod = s.h > prev_hod
    idx = np.arange(n)
    hod_idx = np.maximum.accumulate(np.where(new_hod, idx, 0))
    A["hod"], A["hod_idx"], A["new_hod"] = hod, hod_idx, new_hod
    A["lod"] = np.minimum.accumulate(s.l)
    # lowest low since the bar that set the current high ("low of the move")
    A["move_low"] = pd.Series(s.l).groupby(hod_idx).cummin().to_numpy()
    low_cum = np.minimum.accumulate(s.l)
    A["low_before_hod"] = low_cum[hod_idx]
    pc = float(meta.get("pc", np.nan))
    A["gain_pc"] = hod / pc - 1.0 if np.isfinite(pc) and pc > 0 else np.full(n, np.nan)
    A["runup"] = hod / np.maximum(A["low_before_hod"], 1e-9) - 1.0
    A["mins_since_hod"] = s.mod - s.mod[hod_idx]

    # swing points (stored at confirmation bar) ------------------------------
    A["ph_conf"], A["pl_conf"] = _pivots(s.h, s.l, ic.pivot_left, ic.pivot_right)

    # volume features ----------------------------------------------------------
    red = s.c < s.o
    green = s.c > s.o
    A["red"], A["green"] = red, green
    w = ic.vol_lookback
    rv_prev = _rolling_sum_prev(np.where(red, s.v, 0.0), w)
    rc_prev = _rolling_sum_prev(red.astype(float), w)
    A["sell_expansion"] = red & (rc_prev >= 2) & (s.v >= ic.sell_expansion_mult * rv_prev / np.maximum(rc_prev, 1))
    gv_incl = _rolling_sum_incl(np.where(green, s.v, 0.0), w)
    gc_incl = _rolling_sum_incl(green.astype(float), w)
    av_incl = _rolling_sum_incl(s.v, w) / np.minimum(idx + 1, w)
    A["green_vol_avg"] = np.where(gc_incl > 0, gv_incl / np.maximum(gc_incl, 1), av_incl)
    A["vol_med20"] = pd.Series(s.v).shift(1).rolling(20, min_periods=5).median().to_numpy()
    rv10 = _rolling_sum_incl(np.where(red, s.v, 0.0), w)
    gv10 = _rolling_sum_incl(np.where(green, s.v, 0.0), w)
    A["rg_vol_ratio"] = rv10 / np.maximum(gv10, 1.0)

    # spread estimate (Abdi & Ranaldo 2017, causal rolling window) -------------
    lc = np.log(np.maximum(s.c, 1e-9))
    eta = (np.log(np.maximum(s.h, 1e-9)) + np.log(np.maximum(s.l, 1e-9))) / 2.0
    term = np.full(n, np.nan)
    term[1:] = (lc[:-1] - eta[:-1]) * (lc[:-1] - eta[1:])
    m = pd.Series(term).rolling(ic.spread_window, min_periods=10).mean().to_numpy()
    A["spread_ar"] = np.sqrt(np.clip(np.nan_to_num(m, nan=0.0) * 4.0, 0.0, None))

    # halts: >= 5 min without a print during RTH (LULD pauses last 5 minutes)
    gap = np.concatenate([[0], np.diff(s.mod)])
    prev_rth = np.concatenate([[False], rth[:-1]])
    A["after_gap"] = (gap >= ic.halt_gap_min) & prev_rth & rth
    A["gap_min"] = gap

    # Rule 201 (short-sale restriction) -----------------------------------------
    carry = bool(meta.get("ssr_carry", False))
    trig = 0.9 * pc if np.isfinite(pc) else -np.inf
    A["ssr"] = np.full(n, carry) | (low_cum <= trig)

    # what a trader could see in real time --------------------------------------
    A["known_sub1"] = (pc < 1.0 if np.isfinite(pc) else False) | (low_cum < 1.0)
    A["tick"] = np.where(s.c < 1.0, cfg.session.tick, cfg.session.tick_above_1)
    return s
