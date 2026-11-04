"""Frontside / backside stage classifier (causal state machine).

Each bar gets a stage using only information known at its close; confirmed
swing highs count only from their confirmation bar. Stage 4 is the EARLY
WARNING state, stage 5 the CONFIRMED backside. A stage's start time is the bar
at which its conditions were first observable.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .indicators import SessionData

STAGES = {
    0: "No move yet",
    1: "Initial momentum",
    2: "Strong frontside",
    3: "Weakening frontside",
    4: "Potential backside transition",
    5: "Confirmed backside",
    6: "Extended backside",
    7: "Possible backside exhaustion",
    8: "Reversal / renewed frontside",
}

STAGE_RULES = {
    1: "First push of the day: a new high of day on a volume spike (>= 2.5x the 20-bar median), "
       "above VWAP and the 9 EMA, at least 10% above the day's low so far. Lasts up to 10 minutes.",
    2: "Above VWAP and the 20 EMA, 9 EMA above the 20 EMA, 20 EMA rising, new high within 10 min.",
    3: "Still above VWAP and the 20 EMA, but a close below the 9 EMA, a falling 9 EMA, or no new "
       "high for 10+ minutes.",
    4: "Early warning: any of close < 20 EMA, close < VWAP, 9 EMA < 20 EMA, or a confirmed lower "
       "high - but not all five backside conditions.",
    5: "All five: close < VWAP, close < 20 EMA, 9 EMA < 20 EMA, 20 EMA falling, confirmed lower high.",
    6: "Confirmed for 15+ minutes, 3+ ATR below VWAP, and a new low of the move in the last 5 bars.",
    7: "Inside the backside: 10+ bars without a new low and a close above the 9 EMA, or a red "
       "volume climax (3x median) followed by 3+ bars without a new low, or a close back above "
       "the 20 EMA while still below VWAP.",
    8: "From a backside stage: close above VWAP and the 20 EMA with the 9 EMA above the 20 EMA, "
       "or a new high of day.",
}


def classify(s: SessionData, perbar: dict, cfg: Config) -> np.ndarray:
    n = len(s)
    A = s.arr
    c, h, l, v, mod = s.c, s.h, s.l, s.v, s.mod
    e9, e20, vw = A["ema9"], A["ema20"], A["vwap"]
    atr, hod, new_hod = A["atr"], A["hod"], A["new_hod"]
    med = np.nan_to_num(A["vol_med20"], nan=np.inf)
    red = A["red"]
    lh_price = perbar["lh_price"]
    move_low = A["move_low"]

    stage = np.zeros(n, dtype=np.int8)
    momentum_start = -1
    last_newlow = 0
    last_climax = -100
    bs_since = -1
    p = 0
    for t in range(n):
        if t == 0 or new_hod[t] or move_low[t] < move_low[t - 1]:
            last_newlow = t
        if red[t] and v[t] >= 3.0 * med[t]:
            last_climax = t
        e20_dn = t >= 1 and e20[t] < e20[t - 1]
        e20_up = t >= 1 and e20[t] > e20[t - 1]
        e9_dn = t >= 1 and e9[t] < e9[t - 1]
        lh = not np.isnan(lh_price[t])
        b = (c[t] < vw[t], c[t] < e20[t], e9[t] < e20[t], e20_dn, lh)
        nb = sum(b)
        front_all = c[t] > vw[t] and c[t] > e20[t] and e9[t] > e20[t] and e20_up
        mins_hod = A["mins_since_hod"][t]
        since_low = t - last_newlow

        if momentum_start < 0 and new_hod[t] and c[t] > vw[t] and c[t] > e9[t] \
                and v[t] >= 2.5 * med[t] and A["runup"][t] >= 0.10:
            momentum_start = t

        if new_hod[t]:
            if p in (4, 5, 6, 7):
                st = 8
            elif momentum_start == t or (p == 1 and mod[t] - mod[momentum_start] <= 10):
                st = 1
            elif momentum_start < 0 and p == 0:
                st = 0
            else:
                st = 2 if front_all else 3
        elif p in (5, 6, 7):
            if c[t] > vw[t] and c[t] > e20[t] and e9[t] > e20[t]:
                st = 8
            elif nb == 5 and since_low <= 1 and (p != 7 or c[t] < e9[t]):
                stretched = (vw[t] - c[t]) / atr[t] >= 3.0
                st = 6 if (stretched and bs_since >= 0 and mod[t] - mod[bs_since] >= 15) else 5
            elif (since_low >= 10 and c[t] > e9[t]) or (t - last_climax <= 5 and since_low >= 3) \
                    or (c[t] > e20[t] and c[t] < vw[t]):
                st = 7
            elif p == 6 and since_low > 5:
                st = 5
            else:
                st = p
        elif nb == 5:
            st = 5
        elif nb >= 1:
            # before the first push of the day, a single weak condition is noise
            st = 4 if (momentum_start >= 0 or p in (4, 8)) else 0
        else:
            if momentum_start < 0:
                st = 0
            elif p == 1 and mod[t] - mod[momentum_start] <= 10 and c[t] > e9[t]:
                st = 1
            elif front_all and mins_hod <= 10 and not (c[t] < e9[t] or e9_dn):
                st = 2
            elif p == 8 and not front_all:
                st = 8
            else:
                st = 3
        if st in (5, 6, 7) and p not in (5, 6, 7):
            bs_since = t
        if st not in (5, 6, 7):
            bs_since = -1
        stage[t] = st
        p = st
    return stage


def segments(s: SessionData, stage: np.ndarray) -> list[dict]:
    """Contiguous runs of one stage, with what was observable inside them."""
    out = []
    n = len(stage)
    if not n:
        return out
    A = s.arr
    cum_mean_vol = A["cumvol"] / np.arange(1, n + 1)
    starts = np.flatnonzero(np.concatenate([[True], stage[1:] != stage[:-1]]))
    ends = np.concatenate([starts[1:] - 1, [n - 1]])
    for k, (a, b) in enumerate(zip(starts, ends)):
        seg_v = s.v[a:b + 1]
        red = A["red"][a:b + 1]
        green = A["green"][a:b + 1]
        rv, gv = seg_v[red].sum(), seg_v[green].sum()
        nxt = int(stage[b + 1]) if b + 1 < n else -1
        out.append({
            "sid": s.sid, "ticker": s.ticker, "date": s.date, "seg": k,
            "stage": int(stage[a]), "next": nxt, "a": int(a), "b": int(b),
            "start_mod": int(s.mod[a]), "end_mod": int(s.mod[b]),
            "dur_min": int(s.mod[b] - s.mod[a] + 1),
            "p_start": float(s.c[a]), "p_end": float(s.c[b]),
            "hi": float(s.h[a:b + 1].max()), "lo": float(s.l[a:b + 1].min()),
            "range_pct": float(s.h[a:b + 1].max() / max(s.l[a:b + 1].min(), 1e-9) - 1.0),
            "ret_pct": float(s.c[b] / s.c[a] - 1.0),
            "rel_vol": float(seg_v.mean() / max(cum_mean_vol[a], 1e-9)),
            "rg_vol": float(rv / gv) if gv > 0 else float("nan"),
            "start_below_vwap": bool(s.c[a] < A["vwap"][a]),
            "start_below_e20": bool(s.c[a] < A["ema20"][a]),
            "start_e9_below_e20": bool(A["ema9"][a] < A["ema20"][a]),
            "start_e20_falling": bool(a >= 1 and A["ema20"][a] < A["ema20"][a - 1]),
        })
    return out
