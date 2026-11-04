"""What happened AFTER each signal (labels only - never fed back into signals).

Reference price = the signal bar's close. Windows run from the next bar to
min(signal time + H minutes, regular-session close).
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .indicators import SessionData

HZ = ("5", "15", "30", "60", "eod")


def _first(mask: np.ndarray) -> int:
    if mask.size and mask.any():
        return int(np.argmax(mask))
    return -1


def label(s: SessionData, events: list[dict], cfg: Config) -> None:
    """Adds outcome fields to each event dict in place."""
    n = len(s)
    if not events:
        return
    c, h, l, mod = s.c, s.h, s.l, s.mod
    A = s.arr
    lv = {"ema9": A["ema9"], "ema20": A["ema20"], "vwap": A["vwap"]}
    final_hod = A["hod"][-1]
    oc = cfg.out
    horizons = list(oc.horizons_min)
    for ev in events:
        e = ev["idx"]
        P = c[e]
        j0 = e + 1
        ev["after_true_hod"] = bool(A["hod"][e] >= final_hod)      # hindsight label
        ev["has_future"] = j0 < n
        if j0 >= n:
            continue
        ends = {str(H): int(np.searchsorted(mod, mod[e] + H, side="right") - 1) for H in horizons}
        ends["eod"] = n - 1
        for key, jH in ends.items():
            if jH < j0:
                for f in ("mfe", "mae", "nml", "nlod", "nhod", "fwd"):
                    ev[f"{f}_{key}"] = np.nan
                continue
            lo = l[j0:jH + 1].min()
            hi = h[j0:jH + 1].max()
            ev[f"mfe_{key}"] = (P - lo) / P
            ev[f"mae_{key}"] = (hi - P) / P
            ev[f"nml_{key}"] = float(lo < ev["move_low"])
            ev[f"nlod_{key}"] = float(lo < ev["lod"])
            ev[f"nhod_{key}"] = float(hi > ev["hod"])
            ev[f"fwd_{key}"] = (P - c[jH]) / P
        # extra downside thresholds
        for x in oc.down_moves:
            tag = int(round(x * 100))
            ev[f"dn{tag}_eod"] = float(ev["mfe_eod"] >= x)
            ev[f"dn{tag}_60"] = float(ev.get("mfe_60", np.nan) >= x) if np.isfinite(ev.get("mfe_60", np.nan)) else np.nan
        # time to first new low of the move, and to the low within 60 min
        k = _first(l[j0:] < ev["move_low"])
        ev["t_nml"] = float(mod[j0 + k] - mod[e]) if k >= 0 else np.nan
        j60 = ends["60"]
        if j60 >= j0:
            seg = l[j0:j60 + 1]
            ev["t_low60"] = float(mod[j0 + int(np.argmin(seg))] - mod[e])
        else:
            ev["t_low60"] = np.nan
        # reclaims (only meaningful if price was below the level at the signal)
        for name, arr in lv.items():
            below = c[e] < arr[e]
            ev[f"below_{name}"] = bool(below)
            for key in ("15", "30", "eod"):
                jH = ends[key]
                if not below or jH < j0:
                    ev[f"rec_{name}_{key}"] = np.nan
                else:
                    ev[f"rec_{name}_{key}"] = float((c[j0:jH + 1] > arr[j0:jH + 1]).any())
        # first passage: does price move against the short by X before it moves
        # in favour by X? A bar touching both counts as a failure (conservative).
        jF = int(np.searchsorted(mod, mod[e] + oc.first_passage_horizon_min, side="right") - 1)
        for tag, dist in (("atr", A["atr"][e] * oc.first_passage_atr), ("pct", P * oc.first_passage_pct)):
            if jF < j0:
                ev[f"fp_{tag}"] = np.nan
                continue
            up = _first(h[j0:jF + 1] >= P + dist)
            dn = _first(l[j0:jF + 1] <= P - dist)
            if up < 0 and dn < 0:
                ev[f"fp_{tag}"] = np.nan          # unresolved within the horizon
            elif dn < 0 or (up >= 0 and up <= dn):
                ev[f"fp_{tag}"] = 1.0             # failed
            else:
                ev[f"fp_{tag}"] = 0.0             # worked
