"""Backside signal library - strictly causal.

Each signal is evaluated at a bar's CLOSE using only information available at
that moment. Swing highs/lows become usable only when confirmed
(`pivot_right` bars later) and are timestamped at the confirmation bar.

"First ... after the high": every new running high of day starts a new
*epoch*. Inside an epoch each signal fires at most once. A trader cannot know
in real time whether the current high is the high of the day, so frontside
pullbacks that later resolve higher are counted too - exactly as they would
have looked live. Whether a signal happened after the day's eventual high is
recorded separately as a hindsight label and never used for detection.
"""
from __future__ import annotations

import math

import numpy as np

from .config import Config
from .indicators import SessionData

LEVELS = ("ema9", "ema20", "vwap")

# code -> (name, group, reference level used by "failed reclaim" entries)
SIGNALS = {
    "S01": ("First 1-min close below the 9 EMA", "individual", "ema9"),
    "S02": ("First 1-min close below the 20 EMA", "individual", "ema20"),
    "S03": ("First 1-min close below VWAP", "individual", "vwap"),
    "S04": ("9 EMA crosses below the 20 EMA", "individual", "ema20"),
    "S05": ("20 EMA turns from rising to falling", "individual", "ema20"),
    "S06": ("First lower high after the high", "individual", "ema9"),
    "S07": ("First lower low after the high", "individual", "ema9"),
    "S08": ("First failed 9 EMA reclaim", "individual", "ema9"),
    "S09": ("First failed 20 EMA reclaim", "individual", "ema20"),
    "S10": ("First failed VWAP reclaim", "individual", "vwap"),
    "S11": ("Volume expanding on red candles", "individual", "ema9"),
    "S12": ("Bounce on fading volume", "individual", "ema9"),
    "C01": ("Loses 20 EMA while already below VWAP", "combo", "ema20"),
    "C02": ("Loses 9 EMA, then fails to reclaim it within 10 min", "combo", "ema9"),
    "C03": ("Loses 20 EMA, then makes a lower high", "combo", "ema20"),
    "C04": ("Loses VWAP, then its first reclaim attempt fails", "combo", "vwap"),
    "C05": ("9 EMA crosses below 20 EMA with both below VWAP", "combo", "ema20"),
    "C06": ("Lower high, then loses 20 EMA, then lower low", "combo", "ema20"),
    "C07": ("Loses 20 EMA while 9 and 20 EMA are both falling", "combo", "ema20"),
    "C08": ("Loses VWAP while 20 EMA falls and selling volume expands", "combo", "vwap"),
    "C09": ("Lower high after losing 20 EMA, while below VWAP", "combo", "ema20"),
    "B00": ("Control: every 10th bar after a high (no signal)", "baseline", "ema9"),
}

DEFINITIONS = {
    "S01": "In the current high-of-day epoch, the first bar whose close is below the 9 EMA.",
    "S02": "In the current epoch, the first bar whose close is below the 20 EMA.",
    "S03": "In the current epoch, the first bar whose close is below VWAP.",
    "S04": "First bar in the epoch where the 9 EMA moves from >= the 20 EMA to below it.",
    "S05": "First bar in the epoch where the 20 EMA declines after rising or flat on the prior bar.",
    "S06": "A swing high (2 bars each side) formed after the high and below it; fires when the "
           "swing is confirmed, 2 bars after its peak.",
    "S07": "First bar whose low breaks the most recent confirmed swing low formed after the high.",
    "S08": "After a close below the 9 EMA: a bar that trades up through the 9 EMA but closes "
           "below it, or a close back above that fails within 2 bars. First one in the epoch.",
    "S09": "Same as #8 for the 20 EMA.",
    "S10": "Same as #8 for VWAP.",
    "S11": "A red candle whose volume is at least 1.5x the average of the red candles in the "
           "prior 10 bars (2+ red candles required).",
    "S12": "After at least one red candle, a green candle with volume at most 0.5x the average "
           "green-candle volume of the 10 bars into the high.",
    "C01": "The 20 EMA loss (close moves from above to below the 20 EMA) happens with price "
           "already below VWAP.",
    "C02": "A failed 9 EMA reclaim no more than 10 minutes after the first 9 EMA loss.",
    "C03": "A confirmed lower high whose peak forms after the first close below the 20 EMA.",
    "C04": "Price closes below VWAP; the FIRST attempt to reclaim it fails. If the first "
           "attempt succeeds, the signal cannot fire in that epoch.",
    "C05": "The 9/20 EMA bear cross, with both EMAs below VWAP at the cross.",
    "C06": "In order: a confirmed lower high, a close below the 20 EMA at or after that "
           "confirmation, then a break of the swing low that preceded the lower high.",
    "C07": "A 20 EMA loss while the 9 EMA and the 20 EMA both declined on that bar.",
    "C08": "A VWAP loss while the 20 EMA is declining and a selling-volume expansion bar "
           "(#11) occurred within the last 3 bars.",
    "C09": "Like #C03, but the close at confirmation must be below VWAP.",
    "B00": "Not a signal. Every 5th bar of the session that is not itself a new high. "
           "Shows what happens at a random moment, so each signal can be compared to chance.",
}


def detect(s: SessionData, cfg: Config):
    """Return (events, perbar). events: list of dicts. perbar: arrays used by
    the strategy engine (failed-reclaim flags, lower highs, epoch ids)."""
    n = len(s)
    K = cfg.ind.reclaim_hold_bars
    R = cfg.ind.pivot_right
    fade_mult = cfg.ind.bounce_fade_mult
    every = cfg.out.baseline_every_n_bars

    c, h, l, v, mod = s.c, s.h, s.l, s.v, s.mod
    A = s.arr
    ema9, ema20, vwap = A["ema9"], A["ema20"], A["vwap"]
    lvl = {"ema9": ema9, "ema20": ema20, "vwap": vwap}
    hod, new_hod = A["hod"], A["new_hod"]
    ph_conf, pl_conf = A["ph_conf"], A["pl_conf"]
    sell_exp, green, red = A["sell_expansion"], A["green"], A["red"]
    gva = A["green_vol_avg"]

    fr = {k: np.zeros(n, bool) for k in LEVELS}       # every failed reclaim, any epoch
    lh_flag = np.zeros(n, bool)                         # lower high confirmed at bar
    lh_price = np.full(n, np.nan)                       # latest lower high in epoch (as of t)
    lh_peak_idx = np.full(n, -1)
    epoch_id = np.zeros(n, int)
    events = []

    ep = -1
    for t in range(n):
        if new_hod[t]:
            ep += 1
            t0 = t
            fired = set()
            st = {k: 0 for k in LEVELS}          # 0 above, 1 below, 2 reclaiming
            rc = {k: 0 for k in LEVELS}          # bars held above while reclaiming
            attempts = {k: 0 for k in LEVELS}
            first_loss = {k: -1 for k in LEVELS}
            last_pl = math.nan
            last_pl_idx = -1
            cur_lh = math.nan
            cur_lh_idx = -1
            red_seen = False
            ref_green = gva[t0]
            c06_stage, c06_ref = 0, math.nan
            sell_recent = -10
        epoch_id[t] = ep
        hit = []

        if sell_exp[t]:
            sell_recent = t

        # --- level state machines: loss / failed reclaim -------------------
        loss_now = {}
        for k in LEVELS:
            L = lvl[k][t]
            loss_now[k] = False
            if st[k] == 0:
                if c[t] < L:
                    st[k] = 1
                    loss_now[k] = True
                    if first_loss[k] < 0:
                        first_loss[k] = t
            elif st[k] == 1:
                if c[t] >= L:
                    st[k] = 2
                    rc[k] = 1
                    attempts[k] += 1
                    if rc[k] >= K:
                        st[k] = 0
                elif h[t] >= L:          # traded through, closed below: rejection
                    attempts[k] += 1
                    fr[k][t] = True
            else:  # reclaiming
                if c[t] < L:
                    st[k] = 1
                    fr[k][t] = True
                else:
                    rc[k] += 1
                    if rc[k] >= K:
                        st[k] = 0

        if first_loss["ema9"] == t:
            hit.append("S01")
        if first_loss["ema20"] == t:
            hit.append("S02")
        if first_loss["vwap"] == t:
            hit.append("S03")
        if fr["ema9"][t]:
            hit.append("S08")
            if mod[t] - mod[first_loss["ema9"]] <= 10:
                hit.append("C02")
        if fr["ema20"][t]:
            hit.append("S09")
        if fr["vwap"][t]:
            hit.append("S10")
            if attempts["vwap"] == 1:
                hit.append("C04")
        if loss_now["ema20"]:
            if c[t] < vwap[t]:
                hit.append("C01")
            if t >= 1 and ema9[t] < ema9[t - 1] and ema20[t] < ema20[t - 1]:
                hit.append("C07")
        if loss_now["vwap"] and t >= 1 and ema20[t] < ema20[t - 1] and sell_recent >= t - 2:
            hit.append("C08")

        # --- EMA cross and slope ---------------------------------------------
        if t >= 1 and ema9[t] < ema20[t] and ema9[t - 1] >= ema20[t - 1]:
            hit.append("S04")
            if ema9[t] < vwap[t] and ema20[t] < vwap[t]:
                hit.append("C05")
        if t >= 2 and ema20[t] < ema20[t - 1] and ema20[t - 1] >= ema20[t - 2]:
            hit.append("S05")

        # --- swing structure ---------------------------------------------------
        if not math.isnan(pl_conf[t]) and t - R > t0:
            last_pl, last_pl_idx = pl_conf[t], t - R
        lh_now = False
        if not math.isnan(ph_conf[t]) and t - R > t0 and ph_conf[t] < hod[t]:
            lh_now = True
            cur_lh, cur_lh_idx = ph_conf[t], t - R
            lh_flag[t] = True
            hit.append("S06")
            pk = t - R
            if first_loss["ema20"] >= 0 and pk > first_loss["ema20"]:
                hit.append("C03")
                if c[t] < vwap[t]:
                    hit.append("C09")
            if c06_stage == 0:
                # swing low between the high and this lower high's peak
                if last_pl_idx > t0 and last_pl_idx < pk:
                    c06_ref = last_pl
                else:
                    c06_ref = float(np.min(l[t0:pk + 1]))
                c06_stage = 1
        lh_price[t] = cur_lh
        lh_peak_idx[t] = cur_lh_idx
        if not math.isnan(last_pl) and l[t] < last_pl and t > last_pl_idx + R:
            hit.append("S07")
        if c06_stage == 1 and c[t] < ema20[t]:
            c06_stage = 2
        if c06_stage == 2 and l[t] < c06_ref:
            hit.append("C06")
            c06_stage = 3

        # --- volume -------------------------------------------------------------
        if t > t0:
            if sell_exp[t]:
                hit.append("S11")
            if green[t] and red_seen and v[t] <= fade_mult * ref_green:
                hit.append("S12")
            if red[t]:
                red_seen = True
            if t % every == 0:
                hit.append("B00")

        for code in hit:
            if code == "B00":
                events.append(_event(s, t, code, ep, t0))
            elif code not in fired:
                fired.add(code)
                events.append(_event(s, t, code, ep, t0))

    perbar = {"fr_ema9": fr["ema9"], "fr_ema20": fr["ema20"], "fr_vwap": fr["vwap"],
              "lh_flag": lh_flag, "lh_price": lh_price, "lh_peak_idx": lh_peak_idx,
              "epoch_id": epoch_id}
    return events, perbar


def _event(s: SessionData, t: int, code: str, ep: int, t0: int) -> dict:
    A = s.arr
    c = s.c[t]
    return {
        "sid": s.sid, "ticker": s.ticker, "date": s.date, "sig": code, "idx": t,
        "mod": int(s.mod[t]), "price": float(c), "epoch": ep, "hod_idx": int(t0),
        "hod": float(A["hod"][t]), "mins_since_hod": int(s.mod[t] - s.mod[t0]),
        "move_low": float(A["move_low"][t]), "lod": float(A["lod"][t]),
        "gain_pc": float(A["gain_pc"][t]), "runup": float(A["runup"][t]),
        "pullback": float(1.0 - c / A["hod"][t]),
        "vwap_dist": float(c / A["vwap"][t] - 1.0),
        "below_vwap": bool(c < A["vwap"][t]),
        "e9_above_e20": bool(A["ema9"][t] > A["ema20"][t]),
        "e20_falling": bool(t >= 1 and A["ema20"][t] < A["ema20"][t - 1]),
        "cumvol": float(A["cumvol"][t]), "known_sub1": bool(A["known_sub1"][t]),
        "ssr": bool(A["ssr"][t]), "atr_pct": float(A["atr"][t] / c),
        "spread_est": float(A["spread_ar"][t]), "rg_vol_ratio": float(A["rg_vol_ratio"][t]),
    }
