"""Plain-language answers generated from the statistics.

Rules the wording follows:
  * A claim is "supported" only when its 95% interval excludes zero (or the
    out-of-sample result is positive with an interval above zero).
  * Results that need hindsight are labelled as such.
  * When nothing clears the bar, the answer says so and shows the closest case.
Every number carries a `cohort` so the page can list the sessions behind it.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import analysis as an
from .config import fmt_hm, hm
from .signals import SIGNALS
from .stages import STAGES

MIN_N = 100
PHRASE = {"Pre-market": "in the pre-market", "Before 10:00": "before 10:00", "10:00-11:00": "from 10:00 to 11:00",
          "11:00-12:00": "from 11:00 to 12:00", "Afternoon": "after 12:00"}


def pct(x, d=1, sign=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    s = f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"
    return s.replace("-", "−")


def pp(x, d=1):
    """percentage points"""
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{x * 100:+.{d}f} pts".replace("-", "−")


def num(x):
    return f"{int(x):,}" if x is not None and math.isfinite(x) else "n/a"


def mins(x):
    return f"{x:.0f} min" if x is not None and math.isfinite(x) else "n/a"


def hhmm(m):
    return fmt_hm(int(m)) if m is not None and math.isfinite(m) else "n/a"


def strat_label(r) -> str:
    from .backtest import build_rules
    return (f"{SIGNALS[r['sig']][0]} → {TIMING[r['timing']]}; {STOP_TXT.get(r['stop'], r['stop'])}; "
            f"{TGT_TXT.get(r['target'], r['target'])}; max hold {HOLD_TXT(r['hold'])}; entries {r['window']}")


TIMING = {"signal_close": "short the signal candle's close", "next_open": "short the next candle's open",
          "failed_reclaim": "short the next failed reclaim", "lower_low_break": "short the break of the signal low",
          "lower_high_below_ema20": "short a lower high below the 20 EMA"}


def HOLD_TXT(h):
    return "end of day" if int(h) >= 999 else f"{int(h)} min"


def _rule_text(cfg):
    from .backtest import build_rules
    stops, targets, _ = build_rules(cfg)
    return {r.code: r.label for r in stops}, {r.code: r.label for r in targets}


STOP_TXT: dict = {}
TGT_TXT: dict = {}


def build(res: dict) -> dict:
    cfg = res["cfg"]
    s1, t1 = _rule_text(cfg)
    STOP_TXT.update({k: v.lower() for k, v in s1.items()})
    TGT_TXT.update({k: v.lower() for k, v in t1.items()})
    sig = res["signals"]
    ev = res["events"]
    reps, seed = cfg.val.bootstrap_reps // 2, cfg.val.seed
    out = []
    S = sig[(sig["group"] != "baseline") & (sig["n"] >= MIN_N)].copy()
    ctl = sig[sig["sig"] == "B00"].iloc[0] if (sig["sig"] == "B00").any() else None

    # ---------------------------------------------------------------- Q1
    q1 = {"id": "q1", "question": "What is the earliest reliable sign that a high-volume sub-$1 stock "
                                  "has moved from frontside momentum into backside weakness?"}
    if not len(S):
        q1.update(status="insufficient", answer="Not enough signals yet (each needs at least "
                                                f"{MIN_N} occurrences).", numbers=[])
    else:
        fdr = "edge_q_pos" in S
        rel = S[(S["edge_lift_lo"] > 0) & ((S["edge_q_pos"] <= 0.05) if fdr else True)]
        weak = S[(S["edge_lift_lo"] > 0) & ~S["sig"].isin(rel["sig"])]
        if len(rel):
            e = rel.sort_values(["hod_age_med", "pullback_med"]).iloc[0]
            b = rel.sort_values("edge_lift", ascending=False).iloc[0]
            txt = (f"{e['name']} is the earliest signal that beat random timing. It fires a median of "
                   f"{mins(e['hod_age_med'])} after the high of day, {pct(e['pullback_med'])} below it. "
                   f"Price fell 1 ATR before rising 1 ATR in {pct(e['p_worked'])} of {num(e['n'])} cases "
                   f"and did the opposite in {pct(e['p_failed'])}. That is {pp(e['edge_lift'])} better than "
                   f"shorting at random moments at the same times of day "
                   f"(95% interval {pp(e['edge_lift_lo'])} to {pp(e['edge_lift_hi'])}).")
            if fdr:
                txt += (f" {len(rel)} of {len(S)} signals beat random timing after correcting for testing "
                        f"{len(S)} signals at once (5% false discovery rate).")
            if b["sig"] != e["sig"]:
                txt += (f" The strongest signal overall was {b['name']} ({pp(b['edge_lift'])} vs random), "
                        f"which comes later: median {mins(b['hod_age_med'])} after the high, "
                        f"{pct(b['pullback_med'])} below it.")
            q1.update(status="supported", answer=txt, sig=e["sig"],
                      numbers=[_n("Signals", e["n"], "int", {"kind": "events", "sig": e["sig"]}),
                               _n("Worked (−1 ATR first)", e["p_worked"], "pct", {"kind": "events", "sig": e["sig"], "where": {"fp_atr": 0}}),
                               _n("Failed (+1 ATR first)", e["p_failed"], "pct", {"kind": "events", "sig": e["sig"], "where": {"fp_atr": 1}}),
                               _n("Lift vs random timing", e["edge_lift"], "pts", {"kind": "events", "sig": e["sig"]}),
                               _n("Median minutes after the high", e["hod_age_med"], "min", {"kind": "events", "sig": e["sig"]})])
        else:
            top = S.sort_values("edge_lift_lo", ascending=False).iloc[0]
            q1.update(status="not supported",
                      answer=(f"No signal beat random timing with 95% confidence. The closest was {top['name']}: "
                              f"{pp(top['edge_lift'])} vs random moments at the same times of day "
                              f"(95% interval {pp(top['edge_lift_lo'])} to {pp(top['edge_lift_hi'])}, "
                              f"{num(top['n'])} signals). On this data the common breakdown signals do not "
                              f"identify the backside better than chance."),
                      sig=top["sig"],
                      numbers=[_n("Signals", top["n"], "int", {"kind": "events", "sig": top["sig"]}),
                               _n("Lift vs random timing", top["edge_lift"], "pts", {"kind": "events", "sig": top["sig"]})])
        # simple vs combination
        ind = S[S["group"] == "individual"].sort_values("edge_lift", ascending=False)
        cmb = S[S["group"] == "combo"].sort_values("edge_lift", ascending=False)
        if len(ind) and len(cmb):
            i0, c0 = ind.iloc[0], cmb.iloc[0]
            if c0["edge_lift_lo"] > i0["edge_lift_hi"]:
                verdict = "The best combination is measurably better than the best single signal."
            elif i0["edge_lift_lo"] > c0["edge_lift_hi"]:
                verdict = "The best single signal is measurably better than the best combination."
            else:
                verdict = ("Their intervals overlap, so the data cannot tell them apart. "
                           "The simpler signal does as well.")
            out.append({"id": "simple_vs_combo", "question": "Does a combination beat a single indicator?",
                        "status": "supported" if "measurably" in verdict else "not supported",
                        "answer": (f"Best single signal: {i0['name']} ({pp(i0['edge_lift'])} vs random, "
                                   f"interval {pp(i0['edge_lift_lo'])} to {pp(i0['edge_lift_hi'])}). Best combination: "
                                   f"{c0['name']} ({pp(c0['edge_lift'])}, interval {pp(c0['edge_lift_lo'])} to "
                                   f"{pp(c0['edge_lift_hi'])}). {verdict}"),
                        "numbers": [_n(i0["name"], i0["edge_lift"], "pts", {"kind": "events", "sig": i0["sig"]}),
                                    _n(c0["name"], c0["edge_lift"], "pts", {"kind": "events", "sig": c0["sig"]})]})
    out.insert(0, q1)

    # ---------------------------------------------------------------- Q2
    q2 = {"id": "q2", "question": "When does backside scalping work best, and when should new entries stop?"}
    nums = []
    parts = []
    # Time of day for the signals that actually beat random timing; pooling every
    # signal would let the many breakdown signals that lose to chance swamp them.
    good, cg = [], pd.DataFrame()
    if len(S) and "edge_q_pos" in S:
        good = S[(S["edge_lift_lo"] > 0) & (S["edge_q_pos"] <= 0.05)].sort_values("edge_lift", ascending=False)["sig"].tolist()
    if good:
        cg = an.pooled_by(ev[ev["sig"].isin(good + ["B00"])], "coarse", "edge", reps, seed)
        if len(cg):
            names = ", ".join(SIGNALS[g][0] for g in good[:3])
            b, w = cg.sort_values("lift", ascending=False).iloc[0], cg.sort_values("lift").iloc[0]
            parts.append(f"For the signals that beat random timing ({names}), the edge was largest "
                         f"{PHRASE.get(b['slice'], b['slice'])} ({pp(b['lift'])}, interval {pp(b['lo'])} to "
                         f"{pp(b['hi'])}, {num(b['n'])} signals) and smallest {PHRASE.get(w['slice'], w['slice'])} "
                         f"({pp(w['lift'])}, interval {pp(w['lo'])} to {pp(w['hi'])}).")
            if b["slice"] == "Pre-market":
                rth = cg[cg["slice"] != "Pre-market"].sort_values("lift", ascending=False)
                if len(rth):
                    r1 = rth.iloc[0]
                    parts.append(f"Many brokers restrict pre-market shorting; in regular hours it was best "
                                 f"{PHRASE.get(r1['slice'], r1['slice'])} ({pp(r1['lift'])}, interval {pp(r1['lo'])} "
                                 f"to {pp(r1['hi'])}).")
            for r in cg.itertuples():
                nums.append(_n(f"Best signals, {r.slice}", r.lift, "pts",
                               {"kind": "events", "sig": good[0], "where": {"coarse": r.slice}}))
    co = an.pooled_by(ev, "coarse", "edge", reps, seed)
    if len(co) and good:
        allneg = bool((co["hi"] < 0).all())
        parts.append("Pooled across all signals" + (", every time window did worse than random timing, because "
                     "most breakdown signals fire right before a bounce." if allneg else
                     f", the picture is weaker: largest {PHRASE.get(co.sort_values('lift').iloc[-1]['slice'], '')} "
                     f"({pp(co['lift'].max())})."))
        co = co.iloc[0:0]
    if len(co):
        order = [c[0] for c in an.COARSE]
        co["o"] = co["slice"].map({k: i for i, k in enumerate(order)})
        co = co.sort_values("o")
        best = co.sort_values("lift", ascending=False).iloc[0]
        worst = co.sort_values("lift").iloc[0]
        sig_b = co[co["lo"] > 0]
        parts.append(f"Across all backside signals, the edge over random timing was largest {PHRASE.get(best['slice'], best['slice'])} "
                     f"({pp(best['lift'])}, interval {pp(best['lo'])} to {pp(best['hi'])}, {num(best['n'])} signals) "
                     f"and smallest {PHRASE.get(worst['slice'], worst['slice'])} ({pp(worst['lift'])}).")
        if not len(sig_b):
            parts.append("No time bucket's signal edge was clearly above random timing.")
        for r in co.itertuples():
            nums.append(_n(f"Signal lift, {r.slice}", r.lift, "pts", {"kind": "events", "where": {"coarse": r.slice}, "pooled": True}))
    cut, heat, marg = res["cutoffs"], res["start_heat"], res["marginal_tod"]
    prim = None
    if len(cut):
        prim_cid = res["cut_cids"][0]
        fin = res["finalists"]
        prim = fin[(fin["cid"] == prim_cid)].iloc[0]
        c_all = cut[(cut["cid"] == prim_cid) & (cut["period"] == "all") & (cut["mode"] == "stop new entries")]
        c_all = c_all[c_all["n"] >= 20]
        if len(c_all):
            by_total = c_all.sort_values("total", ascending=False).iloc[0]
            by_sharpe = c_all.sort_values("day_sharpe", ascending=False).iloc[0]
            c_tr = cut[(cut["cid"] == prim_cid) & (cut["period"] == "train") & (cut["mode"] == "stop new entries") & (cut["n"] >= 10)]
            c_te = cut[(cut["cid"] == prim_cid) & (cut["period"] == "test") & (cut["mode"] == "stop new entries") & (cut["n"] >= 10)]
            tr_best = c_tr.sort_values("total", ascending=False)["cutoff"].iloc[0] if len(c_tr) else None
            te_best = c_te.sort_values("total", ascending=False)["cutoff"].iloc[0] if len(c_te) else None
            hp = heat[(heat["cid"] == prim_cid) & (heat["n"] >= 20)]
            start_best = hp.sort_values("total", ascending=False).iloc[0] if len(hp) else None
            verb = "gave the most total profit" if by_total["total"] > 0 else "lost the least"
            parts.append(f"For {strat_label(prim)}, stopping new entries at "
                         f"{by_total['cutoff']} {verb}: {pct(by_total['ev'], 2, True)} per trade after costs, "
                         f"{pct(by_total['win'])} winners, {pct(by_total['pos_days'])} of days positive, "
                         f"{num(by_total['n'])} trades.")
            q2["all_negative"] = bool((c_all["ev"] < 0).all())
            if q2["all_negative"]:
                parts.append("It lost money at every cutoff after costs, so here the cutoff only decides how "
                             "much is lost; it is not a reason to trade.")
            if "ev_gross" in c_all:
                g_best = c_all.sort_values("ev_gross", ascending=False).iloc[0]
                q2["gross_cutoff"], q2["gross_cutoff_ev"] = g_best["cutoff"], float(g_best["ev_gross"])
                parts.append(f"Before costs, the best cutoff for it was {g_best['cutoff']} "
                             f"({pct(g_best['ev_gross'], 2, True)} per trade).")
            if by_sharpe["cutoff"] != by_total["cutoff"]:
                parts.append(f"The steadiest day-to-day results came from stopping at {by_sharpe['cutoff']} "
                             f"(daily mean/stdev {by_sharpe['day_sharpe']:.2f}).")
            if tr_best and te_best:
                agree = "the same" if tr_best == te_best else "a different"
                parts.append(f"In the first 60% of weeks the best cutoff was {tr_best}; in the last 40% it was "
                             f"{te_best} ({agree} answer).")
            if start_best is not None:
                parts.append(f"The best start/stop pair was {start_best['start']} to {start_best['cutoff']} "
                             f"({pct(start_best['ev'], 2, True)} per trade over {num(start_best['n'])} trades).")
            nums += [_n(f"EV per trade, stop at {by_total['cutoff']}", by_total["ev"], "pct2",
                        {"kind": "trades", "cid": int(prim_cid), "before": hm(by_total["cutoff"])}),
                     _n("Win rate", by_total["win"], "pct", {"kind": "trades", "cid": int(prim_cid), "before": hm(by_total["cutoff"])})]
            m = marg[marg["cid"] == prim_cid].sort_values("tod15")
            late = m[(m["tod15"] >= 0) & (m["ev"] <= 0) & (m["n"] >= 10)]
            if len(late):
                first_bad = late.iloc[0]
                parts.append(f"Trades entered in the {first_bad['bucket']} bucket averaged "
                             f"{pct(first_bad['ev'], 2, True)} ({num(first_bad['n'])} trades), the first 15-minute "
                             f"bucket with a non-positive average.")
            q2["cutoff"] = by_total["cutoff"]
            q2["start"] = start_best["start"] if start_best is not None else None
            q2["cid"] = int(prim_cid)
    ok = (len(co) and (co["lo"] > 0).any()) or (good and len(cg) and (cg["lo"] > 0).any())
    q2.update(status="supported" if ok else ("weak" if parts else "insufficient"),
              answer=" ".join(parts) if parts else "Not enough data yet.", numbers=nums)
    q2["answer"] += (" A historical cutoff describes the average past trade; your own fills, size and "
                     "discipline decide whether it applies to you.")
    out.insert(1, q2)

    # ---------------------------------------------------------------- Q3
    wf, split = res["wf"], res["split"]
    q3 = {"id": "q3", "question": "Which entry, stop and profit-taking approach has the strongest "
                                  "out-of-sample expected value after costs?"}
    parts, nums = [], []
    if wf.get("folds"):
        weekly = np.array([f["test_sum"] / f["test_n"] for f in wf["folds"] if f["test_n"]])
        parts.append(f"Walk-forward test: each week was traded with the configuration that ranked best on "
                     f"the earlier weeks only. Out of sample that procedure averaged {pct(wf['oos_ev'], 2, True)} "
                     f"per trade after costs over {num(wf['oos_n'])} trades; {pct(wf['weeks_positive'], 0)} of the "
                     f"{len(weekly)} test weeks were positive. It switched between {wf['picks_distinct']} different "
                     f"configurations.")
        nums.append(_n("Walk-forward EV per trade", wf["oos_ev"], "pct2", {"kind": "walkforward"}))
    wfg = res.get("wf_gross") or {}
    if wfg.get("folds"):
        wkg = np.array([f["test_sum"] / f["test_n"] for f in wfg["folds"] if f["test_n"]])
        gpos = float((wkg > 0).mean()) if len(wkg) else math.nan
        parts.append(f"Before costs, the same procedure averaged {pct(wfg['oos_ev'], 2, True)} per trade "
                     f"({num(wfg['oos_n'])} trades, {pct(gpos, 0)} of test weeks positive).")
        nums.append(_n("Walk-forward EV before costs", wfg["oos_ev"], "pct2", {"kind": "walkforward"}))
        fin = res["finalists"]
        if len(fin) and "cost_total" in fin:
            ct_ = fin[fin["n"] >= 30]
            if len(ct_):
                c = ct_[["cost_total", "cost_spread_slip", "cost_comm", "cost_locate"]].median()
                parts.append(f"A typical short-listed trade paid {pct(c['cost_total'], 2)} in costs: "
                             f"{pct(c['cost_spread_slip'], 2)} spread and slippage, {pct(c['cost_comm'], 2)} "
                             f"commission and {pct(c['cost_locate'], 2)} locate.")
        good_g = wfg.get("oos_ev", math.nan)
        good_n = wf.get("oos_ev", math.nan)
        if good_g == good_g and good_g > 0 and gpos >= 0.6 and not (good_n == good_n and good_n > 0):
            parts.append("So there is a repeatable edge before costs, and the costs erase it. Cheaper "
                         "execution (per-share commission, locate, resting orders instead of market orders) "
                         "decides whether it pays.")
        elif good_g == good_g and good_g <= 0:
            parts.append("Even before costs, the best past setup did not carry forward. The limit is the "
                         "signal, not the costs.")
    top = split.get("top") or []
    if top:
        t0 = top[0]
        parts.append(f"Ranked on the first {int(round(cfg.val.train_frac * 100))}% of weeks, the top configuration was "
                     f"{strat_label(t0)}. It earned {pct(t0['train_ev'], 2, True)} per trade there and "
                     f"{pct(t0['test_ev'], 2, True)} on the later weeks it had never seen ({num(t0['test_n'])} trades).")
        pos = [t for t in top if t["test_ev"] == t["test_ev"] and t["test_ev"] > 0]
        parts.append(f"{len(pos)} of the top {len(top)} training-period configurations stayed positive out of sample.")
        nums.append(_n("Top config, out-of-sample EV", t0["test_ev"], "pct2",
                       {"kind": "trades", "cid": int(t0["cid"]), "window": t0["window"], "from": res["test_from"]}))
    nbr = res.get("neighbors")
    if top and nbr is not None and len(nbr):
        r0 = nbr[(nbr["cid"] == top[0]["cid"]) & (nbr["win_idx"] == top[0]["win_idx"])]
        if len(r0) and r0.iloc[0]["neighbors"] > 0:
            r0 = r0.iloc[0]
            spike = (r0["own_ev"] > 0) and (r0["neighbor_ev"] < r0["own_ev"] / 2)
            parts.append(f"Its {int(r0['neighbors'])} neighbouring settings (one step tighter or looser stop, exit or "
                         f"hold) averaged {pct(r0['neighbor_ev'], 2, True)} per trade against its own "
                         f"{pct(r0['own_ev'], 2, True)}" + (", so it is an isolated peak, the usual sign of overfitting."
                                                           if spike else "."))
    fin_ = res["finalists"]
    if top and len(fin_) and "ev_live" in fin_:
        f0 = fin_[(fin_["cid"] == top[0]["cid"]) & (fin_["win_idx"] == top[0]["win_idx"])]
        if len(f0) and f0.iloc[0].get("live_n", 0):
            f0 = f0.iloc[0]
            parts.append(f"Limited to trades a live screen could have caught (already under $1 with 1M+ shares by the "
                         f"entry minute, {pct(f0['live_share'], 0)} of its trades), it averaged "
                         f"{pct(f0['ev_live'], 2, True)} over the whole period.")
    pb = res["pbo"]
    if pb.get("splits"):
        parts.append(f"Probability that the in-sample winner is below median out of sample (CSCV): "
                     f"{pct(pb['pbo'], 0)} across {pb['splits']} splits.")
        nums.append(_n("Probability of backtest overfitting", pb["pbo"], "pct", {"kind": "pbo"}))
    parts.append(f"{num(res['cand_tested'])} configuration-and-time-window combinations were tested, so the best "
                 f"in-sample number is optimistic by construction; rely on the out-of-sample figures.")
    good = wf.get("oos_ev", float("nan"))
    q3.update(status="supported" if (good == good and good > 0 and wf.get("weeks_positive", 0) >= 0.6) else "not supported",
              answer=" ".join(parts), numbers=nums)
    out.insert(2, q3)

    # --------------------------------------------------- section 20 details
    out += _section20(res, S, ctl, reps, seed)
    kpis = _kpis(res, S, ctl, q2)
    return {"findings": out, "kpis": kpis, "cards": signal_cards(res, S)}


WINDOW_TEXT = {"Pre-market": "in the pre-market", "Before 10:00": "between 9:30 and 10:00 AM",
               "10:00-11:00": "between 10:00 and 11:00 AM", "11:00-12:00": "between 11:00 AM and noon",
               "Afternoon": "after noon"}


def signal_cards(res: dict, S: pd.DataFrame) -> list[dict]:
    """One statement per signal, in the form the user asked for, from real counts.
    "Most effective" is the time window with the largest edge over random
    moments in that same window (30+ signals), and says how sure that is."""
    co = res.get("signal_coarse")
    cards = []
    if co is None or not len(S):
        return cards
    ctl_w = co[co["sig"] == "B00"].set_index("coarse")["edge"] if (co["sig"] == "B00").any() else pd.Series(dtype=float)
    for r in S.sort_values("edge_lift", ascending=False).to_dict("records"):
        w = co[(co["sig"] == r["sig"]) & (co["n"] >= 30)].copy()
        best = None
        if len(w) and len(ctl_w):
            w["lift"] = w["edge"] - w["coarse"].map(ctl_w)
            w = w.dropna(subset=["lift"])
            if len(w):
                best = w.sort_values("lift", ascending=False).iloc[0]
        txt = (f"{r['name']}: price made a new low of the move within 15 minutes in {pct(r['p_nml_15'])} of "
               f"{num(r['n'])} cases (random moments at the same times of day: {pct(r.get('nml_15_ctl'))}). "
               f"Median drop over the next 30 minutes {pct(r['mfe_30_med'])}, median rise {pct(r['mae_30_med'])}. "
               f"It fell 1 ATR before rising 1 ATR {pct(r['p_worked'])} of the time and did the reverse "
               f"{pct(r['p_failed'])} ({pp(r['edge_lift'])} vs random timing).")
        if best is not None:
            if best["lift"] > 0:
                txt += (f" It was most effective {WINDOW_TEXT.get(best['coarse'], best['coarse'])}: "
                        f"{pp(best['lift'])} vs random moments in that window, {num(best['n'])} signals.")
                if best["n"] < 200:
                    txt += " That window has few signals, so treat it as a hint."
            else:
                txt += (f" Even its best window ({WINDOW_TEXT.get(best['coarse'], best['coarse'])}) trailed random "
                        f"moments ({pp(best['lift'])}).")
        q = r.get("edge_q_pos")
        verdict = ("beats random timing after the multiple-testing correction" if q == q and q is not None and q <= 0.05
                   and r["edge_lift_lo"] > 0 else
                   "is worse than random timing" if r["edge_lift_hi"] < 0 else
                   "is not distinguishable from random timing")
        cards.append({"sig": r["sig"], "text": txt, "verdict": verdict, "lift": r["edge_lift"],
                      "cohort": {"kind": "events", "sig": r["sig"]},
                      "window": best["coarse"] if best is not None else None})
    return cards


def _n(label, value, fmt, cohort):
    v = None if value is None or (isinstance(value, float) and not math.isfinite(value)) else float(value)
    return {"label": label, "value": v, "fmt": fmt, "cohort": cohort}


def _section20(res, S, ctl, reps, seed):
    ev, ctx = res["events"], res["context"]
    out = []

    def pooled_dim(dim, title, q, hindsight=False):
        sub = ctx[ctx["dim"] == dim]
        if not len(sub):
            return
        p = an.pooled_by(_ctx_events(ev, dim), dim, "edge", reps, seed)
        if not len(p):
            return
        p = p[p["n"] >= 30]
        if not len(p):
            return
        best = p.sort_values("lift", ascending=False).iloc[0]
        worst = p.sort_values("lift").iloc[0]
        clear = p[p["lo"] > 0]
        txt = (f"Pooled across all backside signals, the edge over random timing was highest for "
               f"{best['slice']} ({pp(best['lift'])}, interval {pp(best['lo'])} to {pp(best['hi'])}, "
               f"{num(best['n'])} signals) and lowest for {worst['slice']} ({pp(worst['lift'])}).")
        if not len(clear):
            txt += " No slice was clearly better than random timing."
        if hindsight:
            txt += " This split uses hindsight (the day's high is only known at the close), so it explains " \
                   "the past but cannot be traded directly; minutes since the high is the live proxy."
        out.append({"id": dim, "question": q, "status": "supported" if len(clear) else "not supported",
                    "answer": txt, "hindsight": hindsight,
                    "numbers": [_n(r.slice, r.lift, "pts", {"kind": "events", "pooled": True, "where": {dim: r.slice}})
                                for r in p.itertuples()]})

    pooled_dim("c_gain", "Move into the high", "Are backside signals more reliable after a large percentage move?")
    pooled_dim("c_true", "Day's high", "Are backside signals more reliable after the stock has made its high of day?",
               hindsight=True)
    pooled_dim("c_age", "Minutes since the high", "How long after the high do signals work best? (live proxy)")
    pooled_dim("c_cross", "9 vs 20 EMA", "Do signals work better before or after the 9 EMA crosses below the 20 EMA?")
    pooled_dim("c_vwap", "VWAP", "Do signals work better once price is below VWAP?")
    pooled_dim("c_price", "Price", "Does the price level change the edge?")
    pooled_dim("c_rt", "Live universe", "Do the results hold for signals a trader could have screened for live?")

    def pair(a, b, q):
        ra, rb = S[S["sig"] == a], S[S["sig"] == b]
        if not len(ra) or not len(rb):
            return
        ra, rb = ra.iloc[0], rb.iloc[0]
        better = None
        if ra["edge_lift_lo"] > rb["edge_lift_hi"]:
            better = ra
        elif rb["edge_lift_lo"] > ra["edge_lift_hi"]:
            better = rb
        verdict = (f"{better['name']} is measurably better." if better is not None
                   else "The intervals overlap; the data does not separate them.")
        out.append({"id": f"{a}_vs_{b}", "question": q, "status": "supported" if better is not None else "not supported",
                    "answer": (f"{ra['name']}: {pp(ra['edge_lift'])} vs random (interval {pp(ra['edge_lift_lo'])} to "
                               f"{pp(ra['edge_lift_hi'])}, n={num(ra['n'])}). {rb['name']}: {pp(rb['edge_lift'])} "
                               f"(interval {pp(rb['edge_lift_lo'])} to {pp(rb['edge_lift_hi'])}, n={num(rb['n'])}). {verdict}"),
                    "numbers": [_n(ra["name"], ra["edge_lift"], "pts", {"kind": "events", "sig": a}),
                                _n(rb["name"], rb["edge_lift"], "pts", {"kind": "events", "sig": b})]})

    stab = res.get("stability")
    if stab is not None and len(stab) and len(S):
        lines, nums = [], []
        for r in S.sort_values("edge_lift", ascending=False).head(3).to_dict("records"):
            g = stab[stab["sig"] == r["sig"]]
            if len(g) >= 2:
                k = int((g["lift"] > 0).sum())
                lines.append(f"{r['name']} beat random timing in {k} of {len(g)} periods "
                             f"(range {pp(g['lift'].min())} to {pp(g['lift'].max())})")
                nums.append(_n(f"{r['name']}: periods ahead of random", k, "int", {"kind": "events", "sig": r["sig"]}))
        if lines:
            unit = "quarter" if stab["block"].str.contains("Q").any() else "month"
            out.append({"id": "stability", "question": f"Do the best signals hold up {unit} after {unit}?",
                        "status": "descriptive",
                        "answer": "; ".join(lines) + f". Each {unit} is compared with random moments at the same "
                                  f"times of day in that {unit}. A signal that only worked in one stretch is fragile.",
                        "numbers": nums})
    pair("S01", "S08", "Is the first 9 EMA breakdown or the first failed 9 EMA reclaim the better short?")
    pair("S02", "S09", "Is the first 20 EMA breakdown or the first failed 20 EMA reclaim the better short?")
    pair("S03", "S10", "Is the first VWAP loss or the first failed VWAP reclaim the better short?")
    pair("S04", "S02", "Is the 9/20 EMA bear cross better than the earlier 20 EMA loss?")
    pair("C01", "S02", "Is a 20 EMA loss below VWAP better than any 20 EMA loss?")

    pm = res["param_marginals"]
    if len(pm):
        t = pm[pm["dimension"] == "timing"].sort_values("mean_ev", ascending=False)
        if len(t):
            lines = "; ".join(f"{TIMING.get(r.value, r.value)}: average {pct(r.mean_ev, 2, True)}, "
                              f"{pct(r.share_positive, 0)} of setups positive" for r in t.itertuples())
            out.append({"id": "confirmation", "question": "Does waiting for confirmation improve results?",
                        "status": "descriptive",
                        "answer": ("Averaged over every signal, stop, exit and hold (so no single tuned setup "
                                   f"drives it): {lines}."),
                        "numbers": [_n(TIMING.get(r.value, r.value), r.mean_ev, "pct2", {"kind": "marginals", "dimension": "timing"})
                                    for r in t.itertuples()]})
    st = res["stages"]
    if len(st) and {4, 5}.issubset(set(st["stage"])):
        a, b = st[st["stage"] == 4].iloc[0], st[st["stage"] == 5].iloc[0]
        out.append({"id": "warning_vs_confirmed", "question": "Early warning or confirmed backside: which is the better time to short?",
                    "status": "descriptive",
                    "answer": (f"At the start of an early-warning phase (stage 4), price fell 1 ATR before rising 1 ATR "
                               f"{pp(a['fwd_edge'])} more often than the reverse, and {pct(a['reach5_30'])} of these "
                               f"phases became a confirmed backside within 30 minutes. At the start of a confirmed "
                               f"backside (stage 5) that edge was {pp(b['fwd_edge'])}, and {pct(b['fwd_nhod_eod'])} "
                               f"still went on to a new high of day. Confirmation arrives a median "
                               f"{mins(a['dur_med'])} after the warning."),
                    "numbers": [_n("Stage 4 edge", a["fwd_edge"], "pts", {"kind": "segments", "stage": 4}),
                                _n("Stage 5 edge", b["fwd_edge"], "pts", {"kind": "segments", "stage": 5})]})
    return out


def _ctx_events(ev, dim):
    e = ev.copy()
    if dim == "c_gain":
        e[dim] = an._bins(e["gain_pc"] * 100, [-1e9, 25, 50, 100, 200, 1e9], ["moves under 25%", "25-50% moves", "50-100% moves", "100-200% moves", "moves over 200%"]).astype(str)
    elif dim == "c_true":
        e[dim] = np.where(e["after_true_hod"], "after the day's high", "before the day's high")
    elif dim == "c_age":
        e[dim] = an._bins(e["mins_since_hod"], [0, 5, 15, 30, 60, 1e9], ["0-5 min after the high", "5-15 min after", "15-30 min after", "30-60 min after", "60+ min after"]).astype(str)
    elif dim == "c_cross":
        e[dim] = np.where(e["e9_above_e20"], "signals before the 9/20 cross", "signals after the 9/20 cross")
    elif dim == "c_vwap":
        e[dim] = np.where(e["below_vwap"], "signals below VWAP", "signals above VWAP")
    elif dim == "c_price":
        e[dim] = an._bins(e["price"], [0, 0.25, 0.5, 0.75, 1.0, 1e9], ["under $0.25", "$0.25-0.50", "$0.50-0.75", "$0.75-1.00", "$1.00 and up"]).astype(str)
    elif dim == "c_rt":
        e[dim] = np.where(e["known_sub1"] & (e["cumvol"] >= 1_000_000), "signals screenable live", "signals that needed hindsight")
    return e


def _kpis(res, S, ctl, q2):
    k = []
    fin = res["finalists"]
    if len(S):
        a = S.sort_values("p_worked", ascending=False).iloc[0]
        k.append({"id": "continuation", "title": "Highest probability of downside continuation", "sig": a["sig"],
                  "value": a["p_worked"], "fmt": "pct", "text": a["name"],
                  "sub": f"fell 1 ATR before rising 1 ATR · n={num(a['n'])} · random timing {pct(ctl['p_worked']) if ctl is not None else 'n/a'}",
                  "cohort": {"kind": "events", "sig": a["sig"], "where": {"fp_atr": 0}}})
        b = S.sort_values("edge_lift_lo", ascending=False).iloc[0]
        k.append({"id": "best_signal", "title": "Best indicator or combination", "sig": b["sig"],
                  "value": b["edge_lift"], "fmt": "pts", "text": b["name"],
                  "sub": f"edge vs random timing, 95% interval {pp(b['edge_lift_lo'])} to {pp(b['edge_lift_hi'])}",
                  "cohort": {"kind": "events", "sig": b["sig"]}})
        c = S.sort_values("p_failed").iloc[0]
        k.append({"id": "lowest_fail", "title": "Lowest failure rate", "sig": c["sig"], "value": c["p_failed"],
                  "fmt": "pct", "text": c["name"], "sub": f"+1 ATR before −1 ATR · n={num(c['n'])}",
                  "cohort": {"kind": "events", "sig": c["sig"], "where": {"fp_atr": 1}}})
        d = S.sort_values("mfe_30_mean", ascending=False).iloc[0]
        k.append({"id": "best_mfe", "title": "Best average downside excursion", "sig": d["sig"],
                  "value": d["mfe_30_mean"], "fmt": "pct", "text": d["name"],
                  "sub": f"max drop within 30 min (median {pct(d['mfe_30_med'])}) · random timing "
                         f"{pct(ctl['mfe_30_mean']) if ctl is not None else 'n/a'}",
                  "cohort": {"kind": "events", "sig": d["sig"]}})
    if q2.get("all_negative"):
        g = q2.get("gross_cutoff")
        if g and not (q2.get("gross_cutoff_ev", -1) > 0):
            g = None                                   # not profitable even before costs
        k.append({"id": "begin", "title": "Best time to begin scalping", "value": None, "fmt": "text",
                  "text": "None after costs",
                  "sub": "the leading setup lost money at every start and stop time once costs were charged",
                  "cohort": {"kind": "trades", "cid": q2["cid"]}})
        k.append({"id": "stop", "title": "Best time to stop new scalps", "value": None, "fmt": "text",
                  "text": ("Before costs: " + ("all day" if g == "16:00" else g)) if g else "None after costs",
                  "sub": (f"before costs its best cutoff averaged {pct(q2.get('gross_cutoff_ev'), 2, True)} per "
                          f"trade; after costs no cutoff made money") if g else
                         "the leading setup lost money at every cutoff, before and after costs",
                  "cohort": {"kind": "trades", "cid": q2["cid"], "before": hm(g) if g and g != "16:00" else 9999}})
    else:
        if q2.get("start"):
            k.append({"id": "begin", "title": "Best time to begin scalping", "value": None, "fmt": "text",
                      "text": q2["start"], "sub": "start of the best start/stop pair for the leading strategy",
                      "cohort": {"kind": "trades", "cid": q2["cid"], "after": hm(q2["start"])}})
        if q2.get("cutoff"):
            k.append({"id": "stop", "title": "Best time to stop new scalps", "value": None, "fmt": "text",
                      "text": "Trade all day" if q2["cutoff"] == "16:00" else q2["cutoff"],
                      "sub": "cutoff with the most total profit for the leading strategy",
                      "cohort": {"kind": "trades", "cid": q2["cid"], "before": hm(q2["cutoff"])}})
    top = res["split"].get("top") or []
    if top:
        t0 = top[0]
        k.append({"id": "best_ev", "title": "Highest expected value setup (out of sample)", "value": t0["test_ev"],
                  "fmt": "pct2", "text": strat_label(t0),
                  "sub": f"after costs, on weeks not used to pick it · {num(t0['test_n'])} trades",
                  "cohort": {"kind": "trades", "cid": int(t0["cid"]), "window": t0["window"], "from": res["test_from"]}})
    if len(fin):
        f = fin[(fin["n"] >= 30) & (fin["ev"] > 0) & (fin["test_ev"] > 0)].copy()
        if len(f):
            f["payoff"] = f["avg_win"] / f["avg_loss"].abs()
            r = f.sort_values("payoff", ascending=False).iloc[0]
            k.append({"id": "best_rr", "title": "Best risk-to-reward combination", "value": r["payoff"], "fmt": "x",
                      "text": strat_label(r),
                      "sub": f"average win / average loss · {pct(r['win'])} winners · positive in and out of sample",
                      "cohort": {"kind": "trades", "cid": int(r["cid"]), "window": r["window"]}})
    return k
