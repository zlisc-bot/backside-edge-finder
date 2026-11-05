"""End-to-end study: database -> every statistic the dashboard shows."""
from __future__ import annotations

import logging
import math
import time

import numpy as np
import pandas as pd

from . import analysis as an
from .backtest import TRADE_SIGNALS
from .config import DERIVED_DIR, Config
from .pipeline import run_pass1, run_pass2
from .validation import Candidates, chrono_split, config_table, pbo, walk_forward, window_label

log = logging.getLogger("bse.study")

REFERENCE_EXIT = ("pct0.03", "tp0.03", 30)     # same exit for every signal, for fair comparison


def _cid(grid, sig, timing, stop, target, hold):
    si = TRADE_SIGNALS.index(sig)
    ti = grid.cfg.strat.entry_timings.index(timing)
    sk = [r.code for r in grid.stops].index(stop)
    gk = [r.code for r in grid.targets].index(target)
    hk = list(grid.holds).index(hold)
    return grid.cfg_id(si, ti, sk, gk, hk)


def shortlist(grid, ct, cand, wf, split) -> pd.DataFrame:
    rows = []
    full_rth = (0, grid.nB - 1)
    # 1. walk-forward picks
    for f in wf["folds"]:
        p = f["pick"]
        rows.append({"cid": p["cid"], "win_idx": p["win_idx"], "why": "walk-forward pick"})
    # 2. chronological-split leaders (ranked on the training weeks only)
    for p in split["top"][:12]:
        rows.append({"cid": p["cid"], "win_idx": p["win_idx"], "why": "train-period leader"})
    # 3. full-sample leaders (in-sample; shown with their decay)
    mean, se, N, _ = cand.stats(np.arange(cand.S.shape[1]))
    score = np.where(N >= grid.cfg.strat.min_trades, mean - 1.96 * se, -np.inf)
    for i in np.argsort(-score)[:12]:
        if np.isfinite(score[i]):
            rows.append({"cid": int(cand.cid[i]), "win_idx": int(cand.win[i]), "why": "full-sample leader"})
    # 4. best whole-day config per signal, and the common reference exit per signal
    w_all = cand.W.index(full_rth)
    ok = ct[ct["n"] >= grid.cfg.strat.min_trades]
    for si, sig in enumerate(TRADE_SIGNALS):
        lo, hi = si * grid.nT * grid.block, (si + 1) * grid.nT * grid.block
        sub = ok[(ok["cid"] >= lo) & (ok["cid"] < hi)]
        if len(sub):
            rows.append({"cid": int(sub.loc[sub["lcb"].idxmax(), "cid"]), "win_idx": w_all, "why": "best for signal"})
        rows.append({"cid": _cid(grid, sig, "signal_close", *REFERENCE_EXIT), "win_idx": w_all, "why": "reference exit"})
    df = pd.DataFrame(rows)
    df = df.groupby(["cid", "win_idx"], as_index=False).agg(why=("why", lambda x: ", ".join(sorted(set(x)))))
    return df


def run_study(cfg: Config, days: list[str], workers: int | None = None) -> dict:
    t0 = time.time()
    ev, segs, sess, grid = run_pass1(cfg, days, workers)
    log.info("pass 1 done in %.0fs: %d events, %d segments, %d sessions", time.time() - t0, len(ev), len(segs), len(sess))
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(DERIVED_DIR / "events.parquet", index=False)
    segs.to_parquet(DERIVED_DIR / "segments.parquet", index=False)
    sess.to_parquet(DERIVED_DIR / "sessions.parquet", index=False)

    ct = config_table(grid)
    sc = cfg.strat
    cand = Candidates(grid, sc.min_trades)
    log.info("%d of %d (config x window) candidates have >= %d trades", len(cand), cand.n_tested, sc.min_trades)
    wf = walk_forward(cand, cfg.val.wf_min_train_weeks, sc.min_trades_train)
    split = chrono_split(cand, cfg.val.train_frac, sc.min_trades_train)
    pb = pbo(cand, cfg.val.cscv_blocks, max(10, sc.min_trades_train // 2))
    log.info("walk-forward OOS EV %.4f over %d trades; PBO %.2f", wf.get("oos_ev", math.nan), wf.get("oos_n", 0), pb.get("pbo", math.nan))

    short = shortlist(grid, ct, cand, wf, split)
    # are the leading setups isolated spikes? (their neighbours' in-sample EV)
    nb_rows = []
    for p in split.get("top", [])[:5]:
        sel = np.flatnonzero((cand.cid == p["cid"]) & (cand.win == p["win_idx"]))
        if len(sel):
            nb_rows.append({"cid": p["cid"], "win_idx": p["win_idx"], **an.neighbors(cand, grid, int(sel[0]))})
    wanted: dict[tuple, list] = {}
    for cid in sorted(set(short["cid"])):
        d = grid.decode(int(cid))
        wanted.setdefault((d["si"], d["ti"]), []).append((d["sk"], d["gk"], d["hk"], int(cid)))
    # the same walk-forward on returns BEFORE costs: is there a repeatable edge at all?
    cand_g = Candidates(grid, sc.min_trades, value="wb_gross")
    wf_gross = walk_forward(cand_g, cfg.val.wf_min_train_weeks, sc.min_trades_train)
    del cand_g
    log.info("walk-forward before costs: OOS EV %.4f over %d trades", wf_gross.get("oos_ev", math.nan),
             wf_gross.get("oos_n", 0))
    cutoffs = [c for c in sc.cutoffs if c != "16:00"]
    t1 = time.time()
    trades = run_pass2(cfg, days, wanted, cutoffs, workers)
    log.info("pass 2 done in %.0fs: %d trades", time.time() - t1, len(trades.get("normal", [])))
    normal = trades.get("normal", pd.DataFrame())

    # split point for train/test reporting (same as the chronological split)
    weeks = sorted(set(grid.week_of_day))
    cut_week = split["cut_week"]
    test_days = [d for d, w in zip(grid.days, grid.week_of_day) if w >= cut_week]
    test_from = test_days[0] if test_days else "9999-99-99"

    # finalist table -----------------------------------------------------------
    reps, seed = cfg.val.bootstrap_reps, cfg.val.seed
    fin_rows = []
    for r in short.itertuples():
        d = grid.decode(int(r.cid))
        w = cand.W[int(r.win_idx)]
        lab = window_label(grid, w)
        e = sc.tod_edges
        lo_m, hi_m = _hm(e[w[0]]), _hm(e[w[1] + 1])
        tr = normal[(normal["cid"] == r.cid) & (normal["entry_mod"] >= lo_m) & (normal["entry_mod"] < hi_m)] \
            if len(normal) else normal
        s_all = an.trade_summary(tr, reps, seed)
        s_tr = an.trade_summary(tr[tr["date"] < test_from], reps // 4, seed) if len(tr) else {"n": 0}
        s_te = an.trade_summary(tr[tr["date"] >= test_from], reps // 4, seed) if len(tr) else {"n": 0}
        row = {"cid": int(r.cid), "win_idx": int(r.win_idx), "window": lab, "why": r.why, **d,
               **{k: v for k, v in s_all.items()},
               "train_n": s_tr.get("n", 0), "train_ev": s_tr.get("ev", math.nan),
               "test_n": s_te.get("n", 0), "test_ev": s_te.get("ev", math.nan),
               "test_ev_lo": s_te.get("ev_lo", math.nan), "test_ev_hi": s_te.get("ev_hi", math.nan),
               "test_win": s_te.get("win", math.nan), "ev_r": float(ct.loc[int(r.cid), "ev_r"])}
        fin_rows.append(row)
    fin = pd.DataFrame(fin_rows)

    # cutoff study on the strongest whole-day configurations -------------------
    w_all = cand.W.index((0, grid.nB - 1))
    pool = fin[fin["win_idx"] == w_all].copy()
    pool = pool[pool["n"] >= sc.min_trades] if len(pool) else pool
    evp = an.prep_events(ev)
    sig = an.signal_tables(evp, cfg)
    st_ = sig["signals"]
    good = st_[(st_["group"] != "baseline") & (st_["n"] >= 100) & (st_["edge_lift_lo"] > 0)
               & (st_.get("edge_q_pos", 0) <= 0.05)].sort_values("edge_lift", ascending=False)
    best_ref = None
    if len(good):
        best_ref = _cid(grid, good.iloc[0]["sig"], "signal_close", *REFERENCE_EXIT)
    cut_cids = []
    if len(pool):
        cut_cids = list(pool.sort_values("train_ev", ascending=False)["cid"].head(4))
        ref_best = pool[pool["why"].str.contains("reference")].sort_values("ev", ascending=False)["cid"].head(1)
        cut_cids += [c for c in ref_best if c not in cut_cids]
    if best_ref is not None and best_ref in set(normal["cid"]) if len(normal) else False:
        cut_cids = [best_ref] + [c for c in cut_cids if c != best_ref]
    # The cutoff study leads with the setup the walk-forward chose most often: that is
    # what a trader following the procedure would actually have been trading.
    if wf.get("folds") and len(normal):
        from collections import Counter
        top_pick = Counter(f["pick"]["cid"] for f in wf["folds"]).most_common(1)[0][0]
        if top_pick in set(normal["cid"]):
            cut_cids = [top_pick] + [c for c in cut_cids if c != top_pick]
    flats = {k: v for k, v in trades.items() if k != "normal"}
    cuts, heat, marg = an.cutoff_tables(normal, flats, cut_cids, cfg, test_from) if cut_cids else (pd.DataFrame(),) * 3

    # events ---------------------------------------------------------------------
    stab = an.stability(evp)
    ctx = an.context_tables(evp, sig["lift"])
    todp = an.tod_profile(evp)
    stage_tab, stage_tm = an.stage_tables(segs)
    marg_params = an.marginals(ct, grid)
    log.info("study finished in %.0fs", time.time() - t0)
    return {"cfg": cfg, "days": days, "grid": grid, "ct": ct, "cand_n": len(cand), "cand_tested": cand.n_tested,
            "wf": wf, "wf_gross": wf_gross, "split": split, "pbo": pb, "neighbors": pd.DataFrame(nb_rows),
            "stability": stab, "finalists": fin, "trades": normal, "flats": flats,
            "cutoffs": cuts, "start_heat": heat, "marginal_tod": marg, "cut_cids": cut_cids,
            "events": evp, "signals": sig["signals"], "signal_tod": sig["signal_tod"],
            "signal_coarse": sig["signal_coarse"], "context": ctx, "tod_profile": todp,
            "stages": stage_tab, "stage_tm": stage_tm, "segments": segs, "sessions": sess,
            "param_marginals": marg_params, "test_from": test_from}


def _hm(s):
    h, m = s.split(":")
    return int(h) * 60 + int(m)
