"""Out-of-sample checks on the strategy grid.

A "candidate" is one configuration (signal, entry timing, stop, exit, max
hold) restricted to one entry-time window (a contiguous run of the time-of-day
buckets). Tens of thousands of candidates are tested, so the best in-sample
number is optimistic by construction. Three defences:

1. Walk-forward: each week is predicted by a candidate chosen using only the
   weeks before it. The concatenated out-of-sample weeks measure the selection
   procedure itself - the honest estimate of what "optimise, then trade" earns.
2. Chronological split: rank on the first 60% of weeks, report the last 40%.
3. CSCV probability of backtest overfitting (Bailey et al.): how often the
   in-sample winner lands in the bottom half out of sample.
Intervals use day- or week-clustered standard errors, because trades on the
same day are correlated.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from .backtest import Grid
from .config import fmt_hm, hm


def windows(grid: Grid) -> list[tuple[int, int]]:
    nb = grid.nB
    return [(a, b) for a in range(nb) for b in range(a, nb)]


def window_label(grid: Grid, w: tuple[int, int]) -> str:
    e = grid.cfg.strat.tod_edges
    return f"{e[w[0]]}-{e[w[1] + 1]}"


def config_table(grid: Grid) -> pd.DataFrame:
    """Whole-sample statistics for every configuration (all RTH entries)."""
    n = grid.d_n.sum(1).astype(float)
    S = grid.d_sum.sum(1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = S / n
        sq = grid.d_sq.sum(1).astype(float)
        var = (sq - n * mean ** 2) / np.maximum(n - 1, 1)
        wins = grid.d_win.sum(1).astype(float)
        resid = grid.d_sum.astype(float) - grid.d_n.astype(float) * mean[:, None]
        se_day = np.sqrt((resid ** 2).sum(1)) / n
        active = grid.d_n > 0
        n_days = active.sum(1)
        pos_days = ((grid.d_sum > 0) & active).sum(1)
        day_pnl = np.where(active, grid.d_sum, np.nan)
        dmean = np.nanmean(day_pnl, axis=1)
        dstd = np.nanstd(day_pnl, axis=1, ddof=1)
        lo = {k: grid.scen[k]["sum"].sum(1) / grid.scen[k]["n"].sum(1) for k in grid.scen}
        df = pd.DataFrame({
            "cid": np.arange(grid.ncfg), "n": n, "ev": mean, "sd": np.sqrt(np.maximum(var, 0)),
            "se": se_day, "lcb": mean - 1.96 * se_day, "win": wins / n,
            "pf": grid.pos / np.abs(np.minimum(grid.neg, -1e-12)),
            "avg_win": grid.pos / np.maximum(wins, 1), "avg_loss": grid.neg / np.maximum(n - wins, 1),
            "ev_r": grid.rmult / n, "hold_avg": grid.hold_min / n,
            "stop_share": grid.reasons[:, 0] / n, "target_share": grid.reasons[:, 1] / n,
            "trail_share": grid.reasons[:, 2] / n, "time_share": grid.reasons[:, 3] / n,
            "days": n_days, "pos_days": pos_days / np.maximum(n_days, 1),
            "day_sharpe": dmean / dstd, "ev_low": lo["low"], "ev_high": lo["high"],
        })
    return df


def _decode_cols(grid: Grid, cids: np.ndarray) -> pd.DataFrame:
    rows = [grid.decode(int(c)) for c in cids]
    return pd.DataFrame(rows)


class Candidates:
    """(config x time window) candidates with per-week sums for fast re-use."""

    def __init__(self, grid: Grid, min_total: int):
        self.grid = grid
        self.W = windows(grid)
        cs_s = np.cumsum(grid.wb_sum.astype(float), axis=2)
        cs_n = np.cumsum(grid.wb_n.astype(float), axis=2)
        cs_w = np.cumsum(grid.wb_win.astype(float), axis=2)
        pad = lambda x: np.concatenate([np.zeros(x.shape[:2] + (1,)), x], axis=2)
        cs_s, cs_n, cs_w = pad(cs_s), pad(cs_n), pad(cs_w)
        tot_n = np.stack([cs_n[:, :, b + 1] - cs_n[:, :, a] for a, b in self.W], axis=1).sum(2)
        keep = tot_n >= min_total                         # (ncfg, nWin)
        self.cid, self.win = np.nonzero(keep)
        a = np.array([self.W[w][0] for w in self.win])
        b = np.array([self.W[w][1] for w in self.win])
        self.S = (cs_s[self.cid, :, b + 1] - cs_s[self.cid, :, a]).astype(np.float32)   # (N, weeks)
        self.N = (cs_n[self.cid, :, b + 1] - cs_n[self.cid, :, a]).astype(np.float32)
        self.Wn = (cs_w[self.cid, :, b + 1] - cs_w[self.cid, :, a]).astype(np.float32)
        self.n_tested = int(keep.size)

    def __len__(self):
        return len(self.cid)

    def stats(self, weeks: np.ndarray):
        S = self.S[:, weeks].sum(1)
        N = self.N[:, weeks].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = S / N
            resid = self.S[:, weeks] - self.N[:, weeks] * mean[:, None]
            se = np.sqrt((resid ** 2).sum(1)) / N
        return mean, se, N, S

    def label(self, i: int) -> dict:
        d = self.grid.decode(int(self.cid[i]))
        d["window"] = window_label(self.grid, self.W[self.win[i]])
        d["win_idx"] = int(self.win[i])
        d["cid"] = int(self.cid[i])
        return d


def walk_forward(cand: Candidates, min_train_weeks: int, min_train_n: int, top_k: int = 10) -> dict:
    nW = cand.S.shape[1]
    folds = []
    for k in range(min_train_weeks, nW):
        tr = np.arange(k)
        mean, se, N, _ = cand.stats(tr)
        score = np.where(N >= min_train_n, mean - 1.96 * se, -np.inf)
        if not np.isfinite(score).any():
            continue
        order = np.argsort(-score)[:top_k]
        best = order[0]
        tS, tN = cand.S[:, k], cand.N[:, k]
        lab = cand.label(best)
        folds.append({"test_week": int(k), "pick": lab, "train_ev": float(mean[best]),
                      "train_n": int(N[best]), "test_sum": float(tS[best]), "test_n": int(tN[best]),
                      "top_sum": float(tS[order].sum()), "top_n": int(tN[order].sum())})
    if not folds:
        return {"folds": [], "oos_ev": math.nan, "oos_n": 0}
    s = sum(f["test_sum"] for f in folds)
    n = sum(f["test_n"] for f in folds)
    ts = sum(f["top_sum"] for f in folds)
    tn = sum(f["top_n"] for f in folds)
    wk = np.array([f["test_sum"] / f["test_n"] for f in folds if f["test_n"]])
    return {"folds": folds, "oos_ev": s / n if n else math.nan, "oos_n": int(n),
            "oos_ev_top": ts / tn if tn else math.nan, "oos_n_top": int(tn),
            "weeks_positive": float((wk > 0).mean()) if len(wk) else math.nan,
            "picks_distinct": len({(f["pick"]["sig"], f["pick"]["timing"], f["pick"]["stop"],
                                    f["pick"]["target"], f["pick"]["hold"], f["pick"]["window"]) for f in folds})}


def chrono_split(cand: Candidates, train_frac: float, min_train_n: int, top: int = 25) -> dict:
    nW = cand.S.shape[1]
    cut = max(1, int(round(nW * train_frac)))
    tr, te = np.arange(cut), np.arange(cut, nW)
    m_tr, se_tr, n_tr, _ = cand.stats(tr)
    m_te, se_te, n_te, _ = cand.stats(te)
    score = np.where(n_tr >= min_train_n, m_tr - 1.96 * se_tr, -np.inf)
    order = np.argsort(-score)[:top]
    rows = []
    for i in order:
        if not np.isfinite(score[i]):
            continue
        d = cand.label(i)
        d.update(train_ev=float(m_tr[i]), train_n=int(n_tr[i]), train_lcb=float(score[i]),
                 test_ev=float(m_te[i]) if n_te[i] else math.nan, test_n=int(n_te[i]),
                 test_se=float(se_te[i]) if n_te[i] else math.nan)
        rows.append(d)
    ok = (n_tr >= min_train_n) & (n_te >= max(10, min_train_n // 3))
    rho = math.nan
    if ok.sum() > 20:
        rho = float(pd.Series(m_tr[ok]).rank().corr(pd.Series(m_te[ok]).rank()))
    share_pos_test = float((m_te[ok] > 0).mean()) if ok.any() else math.nan
    return {"cut_week": int(cut), "top": rows, "rank_corr": rho, "n_compared": int(ok.sum()),
            "share_positive_test": share_pos_test,
            "share_positive_train": float((m_tr[ok] > 0).mean()) if ok.any() else math.nan}


def pbo(cand: Candidates, blocks: int, min_half_n: int, chunk: int = 150_000) -> dict:
    """Probability of backtest overfitting via combinatorially symmetric CV.
    Weeks are grouped into `blocks` contiguous blocks; every half of the
    blocks picks the in-sample winner and the other half ranks it."""
    nW = cand.S.shape[1]
    S_ = min(blocks, nW - (nW % 2))
    if S_ < 4 or len(cand) < 10:
        return {"pbo": math.nan, "splits": 0}
    groups = np.array_split(np.arange(nW), S_)
    BS = np.stack([cand.S[:, g].sum(1) for g in groups], axis=1).astype(np.float32)
    BN = np.stack([cand.N[:, g].sum(1) for g in groups], axis=1).astype(np.float32)
    combos = list(itertools.combinations(range(S_), S_ // 2))
    M = np.zeros((S_, len(combos)), np.float32)
    for j, cmb in enumerate(combos):
        M[list(cmb), j] = 1.0
    Mo = 1.0 - M
    N = len(cand)
    best_is = np.full(len(combos), -np.inf)
    best_os = np.full(len(combos), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        for a in range(0, N, chunk):                       # pass 1: in-sample winners
            bs, bn = BS[a:a + chunk], BN[a:a + chunk]
            n_is = bn @ M
            is_ev = np.where(n_is >= min_half_n, (bs @ M) / n_is, -np.inf)
            os_ev = (bs @ Mo) / (bn @ Mo)
            k = np.argmax(is_ev, axis=0)
            v = is_ev[k, np.arange(len(combos))]
            upd = v > best_is
            best_is[upd] = v[upd]
            best_os[upd] = os_ev[k, np.arange(len(combos))][upd]
        below = np.zeros(len(combos))
        valid = np.zeros(len(combos))
        for a in range(0, N, chunk):                       # pass 2: rank of the winner out of sample
            bs, bn = BS[a:a + chunk], BN[a:a + chunk]
            os_ev = (bs @ Mo) / (bn @ Mo)
            ok = np.isfinite(os_ev)
            below += ((os_ev < best_os[None, :]) & ok).sum(0)
            valid += ok.sum(0)
    keep = np.isfinite(best_is) & np.isfinite(best_os) & (valid >= 10)
    if not keep.any():
        return {"pbo": math.nan, "splits": 0}
    w = np.clip(below[keep] / valid[keep], 1e-6, 1 - 1e-6)
    lam = np.log(w / (1 - w))
    return {"pbo": float((lam <= 0).mean()), "splits": int(keep.sum()),
            "median_logit": float(np.median(lam)), "logits": [round(float(x), 3) for x in lam[:400]]}


def cluster_bootstrap(values: np.ndarray, clusters: np.ndarray, reps: int, seed: int,
                      stat=np.mean) -> tuple[float, float]:
    """95% interval for stat(values), resampling whole clusters (days/sessions)."""
    if len(values) < 5:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    codes, inv = np.unique(clusters, return_inverse=True)
    k = len(codes)
    if k < 3:
        return (math.nan, math.nan)
    sums = np.bincount(inv, weights=values, minlength=k)
    cnts = np.bincount(inv, minlength=k).astype(float)
    if stat is np.mean:
        draw = rng.integers(0, k, size=(reps, k))
        bs = sums[draw].sum(1) / np.maximum(cnts[draw].sum(1), 1)
    else:
        groups = [values[inv == i] for i in range(k)]
        bs = np.array([stat(np.concatenate([groups[j] for j in rng.integers(0, k, size=k)]))
                       for _ in range(reps)])
    return float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))
