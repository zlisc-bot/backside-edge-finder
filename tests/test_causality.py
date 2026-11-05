"""No look-ahead: cutting the data at any minute must not change anything that
was already known at that minute (signals, their context fields, stages)."""
import numpy as np
import pandas as pd

from bse.config import Config
from bse.indicators import prepare_session
from bse.signals import detect
from bse.stages import classify
from bse.synthetic import SyntheticProvider

CTX = ["sig", "idx", "mod", "price", "epoch", "hod_idx", "hod", "move_low", "lod", "gain_pc",
       "runup", "pullback", "vwap_dist", "below_vwap", "e9_above_e20", "e20_falling", "cumvol",
       "known_sub1", "ssr", "atr_pct", "spread_est", "rg_vol_ratio"]


def _sessions(n=12):
    prov = SyntheticProvider(n_tickers=200, runner_rate=0.5)
    out = []
    for i, t in enumerate(prov.tickers):
        d = "2026-10-14"
        if not prov.runner(t, d):
            continue
        bars = next(prov.minute_bars([(t, d)], {d: "2026-10-13"}))
        meta = {"session_id": f"{t}_{d}", "ticker": t, "date": d, "pc": prov.level(t, "2026-10-13"),
                "ssr_carry": False, "v": float(bars.v.sum())}
        out.append((bars, meta))
        if len(out) >= n:
            break
    return out


def _key(evs, upto):
    rows = [tuple(e[k] for k in CTX) for e in evs if e["idx"] <= upto]
    return sorted(rows, key=lambda r: (r[1], r[0]))


def test_truncation_invariance():
    cfg = Config()          # built-in defaults, not config.local.json
    checked = 0
    for bars, meta in _sessions():
        full = prepare_session(bars, meta, cfg)
        ev_full, pb_full = detect(full, cfg)
        st_full = classify(full, pb_full, cfg)
        rng = np.random.default_rng(1)
        for k in sorted(rng.integers(40, len(full) - 1, size=6)):
            cut_ts = full.ts[k]
            part = prepare_session(bars[bars.ts <= cut_ts], meta, cfg)
            assert len(part) == k + 1
            ev_part, pb_part = detect(part, cfg)
            st_part = classify(part, pb_part, cfg)
            a, b = _key(ev_full, k), _key(ev_part, k)
            assert len(a) == len(b), (meta["ticker"], k)
            for ra, rb in zip(a, b):
                for x, y in zip(ra, rb):
                    if isinstance(x, float) and np.isnan(x):
                        assert np.isnan(y)
                    else:
                        assert x == y or (isinstance(x, float) and abs(x - y) < 1e-12), (ra, rb)
            assert (st_full[:k + 1] == st_part).all(), (meta["ticker"], k)
            for name in ("fr_ema9", "fr_ema20", "fr_vwap", "lh_flag", "epoch_id"):
                assert (pb_full[name][:k + 1] == pb_part[name]).all(), name
            checked += 1
    assert checked >= 30


if __name__ == "__main__":
    test_truncation_invariance()
    print("ok")
