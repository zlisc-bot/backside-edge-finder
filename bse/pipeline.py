"""Run the whole study over the stored database.

Pass 1 (parallel by date): indicators -> signals -> outcome labels -> stages ->
strategy grid accumulation -> chart payloads.
Pass 2: detailed trades for the short-listed strategies (and the cutoff
variants), used for drill-down, bootstrap intervals and cutoff tests.
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import calendar as cal
from .backtest import Grid, trades_for
from .config import DERIVED_DIR, Config, hm, load_config
from .indicators import prepare_session
from .outcomes import label
from .signals import detect
from .stages import classify, segments
from .store import Store

log = logging.getLogger("bse.pipeline")


def analysis_days(store: Store, cfg: Config, end: str | None = None) -> list[str]:
    ses = store.read_meta("sessions")
    if not len(ses):
        return []
    have = sorted(set(ses["date"]))
    end = end or have[-1]
    start = cal.months_back(end, cfg.universe.window_months)
    return [d for d in have if start <= d <= end]


def iter_sessions(store: Store, cfg: Config, dates: list[str]):
    ses = store.read_meta("sessions")
    ses = ses[ses["eligible_type"] & ses["date"].isin(set(dates))]
    for d, rows in ses.groupby("date"):
        day = store.read_minute(d)
        if not len(day):
            continue
        prev = store.read_bars_any(cal.prev_trading_day(d))
        day_g = dict(tuple(day.groupby("ticker")))
        prev_g = dict(tuple(prev.groupby("ticker"))) if len(prev) else {}
        for r in rows.to_dict("records"):
            b = day_g.get(r["ticker"])
            if b is None:
                continue
            pb = prev_g.get(r["ticker"])
            bars = pd.concat([pb, b], ignore_index=True) if pb is not None else b
            s = prepare_session(bars, r, cfg)
            if s is not None:
                yield s


def _chart_payload(s, stage):
    A = s.arr
    def r4(x):
        return [None if not np.isfinite(v) else round(float(v), 4 if v < 10 else 2) for v in x]
    return {"sid": s.sid, "ticker": s.ticker, "date": s.date,
            "t": s.mod.tolist(), "o": r4(s.o), "h": r4(s.h), "l": r4(s.l), "c": r4(s.c),
            "v": [int(x) for x in s.v], "e9": r4(A["ema9"]), "e20": r4(A["ema20"]),
            "vw": r4(A["vwap"]), "st": stage.tolist(),
            "pc": None if not np.isfinite(s.meta.get("pc", np.nan)) else float(s.meta["pc"]),
            "ssr_from": int(s.mod[np.argmax(A["ssr"])]) if A["ssr"].any() else None,
            "gaps": [int(s.mod[i]) for i in np.flatnonzero(A["after_gap"])]}


def _stage_segments(s, stage, cfg):
    """Segments plus what happened after each stage began (labels only)."""
    segs = segments(s, stage)
    A = s.arr
    pseudo = [{"sid": s.sid, "idx": g["a"], "move_low": float(A["move_low"][g["a"]]),
               "lod": float(A["lod"][g["a"]]), "hod": float(A["hod"][g["a"]])} for g in segs]
    label(s, pseudo, cfg)
    for g, p in zip(segs, pseudo):
        a = g["a"]
        j30 = int(np.searchsorted(s.mod, s.mod[a] + 30, side="right") - 1)
        g.update({k: p.get(k, np.nan) for k in ("nml_15", "nml_30", "fwd_30", "mfe_30", "mae_30",
                                                  "nhod_30", "nhod_eod", "fp_atr", "fwd_eod")})
        g["reach5_30"] = bool((stage[a:j30 + 1] == 5).any())
        g["reach8_eod"] = bool((stage[a:] == 8).any())
    return segs


def _pass1(args):
    dates, cfg, days_all, chart_dir = args
    store = Store()
    grid = Grid(cfg, days_all)
    evs, segs = [], []
    charts: dict[str, list] = {}
    sess_rows = []
    for s in iter_sessions(store, cfg, dates):
        events, perbar = detect(s, cfg)
        label(s, events, cfg)
        stage = classify(s, perbar, cfg)
        evs.extend(events)
        segs.extend(_stage_segments(s, stage, cfg))
        grid.add_session(s, perbar, [e for e in events if e["sig"] != "B00"])
        cp = _chart_payload(s, stage)
        cp["ev"] = [[int(e["idx"]), e["sig"]] for e in events if e["sig"] != "B00"]
        charts.setdefault(s.date, []).append(cp)
        sess_rows.append({"sid": s.sid, "ticker": s.ticker, "date": s.date, "n_bars": len(s),
                          "hod": float(s.arr["hod"][-1]), "hod_mod": int(s.mod[int(s.arr["hod_idx"][-1])]),
                          "lod": float(s.arr["lod"][-1]), "open_rth": float(s.c[np.argmax(s.arr["is_rth"])]),
                          "close": float(s.c[-1]), "pc": float(s.meta.get("pc", np.nan)),
                          "volume": float(s.meta.get("v", np.nan)), "ssr_any": bool(s.arr["ssr"].any()),
                          "halts": int(s.arr["after_gap"].sum()),
                          "max_stage": int(stage.max())})
    for d, items in charts.items():
        (chart_dir / f"{d}.json").write_text(json.dumps({"date": d, "sessions": items}, separators=(",", ":")))
    return evs, segs, grid, sess_rows


def _pass2(args):
    dates, cfg, days_all, wanted, cutoffs = args
    store = Store()
    grid = Grid.__new__(Grid)       # only rule tables are needed
    from .backtest import build_rules
    grid.stops, grid.targets, grid.holds = build_rules(cfg)
    out = {"normal": []}
    for c in cutoffs:
        out[f"flat_{c}"] = []
    for s in iter_sessions(store, cfg, dates):
        events, perbar = detect(s, cfg)
        evs = [e for e in events if e["sig"] != "B00"]
        out["normal"].extend(trades_for(s, perbar, evs, cfg, wanted, grid))
        for c in cutoffs:
            # stop new entries at c AND cover anything still open at c
            out[f"flat_{c}"].extend(trades_for(s, perbar, evs, cfg, wanted, grid, flatten_at=hm(c)))
    return out


def _chunks(dates, k):
    k = max(1, min(k, len(dates)))
    return [list(x) for x in np.array_split(np.array(dates), k) if len(x)]


def run_pass1(cfg: Config, days: list[str], workers: int | None = None):
    chart_dir = DERIVED_DIR / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    for p in chart_dir.glob("*.json"):
        p.unlink()
    workers = workers or max(1, min(4, os.cpu_count() or 1))
    jobs = [(ch, cfg, days, chart_dir) for ch in _chunks(days, workers * 2)]
    evs, segs, rows, grid = [], [], [], None
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        for e, s, g, r in ex.map(_pass1, jobs):
            evs.extend(e)
            segs.extend(s)
            rows.extend(r)
            if grid is None:
                grid = g
            else:
                grid.merge(g)
            log.info("pass 1: %d events so far", len(evs))
    return pd.DataFrame(evs), pd.DataFrame(segs), pd.DataFrame(rows), grid


def run_pass2(cfg: Config, days: list[str], wanted: dict, cutoffs: list[str], workers: int | None = None):
    workers = workers or max(1, min(4, os.cpu_count() or 1))
    jobs = [(ch, cfg, days, wanted, cutoffs) for ch in _chunks(days, workers * 2)]
    out: dict[str, list] = {}
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        for res in ex.map(_pass2, jobs):
            for k, v in res.items():
                out.setdefault(k, []).extend(v)
    return {k: pd.DataFrame(v) for k, v in out.items()}
