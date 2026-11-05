"""Write the dashboard bundle: report/site/index.html + report/site/data/*.json

Every table is stored column-wise ({"cols": [...], "rows": [[...], ...]}) and
floats are rounded, which keeps the files small enough to publish daily.
Chart files hold one trading day each, prices as integer ticks (x10,000),
delta-coded; the page rebuilds the bars and the EMAs (from seeds) itself.
"""
from __future__ import annotations

import json
import logging
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis as an
from . import findings as fd
from .backtest import TRADE_SIGNALS, build_rules
from .config import DERIVED_DIR, REPORT_DIR, ROOT, fmt_hm
from .signals import DEFINITIONS, SIGNALS
from .stages import STAGE_RULES, STAGES
from .validation import window_label


log = logging.getLogger("bse.report")

def _clean(v):
    if isinstance(v, (np.floating, float)):
        f = float(v)
        if not math.isfinite(f):
            return None
        return round(f, 5) if abs(f) < 1 else round(f, 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def table(df: pd.DataFrame, cols: list[str] | None = None) -> dict:
    if df is None or not len(df):
        return {"cols": cols or [], "rows": []}
    cols = [c for c in (cols or list(df.columns)) if c in df.columns]
    sub = df[cols]
    rows = [[_clean(v) for v in r] for r in sub.itertuples(index=False, name=None)]
    return {"cols": cols, "rows": rows}


def _write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), separators=(",", ":"), allow_nan=False))


EVENT_COLS = ["s", "idx", "mod", "price", "fp_atr", "nml_15", "mfe_30", "mae_30", "fwd_30", "rec_vwap_30",
              "nhod_30", "nhod_eod", "mins_since_hod", "pullback", "gain_pc",
              "c_gain", "c_true", "c_age", "c_cross", "c_vwap", "c_price", "c_rt"]
SLICE_DIMS = ("c_gain", "c_true", "c_age", "c_cross", "c_vwap", "c_price", "c_rt")


def _event_slices(ev: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Slice labels as small integer codes; the label lists go in summary.json."""
    e = ev
    src = e[["gain_pc", "after_true_hod", "mins_since_hod", "e9_above_e20", "below_vwap", "price", "known_sub1", "cumvol"]]
    labels = {}
    for dim in SLICE_DIMS:
        # a missing input (e.g. no prior close, so no gain) gives no label: show it as "n/a"
        raw = fd._ctx_events(src, dim)[dim]
        lab = raw.astype(object).where(raw.notna(), "n/a").astype(str)
        cats = sorted(set(lab))
        labels[dim] = cats
        e[dim] = lab.map({c: i for i, c in enumerate(cats)}).astype(int)
    return e, labels


def _week(d: str) -> str:
    """Monday of the week containing trading day d (YYYY-MM-DD)."""
    import datetime as _dt
    x = _dt.date.fromisoformat(d)
    return (x - _dt.timedelta(days=x.weekday())).isoformat()


def _chart_sessions(src: Path) -> list:
    js = json.loads(src.read_text())
    out = []
    for s in js["sessions"]:
        t = np.array(s["t"], dtype=int)
        px = {k: np.round(np.array([np.nan if x is None else x for x in s[k]], dtype=float) * 10000).astype(np.int64)
              for k in ("o", "h", "l", "c", "vw")}
        c = px["c"]
        prev_c = np.concatenate([[c[0]], c[:-1]])
        e9 = s["e9"][0] if s["e9"] else None
        e20 = s["e20"][0] if s["e20"] else None
        out.append({
            "sid": s["sid"], "ticker": s["ticker"], "pc": s["pc"], "ssr_from": s["ssr_from"], "gaps": s["gaps"],
            "t0": int(t[0]), "dt": np.diff(t).tolist(),
            "c0": int(c[0]), "dc": np.diff(c).tolist(),
            "do": (px["o"] - prev_c).tolist(), "dh": (px["h"] - c).tolist(), "dl": (c - px["l"]).tolist(),
            "dv": (px["vw"] - c).tolist(), "v": s["v"],
            # EMA values AT bar 0 (already include the prior-day warm-up)
            "e9_0": e9, "e20_0": e20,
            "st": _rle(s["st"]), "ev": s.get("ev", []),
        })
    return out


def _rle(x):
    out = []
    for v in x:
        if out and out[-1][0] == v:
            out[-1][1] += 1
        else:
            out.append([v, 1])
    return out


def _backfill_state() -> dict:
    from .store import Store
    st = Store().state()
    return {"pending_months": int(st.get("backfill_pending_months") or 0),
            "target_start": st.get("window_start"), "complete_months": len(st.get("complete_months", []))}


def build_site(res: dict, data_source: str, out_dir: Path | None = None) -> Path:
    cfg = res["cfg"]
    out = Path(out_dir or REPORT_DIR / "site")
    data = out / "data"
    if data.exists():
        shutil.rmtree(data)
    data.mkdir(parents=True)
    grid_rules = build_rules(cfg)
    stops, targets, holds = grid_rules
    fnd = fd.build(res)

    days_all = res["days"]
    dd = cfg.report.drill_days
    drill_from = days_all[-dd] if dd and len(days_all) > dd else days_all[0]
    ev, slice_labels = _event_slices(res["events"].copy())
    sess = res["sessions"].sort_values(["date", "ticker"]).reset_index(drop=True)
    sidx = {sid: i for i, sid in enumerate(sess["sid"])}
    ev["s"] = ev["sid"].map(sidx)
    days = res["days"]
    meta = {
        "title": "Backside Scalping Edge Finder",
        "data_source": data_source,
        "synthetic": str(data_source).strip().upper() == "SYNTHETIC",
        "generated_at": pd.Timestamp.now(tz="America/New_York").isoformat(timespec="minutes"),
        "window": {"start": days[0], "end": days[-1], "trading_days": len(days),
                   "months": cfg.universe.window_months},
        "coverage": {"sessions": int(len(sess)), "tickers": int(sess["ticker"].nunique()) if len(sess) else 0,
                     "signals": int((ev["sig"] != "B00").sum()), "control": int((ev["sig"] == "B00").sum()),
                     "trades_shortlist": int(len(res["trades"])), "configs_tested": int(res["cand_tested"]),
                     "candidates_ranked": int(res["cand_n"]),
                     "ssr_sessions": int(sess["ssr_any"].sum()) if len(sess) else 0,
                     "halt_sessions": int((sess["halts"] > 0).sum()) if len(sess) else 0},
        "universe": {"max_low": cfg.universe.max_low_price, "min_volume": cfg.universe.min_daily_volume,
                     "otc": cfg.universe.include_otc, "types": list(cfg.universe.ticker_types),
                     "adjusted": cfg.universe.adjusted},
        "test_from": res["test_from"], "drill_from": drill_from,
        "backfill": _backfill_state(),
        "signals": {k: {"name": v[0], "group": v[1], "ref": v[2], "def": DEFINITIONS[k]} for k, v in SIGNALS.items()},
        "stages": {str(k): {"name": v, "rule": STAGE_RULES.get(k, "")} for k, v in STAGES.items()},
        "stops": {r.code: r.label for r in stops}, "targets": {r.code: r.label for r in targets},
        "timings": fd.TIMING, "holds": list(holds),
        "costs": {k: getattr(cfg.costs, k) for k in cfg.costs.__dataclass_fields__},
        "cost_scenarios": {k: v for k, v in __import__("bse.config", fromlist=["COST_SCENARIOS"]).COST_SCENARIOS.items()},
        "config": cfg.to_dict(),
        "findings": fnd["findings"], "kpis": fnd["kpis"], "slice_labels": slice_labels,
    }
    _write(data / "summary.json", meta)

    _write(data / "signals.json", table(res["signals"]))
    _write(data / "signal_tod.json", table(res["signal_tod"]))
    _write(data / "signal_coarse.json", table(res["signal_coarse"]))
    _write(data / "context.json", table(res["context"]))
    _write(data / "tod_profile.json", table(res["tod_profile"]))

    fin = res["finalists"].copy()
    if len(fin):
        fin["label"] = [fd.strat_label(r) for r in fin.to_dict("records")]
    wf = res["wf"]
    folds = [{**{k: v for k, v in f.items() if k != "pick"}, **{f"pick_{k}": v for k, v in f["pick"].items()}}
             for f in wf.get("folds", [])]
    split_top = pd.DataFrame(res["split"].get("top", []))
    if len(split_top):
        split_top["label"] = [fd.strat_label(r) for r in split_top.to_dict("records")]
    _write(data / "strategies.json", {
        "finalists": table(fin), "wf": {k: v for k, v in wf.items() if k != "folds"}, "wf_folds": folds,
        "split": {k: v for k, v in res["split"].items() if k != "top"}, "split_top": table(split_top),
        "pbo": res["pbo"], "marginals": table(res["param_marginals"]),
        "cutoffs": table(res["cutoffs"]), "start_heat": table(res["start_heat"]),
        "marginal_tod": table(res["marginal_tod"]), "cut_cids": [int(c) for c in res["cut_cids"]],
    })
    tr = res["trades"].copy()
    if len(tr):
        tr = tr[tr["date"] >= drill_from].copy()
        tr["s"] = tr["sid"].map(sidx)
    _write(data / "trades.json", table(tr, ["cid", "s", "sig_idx", "entry_idx", "entry_mod", "entry_px", "exit_idx",
                                            "exit_mod", "exit_px", "reason", "r", "ssr"]))
    _write(data / "stages.json", {"table": table(res["stages"]), "transitions": table(res["stage_tm"])})
    segs = res["segments"].copy()
    if len(segs):
        segs = segs[segs["date"] >= drill_from].copy()
        segs["s"] = segs["sid"].map(sidx)
    for k in sorted(set(segs["stage"])) if len(segs) else []:
        _write(data / "segments" / f"{int(k)}.json",
               table(segs[segs["stage"] == k], ["s", "next", "a", "b", "start_mod", "end_mod", "dur_min",
                                                "range_pct", "fwd_30"]))
    ev_list = ev[ev["date"] >= drill_from]
    for code in sorted(set(ev_list["sig"])):
        sub = ev_list[ev_list["sig"] == code]
        if code == "B00" and len(sub) > 30000:
            sub = sub.sample(30000, random_state=1).sort_values(["s", "idx"])
        _write(data / "events" / f"{code}.json", table(sub, EVENT_COLS))
    _write(data / "sessions.json", table(sess, ["sid", "ticker", "date", "pc", "hod", "hod_mod", "lod", "close",
                                                "volume", "ssr_any", "halts", "max_stage"]))

    # Session charts: one file per week (named by its Monday), newest week first,
    # until the size budget is used. A 2-year window is ~105 files.
    charts_src = DERIVED_DIR / "charts"
    cd = cfg.report.chart_days
    chart_from = days_all[-cd] if cd and len(days_all) > cd else days_all[0]
    in_window = set(days)
    weeks: dict[str, list[Path]] = {}
    for p in sorted(charts_src.glob("*.json")):
        if p.stem in in_window and p.stem >= chart_from:
            weeks.setdefault(_week(p.stem), []).append(p)
    budget, used, first_day = cfg.report.chart_budget_mb * 1024 * 1024, 0, None
    for w in sorted(weeks, reverse=True):
        sessions = [x for p in weeks[w] for x in _chart_sessions(p)]
        blob = json.dumps(_clean({"week": w, "sessions": sessions}), separators=(",", ":"), allow_nan=False)
        if used + len(blob) > budget:
            log.warning("chart budget (%d MB) reached: charts start at %s", cfg.report.chart_budget_mb, first_day)
            break
        (data / "charts").mkdir(parents=True, exist_ok=True)
        (data / "charts" / f"w-{w}.json").write_text(blob)
        used += len(blob)
        first_day = weeks[w][0].stem
    meta["chart_from"] = first_day or "9999-12-31"
    meta["chart_files"] = "week"
    meta["chart_mb"] = round(used / 1024 / 1024, 1)
    _write(data / "summary.json", meta)

    page = (ROOT / "dashboard" / "index.html").read_text()
    (out / "index.html").write_text(page)
    return out
