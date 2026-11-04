"""Command line.

  python -m bse update            fetch anything missing in the rolling window
  python -m bse analyze           run the study and rebuild report/site
  python -m bse all               update + analyze (what the daily job runs)
  python -m bse report            rebuild report/site from the last analysis (no recompute)
  python -m bse status            what is in the database
  python -m bse synthetic DIR     build a SYNTHETIC test database in DIR and analyze it
                                  (software test only - never publish its output)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bse")
    ap.add_argument("cmd", choices=["update", "analyze", "all", "status", "synthetic", "report"])
    ap.add_argument("path", nargs="?")
    ap.add_argument("--end", help="last session date (YYYY-MM-DD); default = last completed session")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--months", type=int, default=None, help="override the rolling window length")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if a.cmd == "synthetic":
        base = a.path or "synthetic-test"
        os.environ["BSE_DATA_DIR"] = os.path.join(base, "data")
        os.environ["BSE_DERIVED_DIR"] = os.path.join(base, "derived")
        os.environ["BSE_REPORT_DIR"] = os.path.join(base, "report")

    from .config import load_config
    cfg = load_config()
    if a.months:
        cfg.universe.window_months = a.months

    if a.cmd == "status":
        from .store import Store
        st = Store()
        ses = st.read_meta("sessions")
        print(json.dumps({"state": st.state(), "daily_files": len(st.daily_dates()),
                          "sessions": int(len(ses)), "eligible": int(ses["eligible_type"].sum()) if len(ses) else 0,
                          "first": ses["date"].min() if len(ses) else None,
                          "last": ses["date"].max() if len(ses) else None}, indent=1, default=str))
        return 0

    if a.cmd in ("update", "all", "synthetic"):
        from . import collect
        prov = None
        if a.cmd == "synthetic":
            from .synthetic import SyntheticProvider
            prov = SyntheticProvider()
        collect.update(cfg, end=a.end, provider=prov)

    if a.cmd in ("analyze", "all", "synthetic"):
        from .pipeline import analysis_days
        from .report import build_site
        from .store import Store
        from .study import run_study
        store = Store()
        days = analysis_days(store, cfg, a.end)
        if len(days) < 5:
            print(f"only {len(days)} trading days in the database - collect more first", file=sys.stderr)
            return 2
        t = time.time()
        res = run_study(cfg, days, a.workers)
        src = "SYNTHETIC" if a.cmd == "synthetic" else (store.state().get("provider") or "unknown").title()
        res["data_source"] = src
        import pickle
        from .config import DERIVED_DIR
        with open(DERIVED_DIR / "study.pkl", "wb") as f:
            pickle.dump({k: v for k, v in res.items() if k != "grid"}, f)
        out = build_site(res, src)
        print(f"site written to {out} in {time.time() - t:.0f}s")

    if a.cmd == "report":
        import pickle
        from .config import DERIVED_DIR
        from .report import build_site
        with open(DERIVED_DIR / "study.pkl", "rb") as f:
            res = pickle.load(f)
        print(f"site written to {build_site(res, res.get('data_source', 'unknown'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
