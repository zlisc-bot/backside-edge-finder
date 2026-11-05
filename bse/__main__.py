"""Command line.

  python -m bse update            fetch anything missing in the rolling window, newest month first
      --max-minutes N             stop after N minutes; the next run resumes
      --git-checkpoint            commit + push data/ after each month (long backfills)
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


def _git_checkpoint(month_days):
    """Commit and push data/ so a long backfill survives a lost container."""
    import subprocess
    from .config import ROOT
    log = logging.getLogger("bse.checkpoint")

    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)

    ident = [] if git("config", "user.email").stdout.strip() else \
        ["-c", "user.name=Backside Edge Finder", "-c", "user.email=noreply@anthropic.com"]
    git("add", "data")
    if git("diff", "--cached", "--quiet").returncode == 0:
        return
    label = f"{month_days[0][:7]} ({len(month_days)} trading days)"
    if git(*ident, "commit", "-q", "-m", f"Backfill checkpoint: {label}").returncode != 0:
        log.warning("checkpoint commit failed")
        return
    for i in range(4):
        git(*ident, "pull", "--rebase", "-q", "origin", "main")
        if git("push", "-q", "origin", "HEAD:main").returncode == 0:
            log.info("checkpoint pushed: %s", label)
            return
        time.sleep(2 ** (i + 1))
    log.warning("checkpoint push failed; the data is committed locally and the next push will include it")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bse")
    ap.add_argument("cmd", choices=["update", "analyze", "all", "status", "synthetic", "report"])
    ap.add_argument("path", nargs="?")
    ap.add_argument("--end", help="last session date (YYYY-MM-DD); default = last completed session")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--months", type=int, default=None, help="override the rolling window length")
    ap.add_argument("--max-minutes", type=float, default=None, help="time budget for update")
    ap.add_argument("--git-checkpoint", action="store_true", help="commit and push data/ after each month")
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
        from .collect import window_dates
        from .store import Store
        st = Store()
        ses = st.read_meta("sessions")
        state = st.state()
        days, w0, w1 = window_dates(cfg, a.end)
        el = ses[ses["eligible_type"]] if len(ses) else ses
        fetched = 0
        if len(el):
            for d, g in el.groupby("date"):
                fetched += len(set(g["ticker"]) & st.minute_tickers(d))
        months = sorted({d[:7] for d in days})
        print(json.dumps({
            "window": [w0, w1], "trading_days": len(days),
            "months_complete": f"{len(set(state.get('complete_months', [])) & set(months))} of {len(months)}",
            "daily_files_in_window": sum(1 for d in days if st.has_daily(d)),
            "qualifying_sessions": int(len(el)), "sessions_with_minute_bars": int(fetched),
            "sessions_with_no_bars": len(state.get("empty_sessions", [])),
            "dates_not_in_plan": len(state.get("unavailable_dates", [])),
            "first_session": el["date"].min() if len(el) else None, "last_session": el["date"].max() if len(el) else None,
            "backfill_pending_months": state.get("backfill_pending_months"), "provider": state.get("provider"),
            "updated_at": state.get("updated_at")}, indent=1, default=str))
        return 0

    if a.cmd in ("update", "all", "synthetic"):
        from . import collect
        prov = None
        if a.cmd == "synthetic":
            from .synthetic import SyntheticProvider
            prov = SyntheticProvider()
        hook = _git_checkpoint if a.git_checkpoint else None
        collect.update(cfg, end=a.end, provider=prov, max_minutes=a.max_minutes, on_block=hook)

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
