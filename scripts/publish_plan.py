#!/usr/bin/env python3
"""Plan (and record) a publish of report/site to the dashboard artifact.

  python scripts/publish_plan.py            print the plan as JSON (exit 3 if the
                                            site was built from synthetic data)
  python scripts/publish_plan.py --mark     record the current files as published

The plan lists only files whose content changed since the last recorded
publish, split into batches that respect the artifact limits (<= 240 files and
<= 60 MB per publish), plus paths to remove (files no longer produced, e.g. a
trading day that left the rolling window).

State lives in data/meta/published.json and data/meta/artifact.json, which are
committed with the database so the next session can continue.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "report" / "site"
META = ROOT / "data" / "meta"
PUB = META / "published.json"
ART = META / "artifact.json"
MAX_BYTES = 60 * 1024 * 1024
MAX_FILES = 240


def manifest():
    out = {}
    for p in sorted(SITE.rglob("*")):
        if p.is_file() and p.name != "index.html":
            rel = p.relative_to(SITE).as_posix()
            out[rel] = {"sha": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mark", action="store_true")
    ap.add_argument("--all", action="store_true", help="plan every file, not only changed ones")
    a = ap.parse_args()
    if not (SITE / "index.html").exists():
        print("report/site/index.html is missing - run `python -m bse analyze` first", file=sys.stderr)
        return 2
    summary = json.loads((SITE / "data" / "summary.json").read_text())
    if summary.get("synthetic") or str(summary.get("data_source", "")).upper() == "SYNTHETIC":
        print("REFUSING: this site was built from SYNTHETIC test data. Never publish it.", file=sys.stderr)
        return 3
    cur = manifest()
    answers = {"window": summary.get("window"), "generated_at": summary.get("generated_at"),
               "answers": [{k: f.get(k) for k in ("id", "status", "answer")} for f in summary.get("findings", [])[:3]],
               "kpis": [{k: x.get(k) for k in ("id", "title", "value", "text")} for x in summary.get("kpis", [])]}
    last = META / "last_answers.json"
    if a.mark:
        META.mkdir(parents=True, exist_ok=True)
        PUB.write_text(json.dumps(cur, indent=0, sort_keys=True))
        last.write_text(json.dumps(answers, indent=1))
        print(f"recorded {len(cur)} files as published")
        return 0
    old = {} if a.all or not PUB.exists() else json.loads(PUB.read_text())
    changed = [k for k, v in cur.items() if old.get(k, {}).get("sha") != v["sha"]]
    removed = sorted(set(old) - set(cur))
    # charts last: the page works without them, and they are the bulk
    changed.sort(key=lambda k: (k.startswith("data/charts/"), k))
    batches, cur_b, size = [], [], 0
    for k in changed:
        b = cur[k]["bytes"]
        if cur_b and (size + b > MAX_BYTES or len(cur_b) >= MAX_FILES):
            batches.append(cur_b)
            cur_b, size = [], 0
        cur_b.append(k)
        size += b
    if cur_b:
        batches.append(cur_b)
    art = json.loads(ART.read_text()) if ART.exists() else {}
    plan = {
        "artifact_url": art.get("url"),
        "page": str(SITE / "index.html"),
        "root": str(SITE),
        "window": summary.get("window"),
        "data_source": summary.get("data_source"),
        "removed": removed,
        # pass each batch's "files" map as the Artifact `files` argument with `root` set to
        # plan["root"]; null removes a file from the published artifact
        "batches": [{"files": {**{k: k for k in b}, **({r: None for r in removed} if i == 0 else {})},
                     "bytes": sum(cur[k]["bytes"] for k in b)} for i, b in enumerate(batches)]
                   or ([{"files": {r: None for r in removed}, "bytes": 0}] if removed else []),
        "unchanged": len(cur) - len(changed),
        "answers_now": answers,
        "answers_last_published": json.loads(last.read_text()) if last.exists() else None,
    }
    print(json.dumps(plan, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
