# Operating notes for Claude sessions

This repository is the user's Backside Scalping Edge Finder: a rolling 2-year database of
sub-$1, 5M+ share sessions (clipped to the 2 years Massive's free plan serves), a research
engine, and a dashboard artifact named "Backside Scalping Edge Finder". README.md explains
the method.

## Hard rules

- Never publish, quote or describe output built from synthetic data as market results.
  `scripts/publish_plan.py` exits with code 3 on a synthetic build; stop if it does.
- Never fill gaps with invented numbers. If data is missing, say what is missing.
- Never print or commit API keys.
- Keep the published dashboard private unless the user asks otherwise. It shows licensed
  market data.

## Every new container

```bash
pip install -r requirements.txt
python -m pytest -q tests        # must pass before publishing anything
```

Credentials come from the environment: `POLYGON_API_KEY` (or `MASSIVE_API_KEY`), or
`APCA_API_KEY_ID` + `APCA_API_SECRET_KEY`. The provider's host must be allowed by the
environment's network policy (api.polygon.io / api.massive.com, or data.alpaca.markets and
paper-api.alpaca.markets). If a key or host is missing, stop. Tell the user which variable or
host to add, using the environment documentation tool (topics environment.secrets and
environment.network).

## The daily job (also what the scheduled task runs)

1. `python -m bse update --max-minutes 120 --git-checkpoint`. It fetches the newest missing day
   first, then keeps filling older months (newest first) until the time budget runs out, and
   commits + pushes data/ after every month. It is resumable. `python -m bse status` shows
   progress (`months_complete`, `backfill_pending_months`). On the free plan the full 2-year
   backfill takes a day or two of fetching. The daily job fills it in 2 hours at a time,
   and a dedicated long run can use
   `nohup python -m bse update --git-checkpoint > update.log 2>&1 &` with no time budget.
   Publish as soon as the most recent ~3 months are in, then again as history fills in.
2. `python -m bse analyze`: about 10 minutes for 3 months, up to about an hour for the full 2
   years. It analyses only days whose minute bars have arrived, and writes report/site and
   derived/study.pkl. Statistics use the whole window. Example lists and charts cover the most
   recent 126 trading days (artifact size limits).
3. `python -m pytest -q tests`
4. `python scripts/publish_plan.py > plan.json`. Exit code 3 means synthetic: stop.
5. Publish with the Artifact tool, following plan.json:
   - If `artifact_url` is null (first publish): publish `file_path` = plan.page with
     `icon` "chart", `description` "Which signals, times of day and exits give sub-$1 backside
     shorts a real edge, from a rolling 3-month database", `root` = plan.root and
     `files` = batches[0].files. Then publish the remaining batches with `url` set to the new
     artifact, the same `file_path` and `root`. Write `data/meta/artifact.json` as
     `{"url": "<artifact url>"}`.
   - Otherwise, read the artifact first (`action: "read"`, `url`), then publish each batch with
     `url` = plan.artifact_url, `file_path` = plan.page, `root` = plan.root,
     `files` = batch.files. A null in `files` removes a day that left the window.
   - On a conflict, re-read the artifact and publish again. Never pass `force`.
6. `python scripts/publish_plan.py --mark`
7. `git add data && git commit -m "Daily update <last session date>" && git push`
8. Report back in a few lines: sessions added, the window, the artifact link, and any change in
   the three headline answers (plan.json has `answers_now` and `answers_last_published`;
   `--mark` in step 6 saved today's into data/meta/last_answers.json, committed in step 7).

## Scheduled task

After the first successful publish, check whether a scheduled task named
"Backside Edge Finder daily update" exists (list_triggers). If not, create one with
create_trigger:
- `create_new_session_on_fire`: true
- `cron_expression`: `CRON_TZ=America/New_York 12 5 * * 2-6` (5:12 AM ET Tuesday-Saturday:
  the previous session's end-of-day data is published by then, and the dashboard is fresh
  before the open)
- `initiation`: human_request
- `prompt`: "Attach and clone the GitHub repository zlisc-bot/backside-edge-finder with push
  access, then follow 'The daily job' in its CLAUDE.md exactly. If credentials or network
  access are missing, report that and stop."

## Where things are

- Signals and their definitions: `bse/signals.py` (SIGNALS, DEFINITIONS)
- Costs and every threshold: `bse/config.py` (override in `config.local.json`)
- Plain-language answers: `bse/findings.py`
- Dashboard page: `dashboard/index.html` (copied into report/site by `bse/report.py`)
