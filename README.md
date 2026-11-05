# Backside Scalping Edge Finder

A research engine and dashboard for one question: **when does a high-volume sub-$1 stock
stop being a long and start being a short, and does shorting it pay after realistic costs?**

It builds a rolling 2-year database of every qualifying session, tests 21 backside signals
against a random-timing control, classifies every bar into 8 frontside/backside stages,
backtests roughly 100,000 entry/stop/exit/hold combinations across 21 time-of-day windows,
and reports only what survives out-of-sample checks. Every number on the dashboard links to
the sessions behind it.

## What counts as a qualifying session

- The official daily bar traded **below $1.00** at any point, with **5,000,000+ shares** of total
  daily volume (extended hours included).
- The **whole session** is kept (04:00 to 16:00 ET for analysis), including any time above $1.
- Nothing is filtered by outcome: runners, faders, halts and failed breakouts all count.
- Exchange-listed common stock and ADRs only. OTC, warrants, units, rights, preferreds and funds are excluded.
- **Unadjusted prices.** A stock that later reverse-split still counts at the price it actually traded.
- Rolling window: the most recent 24 months, clipped to the 2 years of history Massive's free plan
  serves (`universe.max_history_days`; set it to 0 on a paid plan). Raw data is never deleted.
- The backfill runs newest month first and saves to GitHub after each month, so the dashboard
  can publish as soon as recent months are in and fill backwards from there.

## Data source (required)

No market data is bundled. Add **one** of these to the cloud environment:

| Provider | Environment variables | Allowed hosts | Notes |
|---|---|---|---|
| Polygon.io / Massive (recommended) | `POLYGON_API_KEY` (or `MASSIVE_API_KEY`) | `api.polygon.io`, `api.massive.com` | One whole-market call per day for the screen. Free plan: 5 requests/min, so the first backfill takes a few hours; any paid plan takes minutes. |
| Alpaca | `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY` | `data.alpaca.markets`, `paper-api.alpaca.markets` | Needs SIP historical bars (`APCA_FEED=sip`). IEX-only volume would break the 5M-share screen. |

Put keys in the environment settings, never in this repository.

## Running it

```bash
pip install -r requirements.txt
python -m bse update      # fetch anything missing in the window (resumable, newest month first)
python -m bse update --max-minutes 120 --git-checkpoint   # what the daily job runs
python -m bse analyze     # run the study and build report/site
python -m bse all         # both; this is what the daily job runs
python -m bse status      # what is in the database
python -m pytest -q tests # causality, fill-logic and labeling tests
```

`python -m bse synthetic DIR` builds a **synthetic** database for testing the software only.
Its output is stamped SYNTHETIC, and `scripts/publish_plan.py` refuses to publish it.

## How the answers are computed

**Signals** (`bse/signals.py`). Every new high of day starts an *epoch*; each signal fires at
most once per epoch, at a bar close, using only data known at that close. Swing highs and lows
need 2 bars on each side and are timestamped when they are confirmed. A truncation test
(`tests/test_causality.py`) cuts each session at random minutes and checks that nothing
already known changes, which proves the absence of look-ahead.

**Outcomes** (`bse/outcomes.py`). A signal *worked* if price fell 1 ATR before rising 1 ATR within
30 minutes. It *failed* if the rise came first; a bar touching both counts as failed. Also measured:
new low of the move within 5/15/30/60 min, new low of day, 5/10/20% follow-through, max drop and
max rise (mean and median), time to the next low, reclaims of the 9 EMA, 20 EMA and VWAP, and
new highs.

**Random-timing control.** Every 5th bar after a high is sampled with no signal required.
Lift = signal edge minus control edge in the same 15-minute buckets, with 95% intervals from
resampling whole sessions. A signal with no lift is no better than shorting at a random moment.

**Stages** (`bse/stages.py`): 1 initial momentum, 2 strong frontside, 3 weakening frontside,
4 potential backside (early warning), 5 confirmed backside, 6 extended backside, 7 possible
exhaustion, 8 reversal / renewed frontside. Each stage is assigned causally, bar by bar.

**Strategies** (`bse/backtest.py`). Each signal is tested with 5 entry timings (signal close, next open,
failed reclaim, break of the signal low, lower high below the 20 EMA), 15 stops (9 EMA / 20 EMA /
VWAP / lower-high stop orders, 1-5% fixed, 1-3 ATR, close-above-level exits), 13 exits (0.25% to 10%
targets, cover at the next low, 2-5% and ATR trailing stops, none) and 5 maximum holds. Each
combination is then evaluated in 21 time-of-day windows.

Costs on every trade: an estimated spread (Abdi-Ranaldo, floored at 2 ticks or 0.10%), 5 bps
slippage, stop slippage of 10% of the bar range, $0.002/share commission (min $1, cap 0.5%),
and a 0.3% locate. Rule 201 is applied: under SSR a short must rest on the offer and fill on an
uptick, and break-of-low entries are skipped. Halts appear as gaps, and stops fill at the reopen.
Edit `config.local.json` to use your broker's real costs, for example:

```json
{"costs": {"commission_per_share": 0.0035, "commission_cap_pct": 0.01, "locate_pct": 0.005}}
```

**Out-of-sample checks** (`bse/validation.py`): weekly walk-forward (each week traded with the setup
chosen from earlier weeks only), a 60/40 chronological split, and the CSCV probability of backtest
overfitting. Cutoff tests compare stopping new entries at 10:30, 11:00, 11:30, 12:00 and 13:00, or
trading all day, both with normal exits and with everything covered at the cutoff.

## Known limits

- No quote data: spreads are estimated. The low/high cost scenarios show how sensitive results are.
- Historical borrow availability and locate prices do not exist in the data. Every short is assumed
  borrowable at the stated cost. With Alpaca, a daily shortable snapshot accumulates going forward.
- Shorts under $5 need the greater of $2.50/share or 100% of value as maintenance margin (FINRA 4210).
  Many brokers will not short sub-$1 stocks at all.
- The universe uses end-of-day information. A separate split keeps only signals that were
  screenable live (already below $1 and 1M+ shares by the signal time).
- Every session in the window has a chart on the dashboard (one chart file per week, capped at
  200 MB, newest weeks first; `chart_days` and `chart_budget_mb` in `bse/config.py`). Example
  lists of individual signals and trades cover the most recent 126 trading days (artifact size
  limits). Every statistic uses the full window.

## Layout

```
bse/            engine (providers, store, indicators, signals, outcomes, stages, backtest, validation,
                analysis, findings, report)
dashboard/      the dashboard page (index.html)
data/           the database (Parquet, committed; write-once daily files)
derived/        intermediate results (not committed)
report/site/    the built dashboard bundle (not committed; published as an artifact)
scripts/        publish_plan.py
tests/          causality, backtest and outcome tests
```
