"""Build and extend the historical database.

`update()` is idempotent and resumable: run it daily (or after an interruption)
and it only fetches what is missing.

  1. Whole-market daily bars for every trading day in the window
     (+2 days before it for prior closes and Rule 201 carry-over).
  2. Screen: official daily low < $1.00 and daily volume >= 5,000,000.
     No other filter - not price direction, not popularity, not outcome.
  3. Drop non-stock securities (warrants, units, rights, preferreds, ETFs).
  4. 1-minute bars (04:00-20:00 ET) for each qualifying session and its prior
     trading day (EMA warm-up only).
"""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict

import pandas as pd

from . import calendar as cal
from .config import TZ, Config, load_config
from .providers.base import Provider, ProviderError
from .store import Store

log = logging.getLogger("bse.collect")


def get_provider(cfg: Config) -> Provider:
    choice = os.environ.get("BSE_PROVIDER", "").lower()
    has_poly = bool(os.environ.get("POLYGON_API_KEY") or os.environ.get("MASSIVE_API_KEY"))
    has_alp = bool(os.environ.get("APCA_API_KEY_ID") and os.environ.get("APCA_API_SECRET_KEY"))
    if choice == "polygon" or (not choice and has_poly):
        from .providers.polygon import PolygonProvider
        return PolygonProvider(include_otc=cfg.universe.include_otc)
    if choice == "alpaca" or (not choice and has_alp):
        from .providers.alpaca import AlpacaProvider
        return AlpacaProvider(include_otc=cfg.universe.include_otc)
    raise ProviderError(
        "No market data credentials found. Set POLYGON_API_KEY (Polygon.io / Massive) or "
        "APCA_API_KEY_ID + APCA_API_SECRET_KEY (Alpaca) in the environment, and allow the "
        "provider's API host in the environment's network settings.")


def window_start(cfg: Config, end: str) -> str:
    """First calendar date of the rolling window ending at `end`, clipped to
    the history the data plan serves."""
    start = cal.months_back(end, cfg.universe.window_months)
    lim = cfg.universe.max_history_days
    if lim:
        from datetime import date, timedelta
        start = max(start, (date.fromisoformat(end) - timedelta(days=lim)).isoformat())
    return start


def window_dates(cfg: Config, end: str | None = None) -> tuple[list[str], str, str]:
    end = end or cal.last_complete_session()
    days = cal.trading_days(window_start(cfg, end), end)
    return days, days[0], days[-1]


def _split_ratio_map(splits: pd.DataFrame) -> dict[tuple[str, str], float]:
    if splits is None or not len(splits):
        return {}
    return {(r.ticker, str(r.execution_date)[:10]): float(r.ratio) for r in splits.itertuples()}


def build_sessions(store: Store, cfg: Config, days: list[str]) -> pd.DataFrame:
    """Screen the stored daily bars. Returns one row per qualifying session."""
    pre = [cal.prev_trading_day(days[0], 2), cal.prev_trading_day(days[0], 1)]
    daily = store.read_daily(pre + days)
    if not len(daily):
        return pd.DataFrame()
    daily = daily.sort_values(["ticker", "date"])
    splits = _split_ratio_map(store.read_meta("splits"))

    u = cfg.universe
    cand = daily[(daily["date"].isin(set(days))) & (daily["l"] < u.max_low_price) & (daily["v"] >= u.min_daily_volume)]
    if not len(cand):
        return pd.DataFrame()

    key = daily.set_index(["ticker", "date"])
    rows = []
    for r in cand.itertuples(index=False):
        p1 = cal.prev_trading_day(r.date)
        p2 = cal.prev_trading_day(p1)
        prev = key.loc[(r.ticker, p1)] if (r.ticker, p1) in key.index else None
        prev2 = key.loc[(r.ticker, p2)] if (r.ticker, p2) in key.index else None
        ratio_today = splits.get((r.ticker, r.date), 1.0)
        ratio_prev = splits.get((r.ticker, p1), 1.0)
        pc = float(prev["c"]) * ratio_today if prev is not None else float("nan")
        prev_low = float(prev["l"]) if prev is not None else float("nan")
        pc2 = float(prev2["c"]) * ratio_prev if prev2 is not None else float("nan")
        # Rule 201 carry-over: a 10% drop from the prior close on day P keeps
        # the restriction in force for all of the next trading day.
        ssr_carry = bool(prev is not None and prev2 is not None and prev_low <= 0.9 * pc2)
        rows.append(dict(ticker=r.ticker, date=r.date, prev_date=p1, o=r.o, h=r.h, l=r.l, c=r.c,
                         v=float(r.v), vw=r.vw, n=r.n, pc=pc, prev_low=prev_low, pc2=pc2,
                         ssr_carry=ssr_carry, split_today=ratio_today))
    ses = pd.DataFrame(rows)

    types = store.read_meta("types")
    if len(types):
        ses = ses.merge(types[["ticker", "type", "exchange", "name"]], on="ticker", how="left")
    else:
        ses["type"], ses["exchange"], ses["name"] = "UNKNOWN", "", ""
    ses["type"] = ses["type"].fillna("UNKNOWN")
    ses["eligible_type"] = ses["type"].isin(set(u.ticker_types))
    ses["session_id"] = ses["ticker"] + "_" + ses["date"]
    return ses.sort_values(["date", "ticker"]).reset_index(drop=True)


def _month_blocks(days: list[str]) -> list[list[str]]:
    """Trading days grouped by calendar month, NEWEST month first, so a long
    backfill delivers the most recent data first."""
    by = defaultdict(list)
    for d in days:
        by[d[:7]].append(d)
    return [by[k] for k in sorted(by, reverse=True)]


def _merge_meta(store: Store, name: str, df: pd.DataFrame, key: list[str]):
    old = store.read_meta(name)
    both = pd.concat([old, df], ignore_index=True) if len(old) else df
    store.write_meta(name, both.drop_duplicates(key, keep="last"))


def _update_block(cfg: Config, store: Store, provider: Provider, bdays: list[str], deadline: float | None,
                  settled_before: str = "9999-12-31") -> bool:
    """Fetch everything one month needs. Returns True when every qualifying
    session of the month was fetched (or confirmed to have no bars).

    settled_before: data for dates before this is final. An empty answer for a
    later date may just mean the provider has not published it yet, so it is
    not saved and the next run asks again."""
    pre = [cal.prev_trading_day(bdays[0], 2), cal.prev_trading_day(bdays[0], 1)]
    st = store.state()
    unavailable = set(st.get("unavailable_dates", []))

    # 1. daily bars (the screen) -------------------------------------------------
    need = [d for d in pre + bdays if not store.has_daily(d) and d not in unavailable]
    if need:
        log.info("%s: daily bars for %d dates", bdays[0][:7], len(need))
        daily = provider.daily_bars(need)
        for d in need:
            if d in getattr(provider, "unavailable", set()):
                unavailable.add(d)
                continue
            part = daily[daily["date"] == d]
            if not len(part):
                if d >= settled_before:
                    log.warning("%s: no daily bars yet (not published?) - not saved; the next run retries", d)
                    continue
                log.warning("%s: provider returned no daily bars - treated as a non-trading day", d)
            store.write_daily(d, part)
        st = store.state()
        st["unavailable_dates"] = sorted(unavailable)
        store.save_state(st)

    # splits for this month (cheap; late-announced splits get picked up on the next pass)
    try:
        sp = provider.splits(pre[0], bdays[-1])
        if len(sp):
            _merge_meta(store, "splits", sp, ["ticker", "execution_date"])
    except ProviderError as e:
        log.warning("split lookup failed: %s", e)

    # 2. screen + 3. security types for tickers not seen before ------------------------
    ses = build_sessions(store, cfg, bdays)
    if not len(ses):
        log.info("%s: no qualifying sessions", bdays[0][:7])
        return True
    types = store.read_meta("types")
    known = set(types["ticker"]) if len(types) else set()
    new = sorted(set(ses["ticker"]) - known)
    if new:
        bulk = len(new) > 40           # bulk lists cost ~13 calls; one-by-one costs 1 call each
        log.info("%s: classifying %d new tickers (%s)", bdays[0][:7], len(new), "bulk" if bulk else "one by one")
        t_new = provider.security_types(new, [bdays[len(bdays) // 2]], bulk=bulk)
        _merge_meta(store, "types", t_new, ["ticker"])
        ses = build_sessions(store, cfg, bdays)
    keep = ses[ses["eligible_type"]]
    log.info("%s: %d qualifying sessions (%d after removing non-stock securities), %d tickers",
             bdays[0][:7], len(ses), len(keep), keep["ticker"].nunique())

    # 4. minute bars ----------------------------------------------------------------------
    empty = set(st.get("empty_sessions", []))
    todo, have = [], {}
    for r in keep.itertuples():
        if r.date not in have:
            have[r.date] = store.minute_tickers(r.date)
        if r.ticker not in have[r.date] and r.session_id not in empty:
            todo.append((r.ticker, r.date))
    if not todo:
        return True
    prev_day = {d: cal.prev_trading_day(d) for d in set(keep["date"])}
    log.info("%s: %d sessions need minute bars", bdays[0][:7], len(todo))
    qual = set(zip(keep["ticker"], keep["date"]))
    wanted = defaultdict(set)
    for t, d in todo:
        wanted[t].update({d, prev_day[d]})
    buf: dict[tuple, list[pd.DataFrame]] = defaultdict(list)
    got, warm_have = set(), {}
    n_resp, finished = 0, True

    def flush():
        for (d, kind), frames in buf.items():
            store.append_minute(d, pd.concat(frames, ignore_index=True), kind)
        buf.clear()

    for df in provider.minute_bars(todo, prev_day):
        if len(df):
            dt = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(TZ)
            df = df.assign(date=dt.dt.strftime("%Y-%m-%d"))
            t = df["ticker"].iloc[0]
            for d, part in df.groupby("date"):
                if d not in wanted.get(t, ()):
                    continue
                if (t, d) in qual:
                    buf[(d, "minute")].append(part.drop(columns=["date"]))
                    got.add((t, d))
                else:
                    if d not in warm_have:
                        warm_have[d] = store.minute_tickers(d, "warm")
                    if t not in warm_have[d]:
                        buf[(d, "warm")].append(part.drop(columns=["date"]))
                        warm_have[d].add(t)
        n_resp += 1
        if n_resp % 25 == 0:
            flush()
            log.info("%s: minute bars, %d responses stored", bdays[0][:7], n_resp)
        if deadline and time.monotonic() > deadline:
            log.info("time budget reached inside %s; it resumes on the next run", bdays[0][:7])
            finished = False
            break
    flush()
    if finished:
        # only now is "asked for it, got nothing" a real empty session
        st = store.state()
        empty = set(st.get("empty_sessions", []))
        empty.update(f"{t}_{d}" for t, d in todo if (t, d) not in got and d < settled_before)
        st["empty_sessions"] = sorted(empty)
        store.save_state(st)
    return finished


def update(cfg: Config | None = None, end: str | None = None, provider: Provider | None = None,
           store: Store | None = None, max_minutes: float | None = None, on_block=None) -> pd.DataFrame:
    """Bring the database up to date, newest month first.

    max_minutes: stop after this much time (between months, or inside one);
                 the next run resumes. on_block(month_days): called after each
                 month, e.g. to commit and push a checkpoint."""
    cfg = cfg or load_config()
    store = store or Store()
    provider = provider or get_provider(cfg)
    days, start, end = window_dates(cfg, end)
    log.info("window %s .. %s (%d trading days), provider=%s", start, end, len(days), provider.name)
    t0 = time.monotonic()
    deadline = t0 + max_minutes * 60 if max_minutes else None
    blocks = _month_blocks(days)
    from datetime import date as _date, timedelta as _td
    settled_before = (_date.fromisoformat(end) - _td(days=4)).isoformat()
    st = store.state()
    done = set(st.get("complete_months", []))
    current = end[:7]
    pending = 0
    for i, bdays in enumerate(blocks):
        month = bdays[0][:7]
        if month in done and month != current:
            continue
        if deadline and time.monotonic() > deadline:
            pending = sum(1 for b in blocks[i:] if b[0][:7] not in done or b[0][:7] == current)
            log.info("time budget reached; %d month(s) left for the next run", pending)
            break
        complete = _update_block(cfg, store, provider, bdays, deadline, settled_before)
        st = store.state()
        if complete and month != current:
            st["complete_months"] = sorted(set(st.get("complete_months", [])) | {month})
        st.update(provider=provider.name, updated_at=pd.Timestamp.now(tz=TZ).isoformat(),
                  window_start=start, window_end=end)
        store.save_state(st)
        have_days = [d for d in days if store.has_daily(d)]
        if have_days:
            store.write_meta("sessions", build_sessions(store, cfg, have_days))
        if on_block:
            on_block(bdays)
        if not complete:
            pending = 1
            break
    st = store.state()
    st["last_update_end"] = end
    st["backfill_pending_months"] = pending
    store.save_state(st)

    # optional: today's borrow flags, so shortability history accumulates
    if hasattr(provider, "borrow_snapshot"):
        try:
            store.write_borrow(end, provider.borrow_snapshot())
        except ProviderError as e:
            log.warning("borrow snapshot failed: %s", e)
    return store.read_meta("sessions")
