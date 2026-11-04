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


def window_dates(cfg: Config, end: str | None = None) -> tuple[list[str], str, str]:
    end = end or cal.last_complete_session()
    start = cal.months_back(end, cfg.universe.window_months)
    days = cal.trading_days(start, end)
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


def update(cfg: Config | None = None, end: str | None = None, provider: Provider | None = None,
           store: Store | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    store = store or Store()
    provider = provider or get_provider(cfg)
    days, start, end = window_dates(cfg, end)
    pre = [cal.prev_trading_day(start, 2), cal.prev_trading_day(start, 1)]
    log.info("window %s .. %s (%d trading days), provider=%s", start, end, len(days), provider.name)

    # 1. daily bars ------------------------------------------------------------
    missing = [d for d in pre + days if not store.has_daily(d)]
    if missing:
        log.info("fetching daily bars for %d dates", len(missing))
        daily = provider.daily_bars(missing)
        for d in missing:
            part = daily[daily["date"] == d]
            if not len(part):
                log.warning("%s: provider returned no daily bars - treated as a non-trading day", d)
            store.write_daily(d, part)

    # splits (refetch each run: cheap, and late-announced splits get picked up)
    try:
        store.write_meta("splits", provider.splits(pre[0], end))
    except ProviderError as e:
        log.warning("split lookup failed: %s", e)

    # 2. screen -----------------------------------------------------------------
    ses = build_sessions(store, cfg, days)
    if not len(ses):
        log.warning("no qualifying sessions found")
        return ses

    # 3. security types for new tickers ------------------------------------------
    types = store.read_meta("types")
    known = set(types["ticker"]) if len(types) else set()
    new = sorted(set(ses["ticker"]) - known)
    if new:
        log.info("classifying %d new tickers", len(new))
        t_new = provider.security_types(new, [end, start])
        types = pd.concat([types, t_new], ignore_index=True) if len(types) else t_new
        store.write_meta("types", types.drop_duplicates("ticker", keep="last"))
        ses = build_sessions(store, cfg, days)

    store.write_meta("sessions", ses)
    keep = ses[ses["eligible_type"]]
    log.info("%d qualifying sessions (%d after removing non-stock securities), %d tickers",
             len(ses), len(keep), keep["ticker"].nunique())

    # 4. minute bars ----------------------------------------------------------------
    st = store.state()
    empty = set(st.get("empty_sessions", []))
    todo = []
    have: dict[str, set] = {}
    for r in keep.itertuples():
        if r.date not in have:
            have[r.date] = store.minute_tickers(r.date)
        if r.ticker not in have[r.date] and r.session_id not in empty:
            todo.append((r.ticker, r.date))
    prev_day = {d: cal.prev_trading_day(d) for d in set(keep["date"])}
    log.info("%d sessions need minute bars", len(todo))

    qual = set(zip(keep["ticker"], keep["date"]))
    wanted = defaultdict(set)  # ticker -> dates to store (session dates + their prior days)
    for t, d in todo:
        wanted[t].update({d, prev_day[d]})
    buf: dict[tuple, list[pd.DataFrame]] = defaultdict(list)
    got = set()
    n_resp = 0
    warm_have: dict[str, set] = {}

    def flush():
        for (d, kind), frames in buf.items():
            store.append_minute(d, pd.concat(frames, ignore_index=True), kind)
        buf.clear()

    for df in provider.minute_bars(todo, prev_day):
        if not len(df):
            continue
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
            log.info("minute bars: %d responses stored", n_resp)
    flush()

    for t, d in todo:
        if (t, d) not in got:
            empty.add(f"{t}_{d}")
    st["empty_sessions"] = sorted(empty)
    st["last_update_end"] = end
    st["provider"] = provider.name
    st["updated_at"] = pd.Timestamp.now(tz=TZ).isoformat()
    store.save_state(st)

    # optional: today's borrow flags, so shortability history accumulates
    if hasattr(provider, "borrow_snapshot"):
        try:
            store.write_borrow(end, provider.borrow_snapshot())
        except ProviderError as e:
            log.warning("borrow snapshot failed: %s", e)
    return ses
