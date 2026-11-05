"""Polygon.io / Massive.com REST adapter (same API, two hostnames).

Env:
  POLYGON_API_KEY (or MASSIVE_API_KEY)   required
  POLYGON_BASE_URL                       default: https://api.massive.com, falling back to
                                         https://api.polygon.io if Massive is unreachable
  POLYGON_RATE_PER_MIN                   default 0 (unlimited, slows down on 429).
                                         The free plan allows 5/min.

Endpoints used (all with adjusted=false):
  /v2/aggs/grouped/locale/us/market/stocks/{date}     whole market, one call per day
  /v2/aggs/ticker/{T}/range/1/minute/{from}/{to}      minute bars, 04:00-20:00 ET
  /v3/reference/splits                                reverse/forward splits
  /v3/reference/tickers                               security type (CS, ADRC, ...)
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import date as Date

import pandas as pd
import requests

from .base import BAR_COLS, DAILY_COLS, HttpClient, PlanLimitError, Provider, ProviderError, RateLimiter

log = logging.getLogger("bse.polygon")
BASES = ("https://api.massive.com", "https://api.polygon.io")


class PolygonProvider(Provider):
    name = "polygon"

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 rate_per_min: float | None = None, include_otc: bool = False):
        self.key = api_key or os.environ.get("POLYGON_API_KEY") or os.environ.get("MASSIVE_API_KEY")
        if not self.key:
            raise ProviderError("POLYGON_API_KEY is not set")
        explicit = base_url or os.environ.get("POLYGON_BASE_URL")
        self.base = explicit.rstrip("/") if explicit else self._pick_base()
        rate = rate_per_min if rate_per_min is not None else float(os.environ.get("POLYGON_RATE_PER_MIN", "0"))
        self.http = HttpClient(RateLimiter(rate))
        self.include_otc = include_otc
        self.unavailable: set[str] = set()     # dates the plan does not serve

    # ------------------------------------------------------------------ utils
    def _pick_base(self) -> str:
        """Massive's host first, the original Polygon host second. Any HTTP answer
        (even 401) means the host is reachable; a refused connection means the
        environment's network policy blocks it."""
        for b in BASES:
            try:
                requests.get(f"{b}/v1/marketstatus/now", params={"apiKey": self.key}, timeout=20)
                return b
            except requests.RequestException as e:
                log.warning("%s is not reachable (%s)", b, e.__class__.__name__)
        raise ProviderError("Neither api.massive.com nor api.polygon.io is reachable. Add them under "
                            "Allowed domains in the environment's network settings.")

    def _get(self, path_or_url: str, params: dict | None = None) -> dict:
        url = path_or_url if path_or_url.startswith("http") else self.base + path_or_url
        p = dict(params or {})
        p["apiKey"] = self.key
        return self.http.get_json(url, p)

    def _paged(self, path: str, params: dict) -> list[dict]:
        out, js = [], self._get(path, params)
        out.extend(js.get("results") or [])
        while js.get("next_url"):
            js = self._get(js["next_url"])
            out.extend(js.get("results") or [])
        return out

    # ------------------------------------------------------------------ daily
    def daily_bars(self, dates: list[str]) -> pd.DataFrame:
        frames = []
        for d in dates:
            try:
                js = self._get(f"/v2/aggs/grouped/locale/us/market/stocks/{d}",
                               {"adjusted": "false", "include_otc": str(self.include_otc).lower()})
            except PlanLimitError as e:
                log.warning("%s: not covered by this data plan (%s)", d, str(e)[:120])
                self.unavailable.add(d)
                continue
            res = js.get("results") or []
            if not res:
                log.info("%s: no grouped bars (holiday/weekend?)", d)
                frames.append(pd.DataFrame(columns=DAILY_COLS))
                continue
            df = pd.DataFrame(res).rename(columns={"T": "ticker"})
            df["date"] = d
            for col in ("vw", "n"):
                if col not in df:
                    df[col] = float("nan")
            frames.append(df[DAILY_COLS])
            log.info("%s: %d tickers", d, len(df))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=DAILY_COLS)

    # ----------------------------------------------------------------- minute
    def minute_bars(self, sessions, prev_day):
        by_ticker: dict[str, list[str]] = defaultdict(list)
        for t, d in sessions:
            by_ticker[t].append(d)
        for t, dates in sorted(by_ticker.items()):
            dates = sorted(set(dates))
            # Chunk so one request stays under the 50,000-bar limit
            # (<= ~35 calendar days of 04:00-20:00 minute bars).
            chunk = [dates[0]]
            chunks = []
            for d in dates[1:]:
                if (Date.fromisoformat(d) - Date.fromisoformat(chunk[0])).days > 35:
                    chunks.append(chunk)
                    chunk = [d]
                else:
                    chunk.append(d)
            chunks.append(chunk)
            for ch in chunks:
                start = prev_day.get(ch[0], ch[0])
                try:
                    rows = self._paged(f"/v2/aggs/ticker/{t}/range/1/minute/{start}/{ch[-1]}",
                                       {"adjusted": "false", "sort": "asc", "limit": 50000})
                except PlanLimitError as e:
                    log.warning("%s %s..%s: not covered by this data plan (%s)", t, start, ch[-1], str(e)[:120])
                    continue
                if not rows:
                    log.warning("%s %s..%s: no minute bars", t, start, ch[-1])
                    continue
                df = pd.DataFrame(rows).rename(columns={"t": "ts"})
                df["ticker"] = t
                if "vw" not in df:
                    df["vw"] = (df["h"] + df["l"] + df["c"]) / 3.0
                df["vw"] = df["vw"].fillna((df["h"] + df["l"] + df["c"]) / 3.0)
                if "n" not in df:
                    df["n"] = 0
                yield df[BAR_COLS]

    # ----------------------------------------------------------------- splits
    def splits(self, start: str, end: str) -> pd.DataFrame:
        rows = self._paged("/v3/reference/splits",
                           {"execution_date.gte": start, "execution_date.lte": end, "limit": 1000})
        if not rows:
            return pd.DataFrame(columns=["ticker", "execution_date", "ratio"])
        df = pd.DataFrame(rows)
        # 1-for-10 reverse split: split_from=10, split_to=1 -> new price = 10x old.
        df["ratio"] = df["split_from"].astype(float) / df["split_to"].astype(float)
        return df[["ticker", "execution_date", "ratio"]]

    # ------------------------------------------------------------------ types
    def security_types(self, tickers, dates, bulk: bool = True):
        want = set(tickers)
        found: dict[str, dict] = {}
        for d in (dates if bulk else []):
            # Bulk lists for the types that matter; anything left over (ETFs,
            # funds, delisted names) is looked up one ticker at a time.
            for typ in ("CS", "ADRC", "OS", "WARRANT", "RIGHT", "UNIT", "PFD"):
                if not (want - set(found)):
                    break
                for r in self._paged("/v3/reference/tickers",
                                     {"market": "stocks", "type": typ, "date": d, "limit": 1000}):
                    tk = r.get("ticker")
                    if tk in want and tk not in found:
                        found[tk] = {"ticker": tk, "type": r.get("type", typ),
                                     "exchange": r.get("primary_exchange", ""), "name": r.get("name", "")}
        for tk in sorted(want - set(found)):
            try:
                js = self._get(f"/v3/reference/tickers/{tk}", {"date": dates[-1]})
                r = js.get("results") or {}
                found[tk] = {"ticker": tk, "type": r.get("type", "UNKNOWN"),
                             "exchange": r.get("primary_exchange", ""), "name": r.get("name", "")}
            except ProviderError as e:
                log.warning("type lookup failed for %s: %s", tk, e)
                found[tk] = {"ticker": tk, "type": "UNKNOWN", "exchange": "", "name": ""}
        return pd.DataFrame(list(found.values()), columns=["ticker", "type", "exchange", "name"])
