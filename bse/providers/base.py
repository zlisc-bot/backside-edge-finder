"""Provider interface and shared HTTP plumbing.

A provider answers four questions:
  * daily bars for every listed US stock on a date (the screen),
  * 1-minute bars for a list of (ticker, date) sessions,
  * splits in a date range (to split-adjust the prior close on split days),
  * security type for a ticker (to drop warrants, units, rights, ETFs).

All prices are returned UNADJUSTED.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import pandas as pd
import requests

log = logging.getLogger("bse.provider")

BAR_COLS = ["ticker", "ts", "o", "h", "l", "c", "v", "vw", "n"]
DAILY_COLS = ["date", "ticker", "o", "h", "l", "c", "v", "vw", "n"]


class ProviderError(RuntimeError):
    pass


class AuthError(ProviderError):
    pass


@dataclass
class RateLimiter:
    per_min: float  # 0 = unlimited

    def __post_init__(self):
        self._last = 0.0

    def wait(self):
        if not self.per_min:
            return
        gap = 60.0 / self.per_min
        now = time.monotonic()
        sleep = self._last + gap - now
        if sleep > 0:
            time.sleep(sleep)
        self._last = time.monotonic()

    def slow_down(self):
        # Called on HTTP 429: halve the rate (or start limiting at 5/min).
        self.per_min = max(1.0, (self.per_min or 10.0) / 2.0)
        log.warning("rate limited; slowing to %.1f requests/min", self.per_min)


class HttpClient:
    def __init__(self, limiter: RateLimiter, headers: dict | None = None, timeout: int = 60):
        self.limiter = limiter
        self.session = requests.Session()
        if headers:
            self.session.headers.update(headers)
        self.timeout = timeout

    def get_json(self, url: str, params: dict | None = None, max_tries: int = 8) -> dict:
        delay = 2.0
        for attempt in range(1, max_tries + 1):
            self.limiter.wait()
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as e:  # network blip
                log.warning("GET %s failed (%s); retry %d", url.split("?")[0], e, attempt)
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code in (401, 403):
                raise AuthError(f"{r.status_code} from {url.split('?')[0]}: {r.text[:300]}")
            if r.status_code == 429:
                self.limiter.slow_down()
                time.sleep(max(delay, 12))
                delay = min(delay * 2, 120)
                continue
            if r.status_code >= 500:
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise ProviderError(f"{r.status_code} from {url.split('?')[0]}: {r.text[:300]}")
        raise ProviderError(f"gave up after {max_tries} tries: {url.split('?')[0]}")


class Provider:
    name = "base"

    def daily_bars(self, dates: list[str]) -> pd.DataFrame:
        """All listed stocks, one row per (date, ticker). Columns DAILY_COLS."""
        raise NotImplementedError

    def minute_bars(self, sessions: list[tuple[str, str]], prev_day: dict[str, str]):
        """Yield DataFrames (BAR_COLS) covering each (ticker, date) session plus
        the prior trading day (used only to warm up the EMAs)."""
        raise NotImplementedError

    def splits(self, start: str, end: str) -> pd.DataFrame:
        """Columns: ticker, execution_date, ratio (new price / old price)."""
        return pd.DataFrame(columns=["ticker", "execution_date", "ratio"])

    def security_types(self, tickers: list[str], dates: list[str]) -> pd.DataFrame:
        """Columns: ticker, type, exchange, name."""
        raise NotImplementedError
