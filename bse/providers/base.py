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
import re
import time
from dataclasses import dataclass

import pandas as pd
import requests

log = logging.getLogger("bse.provider")

# Query-string credentials (Polygon/Massive pass apiKey in the URL). Exception
# text from requests repeats the full URL, so scrub it before it reaches a log.
_SECRET_RE = re.compile(r"((?:api_?key|apikey|token|secret)=)[^&\s'\")]+", re.I)


def redact(text) -> str:
    return _SECRET_RE.sub(r"\1***", str(text))


class _RedactFilter(logging.Filter):
    def filter(self, record):
        if record.args:
            record.args = tuple(redact(a) if isinstance(a, (str, Exception)) else a for a in record.args)
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        return True


log.addFilter(_RedactFilter())

BAR_COLS = ["ticker", "ts", "o", "h", "l", "c", "v", "vw", "n"]
DAILY_COLS = ["date", "ticker", "o", "h", "l", "c", "v", "vw", "n"]


class ProviderError(RuntimeError):
    pass


class AuthError(ProviderError):
    """401: the key itself was rejected. Fatal."""


class PlanLimitError(AuthError):
    """403: the key works but the plan does not cover this request (e.g. data
    older than the plan's history). Skip the item and carry on."""


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

    def slow_down(self, retry_after: float | None = None) -> float:
        """Called on HTTP 429. Returns how long to pause before retrying.

        An unlimited start drops to just under the free plan's 5/min. After that
        each further 429 trims the rate by 20% (never below 3/min). The pause
        lets the provider's one-minute window empty, so a single burst does not
        cascade into repeated cuts."""
        if not self.per_min:
            self.per_min = 4.8
        else:
            self.per_min = max(3.0, self.per_min * 0.8)
        log.warning("rate limited; pacing at %.1f requests/min", self.per_min)
        return max(retry_after or 0.0, 61.0)


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
                log.warning("GET %s failed (%s); retry %d", url.split("?")[0], redact(e), attempt)
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 401:
                raise AuthError(f"401 from {url.split('?')[0]}: {r.text[:300]}")
            if r.status_code == 403:
                raise PlanLimitError(f"403 from {url.split('?')[0]}: {r.text[:300]}")
            if r.status_code == 429:
                ra = r.headers.get("Retry-After", "")
                try:
                    ra_s = float(ra)
                except ValueError:
                    ra_s = None
                time.sleep(self.limiter.slow_down(ra_s))
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
