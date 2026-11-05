"""Alpaca Market Data adapter (free plan works: historical SIP bars, 200 req/min).

Env:
  APCA_API_KEY_ID, APCA_API_SECRET_KEY   required
  APCA_DATA_URL       default https://data.alpaca.markets
  APCA_TRADING_URL    default https://paper-api.alpaca.markets (asset list only)
  APCA_FEED           default sip
  APCA_RATE_PER_MIN   default 180

Alpaca has no whole-market "grouped" endpoint, so the screen pulls daily bars
for every listed asset (active and inactive) in batches of 200 symbols. Asset
records carry no security type, so warrants/units/rights/preferreds/ETFs are
dropped by name and symbol pattern.
"""
from __future__ import annotations

import logging
import os
import re
from collections import defaultdict

import pandas as pd

from ..config import TZ
from .base import BAR_COLS, DAILY_COLS, HttpClient, Provider, ProviderError, RateLimiter

log = logging.getLogger("bse.alpaca")

_EXCLUDE_NAME = re.compile(
    r"\bwarrants?\b|\brights?\b|\bunits?\b|preferred|\bpfd\b|\betf\b|\betn\b|\bfund\b|"
    r"\bindex\b|exchange[- ]traded|\bnotes? due\b|debentures?", re.I)
_EXCLUDE_SYMBOL = re.compile(r"(\.WS|\.U|\.RT|\.PR[A-Z]?|\.W)$|^[A-Z]{4}[WUR]$")
_LISTED = {"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS"}


class AlpacaProvider(Provider):
    name = "alpaca"

    def __init__(self, include_otc: bool = False):
        kid = os.environ.get("APCA_API_KEY_ID")
        sec = os.environ.get("APCA_API_SECRET_KEY")
        if not (kid and sec):
            raise ProviderError("APCA_API_KEY_ID / APCA_API_SECRET_KEY are not set")
        headers = {"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec}
        self.data = os.environ.get("APCA_DATA_URL", "https://data.alpaca.markets").rstrip("/")
        self.trading = os.environ.get("APCA_TRADING_URL", "https://paper-api.alpaca.markets").rstrip("/")
        self.feed = os.environ.get("APCA_FEED", "sip")
        self.http = HttpClient(RateLimiter(float(os.environ.get("APCA_RATE_PER_MIN", "180"))), headers)
        self.include_otc = include_otc
        self._assets: pd.DataFrame | None = None

    # ----------------------------------------------------------------- assets
    def assets(self) -> pd.DataFrame:
        if self._assets is None:
            rows = []
            for status in ("active", "inactive"):
                js = self.http.get_json(f"{self.trading}/v2/assets",
                                        {"status": status, "asset_class": "us_equity"})
                rows.extend(js)
            df = pd.DataFrame(rows)
            keep = df["exchange"].isin(_LISTED | ({"OTC"} if self.include_otc else set()))
            self._assets = df[keep].drop_duplicates("symbol").reset_index(drop=True)
        return self._assets

    def borrow_snapshot(self) -> pd.DataFrame:
        """Current shortable / easy-to-borrow flags (only today's status exists)."""
        a = self.assets()
        a = a[a["status"] == "active"]
        return a[["symbol", "shortable", "easy_to_borrow"]].rename(columns={"symbol": "ticker"})

    def _bars(self, symbols: list[str], timeframe: str, start: str, end: str):
        params = {"symbols": ",".join(symbols), "timeframe": timeframe, "start": start, "end": end,
                  "adjustment": "raw", "feed": self.feed, "limit": 10000, "sort": "asc"}
        while True:
            js = self.http.get_json(f"{self.data}/v2/stocks/bars", params)
            for sym, bars in (js.get("bars") or {}).items():
                if bars:
                    yield sym, bars
            tok = js.get("next_page_token")
            if not tok:
                break
            params["page_token"] = tok

    # ------------------------------------------------------------------ daily
    def daily_bars(self, dates):
        dates = sorted(dates)
        syms = self.assets()["symbol"].tolist()
        frames = []
        for i in range(0, len(syms), 200):
            for sym, bars in self._bars(syms[i:i + 200], "1Day", dates[0], dates[-1]):
                df = pd.DataFrame(bars)
                df["date"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert(TZ).dt.strftime("%Y-%m-%d")
                df["ticker"] = sym
                frames.append(df)
        if not frames:
            return pd.DataFrame(columns=DAILY_COLS)
        out = pd.concat(frames, ignore_index=True)
        for col in ("vw", "n"):
            if col not in out:
                out[col] = float("nan")
        return out[out["date"].isin(set(dates))][DAILY_COLS].reset_index(drop=True)

    # ----------------------------------------------------------------- minute
    def minute_bars(self, sessions, prev_day):
        by_date: dict[str, list[str]] = defaultdict(list)
        for t, d in sessions:
            by_date[d].append(t)
        for d, syms in sorted(by_date.items()):
            start = pd.Timestamp(f"{prev_day.get(d, d)} 04:00", tz=TZ).isoformat()
            end = pd.Timestamp(f"{d} 20:00", tz=TZ).isoformat()
            syms = sorted(set(syms))
            acc: dict[str, list] = defaultdict(list)
            for i in range(0, len(syms), 50):
                for sym, bars in self._bars(syms[i:i + 50], "1Min", start, end):
                    acc[sym].extend(bars)
            for sym, bars in acc.items():
                df = pd.DataFrame(bars)
                # resolution-independent (pandas may parse to s, ms, us or ns)
                df["ts"] = ((pd.to_datetime(df["t"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC"))
                            // pd.Timedelta(milliseconds=1)).astype("int64")
                df["ticker"] = sym
                if "vw" not in df:
                    df["vw"] = (df["h"] + df["l"] + df["c"]) / 3.0
                if "n" not in df:
                    df["n"] = 0
                yield df[BAR_COLS]

    # ----------------------------------------------------------------- splits
    def splits(self, start, end):
        rows, params = [], {"types": "reverse_split,forward_split", "start": start, "end": end, "limit": 1000}
        try:
            while True:
                js = self.http.get_json(f"{self.data}/v1/corporate-actions", params)
                ca = js.get("corporate_actions") or {}
                for kind in ("reverse_splits", "forward_splits"):
                    for r in ca.get(kind) or []:
                        old, new = float(r.get("old_rate", 1)), float(r.get("new_rate", 1))
                        rows.append({"ticker": r.get("symbol"), "execution_date": r.get("ex_date"),
                                     "ratio": old / new if new else 1.0})
                tok = js.get("next_page_token")
                if not tok:
                    break
                params["page_token"] = tok
        except ProviderError as e:
            log.warning("corporate actions unavailable (%s); prior closes on split days stay unadjusted", e)
        return pd.DataFrame(rows, columns=["ticker", "execution_date", "ratio"])

    # ------------------------------------------------------------------ types
    def security_types(self, tickers, dates, bulk: bool = True):
        a = self.assets().set_index("symbol")
        out = []
        for t in tickers:
            name = str(a["name"].get(t, "")) if t in a.index else ""
            exch = str(a["exchange"].get(t, "")) if t in a.index else ""
            if not name:
                typ = "UNKNOWN"
            elif _EXCLUDE_NAME.search(name) or _EXCLUDE_SYMBOL.search(t):
                typ = "OTHER"
            else:
                typ = "CS"
            out.append({"ticker": t, "type": typ, "exchange": exch, "name": name})
        return pd.DataFrame(out, columns=["ticker", "type", "exchange", "name"])
