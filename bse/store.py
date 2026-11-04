"""On-disk database (Parquet, partitioned by trading date).

data/
  daily/YYYY-MM-DD.parquet     whole-market daily bars (the screen input)
  minute/YYYY-MM-DD.parquet    1-min bars of the sessions that qualified on that date
  warm/YYYY-MM-DD.parquet      1-min bars on that date for tickers that qualified the NEXT
                               trading day (EMA warm-up only). Splitting them keeps every
                               file write-once, so the git history only grows by new days.
  meta/sessions.parquet        one row per qualifying (ticker, date) session
  meta/splits.parquet          split ratios
  meta/types.parquet           security type per ticker
  meta/borrow/YYYY-MM-DD.parquet  optional daily shortable snapshot
  meta/state.json              collector bookkeeping

Raw data is never deleted; the analysis window is applied when loading.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import DATA_DIR


class Store:
    def __init__(self, root: Path = DATA_DIR):
        self.root = Path(root)
        for sub in ("daily", "minute", "warm", "meta", "meta/borrow"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- daily bars
    def daily_path(self, d: str) -> Path:
        return self.root / "daily" / f"{d}.parquet"

    def has_daily(self, d: str) -> bool:
        return self.daily_path(d).exists()

    def write_daily(self, d: str, df: pd.DataFrame):
        df.to_parquet(self.daily_path(d), index=False, compression="zstd")

    def read_daily(self, dates: list[str]) -> pd.DataFrame:
        frames = [pd.read_parquet(self.daily_path(d)) for d in dates if self.has_daily(d)]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def daily_dates(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "daily").glob("*.parquet"))

    # ------------------------------------------------------------ minute bars
    def minute_path(self, d: str, kind: str = "minute") -> Path:
        return self.root / kind / f"{d}.parquet"

    def read_minute(self, d: str, kind: str = "minute") -> pd.DataFrame:
        p = self.minute_path(d, kind)
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()

    def read_bars_any(self, d: str) -> pd.DataFrame:
        """All stored bars for date d (session bars and warm-up bars)."""
        parts = [x for x in (self.read_minute(d, "minute"), self.read_minute(d, "warm")) if len(x)]
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True).drop_duplicates(["ticker", "ts"])

    def minute_tickers(self, d: str, kind: str = "minute") -> set[str]:
        p = self.minute_path(d, kind)
        if not p.exists():
            return set()
        return set(pd.read_parquet(p, columns=["ticker"])["ticker"].unique())

    def append_minute(self, d: str, df: pd.DataFrame, kind: str = "minute"):
        """Merge bars for date d, de-duplicating on (ticker, ts)."""
        old = self.read_minute(d, kind)
        both = pd.concat([old, df], ignore_index=True) if len(old) else df
        both = both.drop_duplicates(["ticker", "ts"], keep="last").sort_values(["ticker", "ts"])
        both.to_parquet(self.minute_path(d, kind), index=False, compression="zstd")

    # ------------------------------------------------------------------- meta
    def _meta(self, name: str) -> Path:
        return self.root / "meta" / name

    def read_meta(self, name: str) -> pd.DataFrame:
        p = self._meta(f"{name}.parquet")
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()

    def write_meta(self, name: str, df: pd.DataFrame):
        df.to_parquet(self._meta(f"{name}.parquet"), index=False, compression="zstd")

    def state(self) -> dict:
        p = self._meta("state.json")
        return json.loads(p.read_text()) if p.exists() else {}

    def save_state(self, st: dict):
        self._meta("state.json").write_text(json.dumps(st, indent=1, sort_keys=True))

    def write_borrow(self, d: str, df: pd.DataFrame):
        df.to_parquet(self.root / "meta" / "borrow" / f"{d}.parquet", index=False, compression="zstd")
