"""SYNTHETIC test data - for exercising the code only.

These bars come from a random generator. They say nothing about real markets
and must never be published or quoted as results. Every artifact produced from
them is stamped data_source = "SYNTHETIC".
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .calendar import prev_trading_day
from .config import TZ
from .providers.base import BAR_COLS, DAILY_COLS, Provider


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _round(p):
    return np.where(p < 1.0, np.round(p, 4), np.round(p, 2))


def day_bars(ticker: str, d: str, start_px: float, runner: bool) -> pd.DataFrame:
    rng = np.random.default_rng(_seed(ticker, d))
    minutes = np.arange(4 * 60, 20 * 60)
    n = len(minutes)
    rth = (minutes >= 570) & (minutes < 960)
    sigma = np.where(rth, 0.006, 0.003)
    drift = np.zeros(n)
    volm = np.where(rth, 1.0, 0.15)
    if runner:
        start = int(rng.integers(150, 420))           # 06:30 .. 11:00
        up_len = int(rng.integers(8, 60))
        run = rng.uniform(0.15, 1.5)
        drift[start:start + up_len] += np.log1p(run) / up_len
        volm[start:start + up_len] *= rng.uniform(5, 15)
        fade_len = int(rng.integers(30, 240))
        fade = rng.uniform(0.2, 0.8) * np.log1p(run)
        drift[start + up_len:start + up_len + fade_len] -= fade / fade_len
        volm[start + up_len:start + up_len + fade_len] *= rng.uniform(2, 5)
        if rng.random() < 0.3:                        # a second leg up
            s2 = start + up_len + fade_len // 2
            drift[s2:s2 + 10] += np.log1p(run * 0.4) / 10
            volm[s2:s2 + 10] *= 6
    steps = 10
    inc = rng.standard_normal((n, steps)) * (sigma[:, None] / np.sqrt(steps)) + drift[:, None] / steps
    logp = np.log(start_px) + rng.normal(0, 0.03) + np.cumsum(inc.ravel()).reshape(n, steps)
    prev_last = np.concatenate([[logp[0, 0]], logp[:-1, -1]])
    o = np.exp(prev_last)
    c = np.exp(logp[:, -1])
    h = np.maximum(np.exp(logp.max(axis=1)), o)
    l = np.minimum(np.exp(logp.min(axis=1)), o)
    v = np.round(rng.lognormal(np.log(25000 if runner else 3000), 1.0, n) * volm)
    keep = rng.random(n) < np.where(rth, 0.995, 0.55)
    if runner and rng.random() < 0.25:                # a 5-minute halt somewhere in RTH
        hs = int(rng.integers(340, 900))
        keep[hs:hs + 5] = False
    o, h, l, c = (_round(x) for x in (o, h, l, c))
    h = np.maximum.reduce([h, o, c])
    l = np.maximum(np.minimum.reduce([l, o, c]), 0.0001)
    vw = (h + l + c) / 3.0
    ts = (pd.Timestamp(d, tz=TZ) + pd.to_timedelta(minutes, unit="m")).tz_convert("UTC")
    ms = (ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)
    df = pd.DataFrame({"ticker": ticker, "ts": np.asarray(ms, dtype="int64"), "o": o, "h": h, "l": l,
                       "c": c, "v": np.maximum(v, 100.0), "vw": vw, "n": 10})
    return df[keep].reset_index(drop=True)[BAR_COLS]


class SyntheticProvider(Provider):
    """A fake market: a few hundred tickers, ~8% of them sub-$1 runners per day."""
    name = "SYNTHETIC"

    def __init__(self, n_tickers: int = 400, runner_rate: float = 0.08):
        self.tickers = [f"SYN{i:03d}" for i in range(n_tickers)]
        self.rate = runner_rate

    def level(self, t, d):
        return float(np.round(np.random.default_rng(_seed("pc", t, d)).uniform(0.12, 1.3), 4))

    def runner(self, t, d):
        return np.random.default_rng(_seed("run", t, d)).random() < self.rate

    def bars(self, t, d):
        return day_bars(t, d, self.level(t, prev_trading_day(d)), self.runner(t, d))

    def daily_bars(self, dates):
        rows = []
        for d in dates:
            if pd.Timestamp(d).weekday() >= 5:
                continue
            for t in self.tickers:
                if self.runner(t, d):
                    b = self.bars(t, d)
                    rows.append([d, t, b.o.iloc[0], b.h.max(), b.l.min(), b.c.iloc[-1], float(b.v.sum()),
                                 float((b.vw * b.v).sum() / b.v.sum()), len(b)])
                else:
                    p = self.level(t, d)
                    vol = float(np.random.default_rng(_seed("v", t, d)).integers(100_000, 3_000_000))
                    rows.append([d, t, p, p * 1.03, p * 0.97, p, vol, p, 100])
        return pd.DataFrame(rows, columns=DAILY_COLS)

    def minute_bars(self, sessions, prev_day):
        for t, d in sessions:
            frames = []
            pdte = prev_day.get(d)
            if pdte:
                frames.append(self.bars(t, pdte))
            frames.append(self.bars(t, d))
            yield pd.concat(frames, ignore_index=True)

    def security_types(self, tickers, dates):
        return pd.DataFrame([{"ticker": t, "type": "WARRANT" if t.endswith("7") else "CS",
                              "exchange": "XNAS", "name": f"Synthetic {t}"} for t in tickers])
