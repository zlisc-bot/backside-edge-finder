"""NYSE trading calendar (holidays and 1 PM early closes), 2025-2027.

The daily screen double-checks it: a weekday with no market-wide bars is
treated as a holiday even if it is missing from this list.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from .config import TZ

HOLIDAYS = {
    # 2025
    "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
    "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18",
    "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}
EARLY_CLOSE = {"2025-07-03", "2025-11-28", "2025-12-24", "2026-11-27", "2026-12-24",
               "2027-11-26"}


def is_trading_day(d: str) -> bool:
    dt = date.fromisoformat(d)
    return dt.weekday() < 5 and d not in HOLIDAYS


def trading_days(start: str, end: str) -> list[str]:
    out, cur, stop = [], date.fromisoformat(start), date.fromisoformat(end)
    while cur <= stop:
        s = cur.isoformat()
        if is_trading_day(s):
            out.append(s)
        cur += timedelta(days=1)
    return out


def prev_trading_day(d: str, n: int = 1) -> str:
    cur = date.fromisoformat(d)
    while n:
        cur -= timedelta(days=1)
        if is_trading_day(cur.isoformat()):
            n -= 1
    return cur.isoformat()


def rth_end_minute(d: str) -> int:
    return 13 * 60 if d in EARLY_CLOSE else 16 * 60


def now_et() -> datetime:
    return datetime.now(ZoneInfo(TZ))


def last_complete_session(now: datetime | None = None) -> str:
    """Most recent date whose extended session (to 20:00 ET) is finished.
    Data vendors finalise the day a little after 20:00, so wait until 20:30."""
    now = now or now_et()
    d = now.date().isoformat()
    if is_trading_day(d) and (now.hour * 60 + now.minute) >= 20 * 60 + 30:
        return d
    return prev_trading_day(d)


def months_back(d: str, months: int) -> str:
    dt = date.fromisoformat(d)
    y, m = dt.year, dt.month - months
    while m <= 0:
        m += 12
        y -= 1
    for day in (dt.day, 30, 29, 28):
        try:
            return date(y, m, day).isoformat()
        except ValueError:
            continue
    raise ValueError(d)
