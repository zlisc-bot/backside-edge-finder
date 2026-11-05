"""Every tunable assumption in one place.

Nothing in the engine hard-codes a threshold that is defined here. When the
dashboard quotes a definition, it reads it from these values, so the page and
the code cannot drift apart.

Override any value without editing code by putting a JSON file at
``config.local.json`` in the repository root, e.g.::

    {"costs": {"commission_per_share": 0.0035}, "universe": {"include_otc": true}}
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("BSE_DATA_DIR", ROOT / "data"))
DERIVED_DIR = Path(os.environ.get("BSE_DERIVED_DIR", ROOT / "derived"))
REPORT_DIR = Path(os.environ.get("BSE_REPORT_DIR", ROOT / "report"))
TZ = "America/New_York"


def hm(s: str) -> int:
    """'09:30' -> minutes after midnight (570)."""
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def fmt_hm(m: int) -> str:
    return f"{int(m) // 60:02d}:{int(m) % 60:02d}"


@dataclass
class Universe:
    # A session qualifies when the official daily bar shows a low below this
    # price AND total daily volume (extended hours included) of at least
    # min_daily_volume. The whole session is kept, including any time above $1.
    max_low_price: float = 1.00
    min_daily_volume: int = 5_000_000
    # Exchange-listed only by default. OTC sub-penny names trade hundreds of
    # millions of shares and would swamp the sample; most brokers cannot short
    # them and LULD / Rule 201 mechanics differ.
    include_otc: bool = False
    # Polygon ticker types kept (common stock, ADR common, ordinary shares).
    # Warrants, rights, units, preferreds, ETFs and funds are excluded.
    ticker_types: tuple = ("CS", "ADRC", "OS")
    # Rolling window for collection and analysis: the most recent 24 calendar
    # months, clipped to what the data plan serves (Massive's free plan: 2 years).
    # Raw data is never deleted.
    window_months: int = 24
    max_history_days: int = 725      # 0 = no plan limit (paid plans)
    # Prices are UNADJUSTED. A stock that later reverse-split traded below $1 at
    # the time; split-adjusted prices would hide it (selection bias).
    adjusted: bool = False


@dataclass
class Session:
    ext_start: str = "04:00"
    rth_start: str = "09:30"
    rth_end: str = "16:00"
    ext_end: str = "20:00"
    # VWAP anchor used by every signal: "ext" = from 04:00 (what most small-cap
    # platforms show with extended hours on), "rth" = from 09:30.
    vwap_anchor: str = "ext"
    # Signals are detected from 04:00 to rth_end. Trades are only simulated for
    # entries inside [trade_start, trade_last_entry].
    trade_start: str = "09:30"
    trade_last_entry: str = "15:45"
    flat_by: str = "15:55"          # every position is covered by this bar close
    tick: float = 0.0001            # sub-$1 minimum price increment
    tick_above_1: float = 0.01


@dataclass
class Indicators:
    ema_fast: int = 9
    ema_slow: int = 20
    atr_len: int = 14
    pivot_left: int = 2             # swing high/low = fractal with 2 bars each side
    pivot_right: int = 2            # -> a pivot is only KNOWN 2 bars later
    reclaim_hold_bars: int = 2      # closes above a level this many bars = real reclaim
    vol_lookback: int = 10
    sell_expansion_mult: float = 1.5   # red bar vol >= 1.5x avg prior red-bar vol
    bounce_fade_mult: float = 0.5      # green bar vol <= 0.5x avg green vol into the high
    halt_gap_min: int = 5           # >= 5 min with no prints during RTH = suspected halt
    spread_window: int = 30         # bars used by the Abdi-Ranaldo spread estimator


@dataclass
class Outcomes:
    horizons_min: tuple = (5, 15, 30, 60)    # plus end of day
    down_moves: tuple = (0.05, 0.10, 0.20)
    first_passage_atr: float = 1.0           # failure = +1 ATR before -1 ATR
    first_passage_pct: float = 0.02          # same test in fixed percent
    first_passage_horizon_min: int = 30
    tod_bucket_min: int = 15
    baseline_every_n_bars: int = 10          # control sample: every 10th bar after a high


@dataclass
class Strategy:
    entry_timings: tuple = ("signal_close", "next_open", "failed_reclaim",
                            "lower_low_break", "lower_high_below_ema20")
    confirm_window_min: int = 15             # how long to wait for a confirmation entry
    lower_high_window_min: int = 20
    targets_pct: tuple = (0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10)
    trail_pct: tuple = (0.02, 0.03, 0.05)
    trail_atr: tuple = (1.5,)
    stops_pct: tuple = (0.01, 0.02, 0.03, 0.05)
    stops_atr: tuple = (1.0, 1.5, 2.0, 3.0)
    stop_buffer_pct: float = 0.002           # stop sits 0.2% above the level
    holds_min: tuple = (5, 15, 30, 60, 999)  # 999 = until flat_by
    # Entry-time buckets used to optimise the time-of-day window.
    tod_edges: tuple = ("09:30", "10:00", "10:30", "11:00", "12:00", "13:00", "16:00")
    cutoffs: tuple = ("10:30", "11:00", "11:30", "12:00", "13:00", "16:00")
    start_times: tuple = ("09:30", "09:35", "09:40", "09:45", "10:00", "10:15", "10:30")
    min_trades: int = 60                     # a config needs this many trades to rank
    min_trades_train: int = 30


@dataclass
class Costs:
    """Base-case trading frictions. Edit these to match YOUR broker."""
    notional_usd: float = 10_000.0           # position size used for per-share fees
    spread_floor_ticks: int = 2
    spread_floor_bps: float = 10.0           # never assume a spread under 0.10%
    spread_cap_bps: float = 500.0
    spread_mult: float = 1.0                 # scales the estimated spread
    slippage_bps: float = 5.0                # extra on every market order
    stop_slip_range_frac: float = 0.10       # stop fills lose 10% of the bar range
    commission_per_share: float = 0.002
    commission_min: float = 1.0
    commission_cap_pct: float = 0.005        # many brokers cap at 0.5%-1% of value
    locate_pct: float = 0.003                # 0.3% of position per trade
    locate_once_per_session: bool = False
    # Rule 201: "uptick" = shorts must rest on the offer; filled only if price
    # trades back up through the limit within 2 bars. "skip" drops the trade.
    # "ignore" pretends SSR does not exist (optimistic; for comparison only).
    ssr_policy: str = "uptick"
    ssr_fill_bars: int = 2
    min_short_price: float = 0.0             # set 1.0 if your broker blocks sub-$1 shorts
    skip_after_halt_bars: int = 0
    # FINRA 4210 maintenance on shorts under $5: max($2.50/share, 100% of value).
    margin_per_share_under_5: float = 2.50


COST_SCENARIOS = {
    "low": dict(spread_mult=0.5, slippage_bps=2.0, stop_slip_range_frac=0.05,
                commission_per_share=0.0005, commission_cap_pct=0.002, locate_pct=0.0),
    "base": {},
    "high": dict(spread_mult=1.5, slippage_bps=10.0, stop_slip_range_frac=0.20,
                 commission_per_share=0.004, commission_cap_pct=0.01, locate_pct=0.01),
}


@dataclass
class Validation:
    train_frac: float = 0.6                  # chronological split
    wf_min_train_weeks: int = 8
    wf_test_weeks: int = 1
    bootstrap_reps: int = 1000
    cscv_blocks: int = 10
    seed: int = 7


@dataclass
class Report:
    # Lists of individual signals/trades and the session charts cover the most
    # recent `drill_days` trading days (artifact size limits). Statistics always
    # use the whole window.
    drill_days: int = 126
    # Session charts are written one file per week, so the whole window fits the
    # artifact's file-count limit. chart_days = 0 charts every session in the window;
    # chart_budget_mb caps their total size (newest weeks are kept first).
    chart_days: int = 0
    chart_budget_mb: int = 200


@dataclass
class Config:
    universe: Universe = field(default_factory=Universe)
    session: Session = field(default_factory=Session)
    ind: Indicators = field(default_factory=Indicators)
    out: Outcomes = field(default_factory=Outcomes)
    strat: Strategy = field(default_factory=Strategy)
    costs: Costs = field(default_factory=Costs)
    val: Validation = field(default_factory=Validation)
    report: Report = field(default_factory=Report)

    def to_dict(self) -> dict:
        return asdict(self)

    def costs_for(self, scenario: str) -> Costs:
        c = copy.deepcopy(self.costs)
        for k, v in COST_SCENARIOS[scenario].items():
            setattr(c, k, v)
        return c


def _merge(dc, overrides: dict):
    for k, v in overrides.items():
        cur = getattr(dc, k)
        if hasattr(cur, "__dataclass_fields__") and isinstance(v, dict):
            _merge(cur, v)
        else:
            setattr(dc, k, tuple(v) if isinstance(cur, tuple) else v)


def load_config() -> Config:
    cfg = Config()
    local = ROOT / "config.local.json"
    if local.exists():
        _merge(cfg, json.loads(local.read_text()))
    return cfg
