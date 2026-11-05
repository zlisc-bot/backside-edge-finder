"""Hand-built paths with known answers for the fill and exit rules."""
import copy

import numpy as np
import pandas as pd
import pytest

from bse.backtest import Entries, Rule, net_returns, no_overlap, simulate, build_entries
from bse.config import TZ, Config
from bse.indicators import prepare_session

DAY = "2026-10-14"


def make_session(closes, opens=None, highs=None, lows=None, start="09:30", pc=0.6, ssr_carry=False, gaps=None):
    n = len(closes)
    c = np.array(closes, float)
    o = np.array(opens if opens is not None else np.concatenate([[c[0]], c[:-1]]), float)
    h = np.array(highs if highs is not None else np.maximum(o, c) + 0.0005, float)
    l = np.array(lows if lows is not None else np.minimum(o, c) - 0.0005, float)
    t0 = pd.Timestamp(f"{DAY} {start}", tz=TZ)
    mins = np.arange(n) if gaps is None else np.array(gaps)
    ts = ((t0 + pd.to_timedelta(mins, unit="m")).tz_convert("UTC") - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)
    bars = pd.DataFrame({"ts": np.asarray(ts, "int64"), "o": o, "h": h, "l": l, "c": c, "v": 10000.0, "vw": (h + l + c) / 3})
    meta = {"session_id": f"TEST_{DAY}", "ticker": "TEST", "date": DAY, "pc": pc, "ssr_carry": ssr_carry}
    s = prepare_session(bars, meta, zero_cost_cfg())
    s.arr["_lh_price"] = np.full(len(s), np.nan)
    return s


def zero_cost_cfg():
    cfg = Config()          # built-in defaults, not config.local.json
    c = cfg.costs
    c.spread_floor_ticks, c.spread_floor_bps, c.spread_mult = 0, 0.0, 0.0
    c.slippage_bps, c.stop_slip_range_frac = 0.0, 0.0
    c.commission_per_share, c.commission_min, c.locate_pct = 0.0, 0.0, 0.0
    c.sell_fee_per_share = 0.0
    c.ssr_policy = "ignore"
    return cfg


def entry_at(idx, price, start=None, t_in=None):
    one = lambda v, dt=float: np.array([v], dtype=dt)
    return Entries(0, 0, one(idx, int), one(idx, int), one(idx, int), one(price), one(start if start is not None else idx + 1, int),
                   one(t_in if t_in is not None else idx + 0.99), one(False, bool), one(False, bool), one(False, bool))


def run(s, ent, stops, targets, holds, cfg=None):
    cfg = cfg or zero_cost_cfg()
    j_flat = len(s) - 1
    sim = simulate(s, ent, stops, targets, holds, cfg, j_flat)
    r = net_returns(s, ent, sim, cfg.costs)
    return sim, r


PCT2 = Rule("pct0.02", "", "pct", 0.02)
TP1 = Rule("tp0.01", "", "pct", 0.01)
NONE = Rule("none", "", "none", None)


def test_target_fills_at_limit_after_trade_through():
    closes = [0.50] * 31 + [0.499, 0.498, 0.497, 0.496, 0.495, 0.494, 0.493]
    s = make_session(closes)
    sim, r = run(s, entry_at(30, 0.50), [PCT2], [TP1], [999])
    # target 0.4950 needs a print <= 0.4949: bar 35 has low 0.4945
    assert sim.j_exit[0, 0, 0, 0] == 35
    assert sim.reason[0, 0, 0, 0] == 1
    assert sim.p_exit[0, 0, 0, 0] == pytest.approx(0.495)
    assert r[0, 0, 0, 0] == pytest.approx(0.01)


def test_stop_fills_at_stop_price():
    closes = [0.50] * 31 + [0.502, 0.504, 0.506, 0.508, 0.510, 0.512]
    s = make_session(closes)
    sim, r = run(s, entry_at(30, 0.50), [PCT2], [NONE], [999])
    assert sim.j_exit[0, 0, 0, 0] == 35          # high 0.5105 >= 0.51
    assert sim.reason[0, 0, 0, 0] == 0
    assert r[0, 0, 0, 0] == pytest.approx(-0.02)


def test_gap_through_stop_fills_at_open():
    closes = [0.50] * 31 + [0.501, 0.53, 0.53]
    opens = [0.50] * 31 + [0.50, 0.53, 0.53]       # e.g. reopen after a halt
    s = make_session(closes, opens=opens)
    sim, r = run(s, entry_at(30, 0.50), [PCT2], [NONE], [999])
    assert sim.j_exit[0, 0, 0, 0] == 32
    assert sim.p_exit[0, 0, 0, 0] == pytest.approx(0.53)
    assert r[0, 0, 0, 0] == pytest.approx(-0.06)


def test_bar_touching_stop_and_target_counts_as_stopped():
    closes = [0.50] * 31 + [0.50, 0.50]
    highs = [0.5005] * 31 + [0.52, 0.5005]
    lows = [0.4995] * 31 + [0.48, 0.4995]
    s = make_session(closes, highs=highs, lows=lows)
    sim, r = run(s, entry_at(30, 0.50), [PCT2], [TP1], [999])
    assert sim.j_exit[0, 0, 0, 0] == 31
    assert sim.reason[0, 0, 0, 0] == 0
    assert r[0, 0, 0, 0] == pytest.approx(-0.02)


def test_open_beyond_target_takes_the_target_first():
    closes = [0.50] * 31 + [0.50, 0.50]
    opens = [0.50] * 31 + [0.49, 0.50]
    highs = [0.5005] * 31 + [0.52, 0.5005]
    lows = [0.4995] * 31 + [0.485, 0.4995]
    s = make_session(closes, opens=opens, highs=highs, lows=lows)
    sim, r = run(s, entry_at(30, 0.50), [PCT2], [TP1], [999])
    assert sim.reason[0, 0, 0, 0] == 1
    assert sim.p_exit[0, 0, 0, 0] == pytest.approx(0.49)      # filled at the open, below the limit
    assert r[0, 0, 0, 0] == pytest.approx(0.02)


def test_time_exit_at_close_of_last_bar_inside_hold():
    closes = [0.50] * 31 + [0.501, 0.502, 0.503, 0.504, 0.505, 0.506, 0.507]
    s = make_session(closes)
    sim, r = run(s, entry_at(30, 0.50), [Rule("pct0.5", "", "pct", 0.5)], [NONE], [5])
    assert sim.j_exit[0, 0, 0, 0] == 35          # 09:35 is the last bar within 5 minutes
    assert sim.reason[0, 0, 0, 0] == 3
    assert r[0, 0, 0, 0] == pytest.approx((0.50 - 0.505) / 0.50)


def test_costs_are_charged():
    closes = [0.50] * 31 + [0.499, 0.498, 0.497, 0.496, 0.495, 0.494, 0.493]
    s = make_session(closes)
    cfg = zero_cost_cfg()
    cfg.costs.commission_per_share = 0.001       # = 0.2% per side at $0.50
    cfg.costs.locate_pct = 0.003
    sim, r = run(s, entry_at(30, 0.50), [PCT2], [TP1], [999], cfg)
    # entry fee 0.001/0.50 = 0.2% of entry value; exit fee 0.001 x shares = 0.2% of entry value
    exp = 0.01 - 0.002 - 0.002 - 0.003
    assert r[0, 0, 0, 0] == pytest.approx(exp, abs=1e-9)


def test_no_overlap_skips_entries_inside_an_open_trade():
    closes = [0.50] * 31 + [0.501] * 10
    s = make_session(closes)
    one = lambda v, dt=float: np.array(v, dtype=dt)
    ent = Entries(0, 0, one([30, 32], int), one([30, 32], int), one([30, 32], int), one([0.5, 0.501]),
                  one([31, 33], int), one([30.99, 32.99]), one([False, False], bool), one([False, False], bool),
                  one([False, False], bool))
    sim = simulate(s, ent, [Rule("pct0.5", "", "pct", 0.5)], [NONE], [5], zero_cost_cfg(), len(s) - 1)
    keep = no_overlap(ent, sim)
    assert keep[0, 0, 0, 0] and not keep[1, 0, 0, 0]


def test_ssr_requires_an_uptick_fill():
    cfg = zero_cost_cfg()
    cfg.costs.ssr_policy = "uptick"
    closes = [0.50] * 31 + [0.49, 0.48, 0.47]      # keeps falling: a resting offer never fills
    s = make_session(closes, highs=[0.5005] * 31 + [0.5001, 0.49, 0.48], pc=0.6, ssr_carry=True)
    perbar = {"epoch_id": np.zeros(len(s), int), "fr_ema9": np.zeros(len(s), bool), "fr_ema20": np.zeros(len(s), bool),
              "fr_vwap": np.zeros(len(s), bool), "lh_flag": np.zeros(len(s), bool), "lh_peak_idx": np.full(len(s), -1),
              "lh_price": np.full(len(s), np.nan)}
    ev = [{"sig": "S01", "idx": 30}]
    assert not any(e.tim_i == 0 for e in build_entries(s, perbar, ev, cfg))
    closes2 = [0.50] * 31 + [0.5003, 0.49]          # ticks back up through the offer
    s2 = make_session(closes2, highs=[0.5005] * 31 + [0.5008, 0.5003], pc=0.6, ssr_carry=True)
    ents = [e for e in build_entries(s2, perbar, ev, cfg) if e.tim_i == 0]
    assert len(ents) == 1 and ents[0].passive[0] and ents[0].price[0] == pytest.approx(0.5001)


def test_sell_fee_is_charged_on_the_short_sale_only():
    closes = [0.50] * 31 + [0.499, 0.498, 0.497, 0.496, 0.495, 0.494, 0.493]
    s = make_session(closes)
    cfg = zero_cost_cfg()
    cfg.costs.sell_fee_per_share = 0.000166            # FINRA TAF, $10,000 position
    sim, r = run(s, entry_at(30, 0.50), [PCT2], [TP1], [999], cfg)
    # 20,000 shares x $0.000166 = $3.32 on a $10,000 sale = 0.0332%
    assert r[0, 0, 0, 0] == pytest.approx(0.01 - 0.000332, abs=1e-9)
