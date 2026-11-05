import numpy as np
import pytest

from bse.outcomes import label
from tests.test_backtest import make_session, zero_cost_cfg


def _ev(s, idx):
    A = s.arr
    return {"idx": idx, "move_low": float(A["move_low"][idx]), "lod": float(A["lod"][idx]), "hod": float(A["hod"][idx])}


def test_first_passage_worked_failed_and_tie():
    cfg = zero_cost_cfg()
    base = [0.50] * 40
    down = make_session(base + [0.49, 0.47, 0.45, 0.44] + [0.44] * 30)
    up = make_session(base + [0.51, 0.53, 0.55, 0.56] + [0.56] * 30)
    tie = make_session(base + [0.50] + [0.50] * 30, highs=[0.5005] * 40 + [0.60] + [0.5005] * 30,
                       lows=[0.4995] * 40 + [0.40] + [0.4995] * 30)
    for s, want in ((down, 0.0), (up, 1.0), (tie, 1.0)):
        e = _ev(s, 39)
        label(s, [e], cfg)
        assert e["fp_atr"] == want


def test_new_low_of_move_and_reclaim():
    cfg = zero_cost_cfg()
    closes = [0.50 + 0.002 * i for i in range(40)] + [0.56, 0.55, 0.54, 0.53, 0.52, 0.50, 0.49] + [0.58] * 30
    s = make_session(closes)
    e = _ev(s, 42)
    label(s, [e], cfg)
    assert e["nml_15"] == 1.0                 # later bars break the low since the high
    assert e["rec_vwap_eod"] in (0.0, 1.0, None) or np.isnan(e["rec_vwap_eod"])
    assert e["nhod_eod"] == 1.0               # 0.58 closes take out the high
    assert e["t_nml"] == pytest.approx(1.0)


def test_chart_week_is_monday():
    from bse.report import _week
    assert _week("2026-10-16") == "2026-10-12"   # Friday -> Monday
    assert _week("2026-10-12") == "2026-10-12"   # Monday stays
    assert _week("2026-11-01") == "2026-10-26"   # Sunday -> previous Monday


def test_event_slices_tolerate_missing_gain():
    import numpy as np
    import pandas as pd
    from bse.report import _event_slices
    n = 4
    ev = pd.DataFrame({"gain_pc": [0.5, np.nan, 2.0, 1.0], "after_true_hod": [True, False, True, False],
                       "mins_since_hod": [5, 30, 90, 0], "e9_above_e20": [True, False, True, False],
                       "below_vwap": [False, True, False, True], "price": [0.5, 0.8, 0.3, 0.9],
                       "known_sub1": [True] * n, "cumvol": [2e6, 6e6, 1e6, 9e6]})
    e, labels = _event_slices(ev)
    assert "n/a" in labels["c_gain"]
    assert e["c_gain"].dtype.kind == "i"
