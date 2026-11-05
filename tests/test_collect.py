"""A day the provider has not published yet must not be saved as a holiday."""
import importlib

import pandas as pd


def test_unpublished_day_is_retried(tmp_path, monkeypatch):
    monkeypatch.setenv("BSE_DATA_DIR", str(tmp_path / "data"))
    import bse.config, bse.store, bse.collect
    importlib.reload(bse.config); importlib.reload(bse.store); importlib.reload(bse.collect)
    from bse.synthetic import SyntheticProvider

    class Late(SyntheticProvider):
        hide = "2026-10-30"
        def daily_bars(self, dates):
            df = super().daily_bars(dates)
            return df[df["date"] != self.hide] if self.hide else df

    cfg = bse.config.load_config()
    cfg.universe.window_months = 1
    cfg.universe.max_history_days = 0
    prov = Late(n_tickers=40)
    st = bse.store.Store()
    bse.collect.update(cfg, end="2026-10-30", provider=prov)
    assert not st.has_daily("2026-10-30")            # not saved as an empty "holiday"
    assert st.has_daily("2026-10-29")
    prov.hide = None                                 # published now
    bse.collect.update(cfg, end="2026-10-30", provider=prov)
    assert st.has_daily("2026-10-30") and len(pd.read_parquet(st.daily_path("2026-10-30")))
