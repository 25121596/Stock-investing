"""Offline synthetic checks for the ML feature/label panel."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from engine import Config
from ml_features import FEATURES, build_panel


def synthetic(code="sh.600001", n=270, start="2020-01-02"):
    dates = pd.bdate_range(start, periods=n).strftime("%Y-%m-%d")
    close = 10.0 + np.arange(n) * 0.002
    frame = pd.DataFrame({
        "date": dates,
        "code": code,
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "preclose": np.r_[close[0] / 1.001, close[:-1]],
        "volume": np.full(n, 2_000_000.0),
        "amount": np.full(n, 30_000_000.0),
        "tradestatus": np.ones(n, dtype=int),
        "isST": np.zeros(n, dtype=int),
        "adj_close": close * 1.25,
        "pctChg": np.full(n, 0.02),
        "source_index": np.arange(n),
    })
    return frame


class MLFeatureTests(unittest.TestCase):
    def test_future_prices_do_not_change_past_features(self):
        frame = synthetic(n=280, start="2023-01-02")
        day = frame.date.iloc[260]
        baseline = build_panel({"sh.600001": frame}, frame.date, config=Config())
        changed = frame.copy()
        future = changed.index > 260
        for col in ("open", "high", "low", "close", "preclose", "adj_close"):
            changed.loc[future, col] *= 1.7
        revised = build_panel({"sh.600001": changed}, frame.date, config=Config())
        a = baseline.loc[baseline.date.eq(day), FEATURES].iloc[0]
        b = revised.loc[revised.date.eq(day), FEATURES].iloc[0]
        pd.testing.assert_series_equal(a, b, check_names=False)

    def test_global_adjustment_factor_does_not_change_features(self):
        frame = synthetic(n=280)
        day = frame.date.iloc[260]
        baseline = build_panel({"sh.600001": frame}, frame.date, config=Config())
        rescaled = frame.copy()
        rescaled["adj_close"] *= 10
        revised = build_panel({"sh.600001": rescaled}, frame.date, config=Config())
        a = baseline.loc[baseline.date.eq(day), FEATURES].iloc[0].to_numpy(dtype=float)
        b = revised.loc[revised.date.eq(day), FEATURES].iloc[0].to_numpy(dtype=float)
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_label_uses_next_open_and_real_exit_date(self):
        frame = synthetic(n=280)
        signal_i = 260
        frame.loc[signal_i, "close"] = 10.0
        frame.loc[signal_i, "high"] = 10.1
        frame.loc[signal_i, "low"] = 9.9
        frame.loc[signal_i + 1, "preclose"] = 10.0
        frame.loc[signal_i + 1, "open"] = 10.2  # distinct from the signal close
        frame.loc[signal_i + 1, "close"] = 10.2
        frame.loc[signal_i + 2, "preclose"] = 10.2
        frame.loc[signal_i + 3, "preclose"] = frame.loc[signal_i + 2, "close"]
        frame.loc[signal_i + 4, "preclose"] = frame.loc[signal_i + 3, "close"]
        frame.loc[signal_i + 4, "open"] = 10.3
        config = Config(hold_days=3)
        panel = build_panel({"sh.600001": frame}, frame.date, config=config)
        row = panel.loc[panel.date.eq(frame.date.iloc[signal_i])].iloc[0]
        quantity = int((5000 - config.min_commission) /
                       (10.30 * (1 + config.commission + config.transfer_fee + config.slippage))) // 100 * 100
        # Actual entry uses the next session open; the fixed horizon exits at
        # entry index + hold_days, i.e. the fourth date from the signal.
        entry = quantity * 10.2
        exit_value = quantity * 10.3
        expected = (exit_value - sum(config.fees(exit_value, sell=True).values())
                    - entry - sum(config.fees(entry).values())) / (entry + sum(config.fees(entry).values()))
        self.assertAlmostEqual(row.net_return, expected)
        self.assertEqual(row.label_end, frame.date.iloc[signal_i + 4])
        self.assertNotEqual(row.date[:4], row.label_end[:4])

    def test_corporate_action_masks_label_but_keeps_prediction_row(self):
        frame = synthetic(n=280)
        signal_day = frame.date.iloc[260]
        action_day = frame.date.iloc[262]
        dividends = [{"code": "sh.600001", "dividRegistDate": action_day,
                      "dividOperateDate": "", "dividPayDate": "", "dividStockMarketDate": ""}]
        panel = build_panel({"sh.600001": frame}, frame.date, dividends, Config())
        row = panel.loc[panel.date.eq(signal_day)]
        self.assertEqual(len(row), 1)
        self.assertTrue(bool(row.iloc[0].eligible))
        self.assertTrue(pd.isna(row.iloc[0].net_return))
        self.assertTrue(pd.isna(row.iloc[0].y_win))
        self.assertIsNone(row.iloc[0].label_end)


if __name__ == "__main__":
    unittest.main()
