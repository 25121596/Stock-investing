"""Focused synthetic checks for exploratory, past-only features."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from exploration_features import breadth_days, extend_features, friday_mask


def synthetic(n=150, code="sh.600001"):
    dates = pd.bdate_range("2023-01-02", periods=n).strftime("%Y-%m-%d")
    close = 10.0 * np.exp(np.arange(n) * 0.001)
    return pd.DataFrame({
        "date": dates,
        "code": code,
        "close": close,
        "adj_close": close * 1.7,
        "volume": np.full(n, 1000.0),
        "tradestatus": np.ones(n, dtype=int),
    })


class ExplorationFeatureTests(unittest.TestCase):
    def test_future_price_changes_do_not_change_past_60_features(self):
        frame = synthetic()
        target_i = 100
        panel = pd.DataFrame({"date": [frame.date.iloc[target_i]], "code": ["sh.600001"],
                              "net_return": [np.nan], "label_end": [None]})
        original = extend_features(panel, {"sh.600001": frame})
        changed = frame.copy()
        changed.loc[target_i + 1:, ["close", "adj_close"]] *= 3
        revised = extend_features(panel, {"sh.600001": changed})
        np.testing.assert_allclose(original[["adj_ret60", "ret_vol60"]],
                                   revised[["adj_ret60", "ret_vol60"]], equal_nan=True)

    def test_uniform_adjustment_scale_preserves_both_features(self):
        frame = synthetic()
        target_i = 120
        panel = pd.DataFrame({"date": [frame.date.iloc[target_i]], "code": ["sh.600001"]})
        original = extend_features(panel, {"sh.600001": frame})
        scaled = frame.copy()
        scaled["adj_close"] *= 11.0
        revised = extend_features(panel, {"sh.600001": scaled})
        np.testing.assert_allclose(original[["adj_ret60", "ret_vol60"]],
                                   revised[["adj_ret60", "ret_vol60"]], rtol=1e-12, atol=1e-12)

    def test_panel_order_values_and_missing_future_labels_are_preserved(self):
        frame = synthetic()
        panel = pd.DataFrame({"date": [frame.date.iloc[120], frame.date.iloc[90]],
                              "code": ["sh.600001", "sh.600001"],
                              "tag": ["later", "earlier"], "net_return": [np.nan, np.nan],
                              "label_end": [None, None]})
        extended = extend_features(panel, {"sh.600001": frame})
        self.assertEqual(extended["tag"].tolist(), ["later", "earlier"])
        pd.testing.assert_series_equal(extended["net_return"], panel["net_return"])
        pd.testing.assert_series_equal(extended["label_end"], panel["label_end"])
        self.assertTrue(extended["adj_ret60"].notna().all())

    def test_breadth_threshold_and_missing_values(self):
        panel = pd.DataFrame({
            "date": ["2024-01-02"] * 100 + ["2024-01-03"] * 99 + ["2024-01-04"] * 100,
            "adj_ret5": ([0.01] * 50 + [-0.01] * 50
                         + [0.02] * 99
                         + [0.01] * 49 + [-0.01] * 50 + [np.nan]),
            "net_return": [np.nan] * 299,
        })
        result = breadth_days(panel).set_index("date")
        self.assertEqual(result.loc["2024-01-02", "pool_size"], 100)
        self.assertEqual(result.loc["2024-01-02", "positive5_count"], 50)
        self.assertEqual(result.loc["2024-01-02", "breadth5"], 0.5)
        self.assertTrue(result.loc["2024-01-02", "allow_entry"])
        self.assertFalse(result.loc["2024-01-03", "allow_entry"])
        self.assertEqual(result.loc["2024-01-04", "pool_size"], 100)
        self.assertEqual(result.loc["2024-01-04", "positive5_count"], 49)
        self.assertFalse(result.loc["2024-01-04", "allow_entry"])

    def test_friday_mask_does_not_promote_holiday_week_last_session(self):
        dates = pd.Series(["2024-01-04", "2024-01-05", "2024-02-08", "2024-02-09"])
        mask = friday_mask(dates)
        self.assertEqual(mask.tolist(), [False, True, False, True])

    def test_breadth_days_custom_return_col(self):
        panel = pd.DataFrame({
            "date": ["2024-01-02"] * 100,
            "adj_ret10": [0.05] * 60 + [-0.02] * 40,
        })
        result = breadth_days(panel, return_col="adj_ret10").set_index("date")
        self.assertEqual(result.loc["2024-01-02", "positive5_count"], 60)
        self.assertEqual(result.loc["2024-01-02", "breadth5"], 0.6)
        self.assertTrue(result.loc["2024-01-02", "allow_entry"])

    def test_week_last_day_mask(self):
        from exploration_features import week_last_day_mask
        # Week 1: Mon-Fri (Friday Jan 5 is last day)
        # Week 2: Mon-Thu (Thursday Jan 11 is last day because Friday Jan 12 is holiday)
        calendar = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
                    "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11"]
        dates = pd.Series(["2024-01-04", "2024-01-05", "2024-01-10", "2024-01-11"])
        mask = week_last_day_mask(dates, calendar)
        # 2024-01-05 is Friday (last of week 1) -> True
        # 2024-01-11 is Thursday (last of week 2 in calendar) -> True
        self.assertEqual(mask.tolist(), [False, True, False, True])


if __name__ == "__main__":
    unittest.main()
