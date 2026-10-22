"""Unit tests for monitor_drift.py feature drift monitoring and Rank IC analysis."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from monitor_drift import (
    feature_drift_report,
    population_stability_index,
    rank_ic_series,
    rolling_ic_summary,
)


class TestPopulationStabilityIndex(unittest.TestCase):
    def test_identical_distributions_have_zero_or_negligible_psi(self):
        rng = np.random.RandomState(42)
        expected = rng.normal(loc=0.0, scale=1.0, size=5000)
        actual = expected.copy()
        psi = population_stability_index(expected, actual, bins=10)
        self.assertAlmostEqual(psi, 0.0, places=4)

    def test_minor_shift_has_low_psi(self):
        rng = np.random.RandomState(42)
        expected = rng.normal(loc=0.0, scale=1.0, size=5000)
        actual = rng.normal(loc=0.05, scale=1.0, size=5000)
        psi = population_stability_index(expected, actual, bins=10)
        self.assertLess(psi, 0.10)

    def test_moderate_shift_status(self):
        rng = np.random.RandomState(42)
        expected = rng.normal(loc=0.0, scale=1.0, size=5000)
        actual = rng.normal(loc=0.35, scale=1.0, size=5000)
        psi = population_stability_index(expected, actual, bins=10)
        self.assertGreaterEqual(psi, 0.08)

    def test_significant_shift_status(self):
        rng = np.random.RandomState(42)
        expected = rng.normal(loc=0.0, scale=1.0, size=5000)
        actual = rng.normal(loc=1.5, scale=1.0, size=5000)
        psi = population_stability_index(expected, actual, bins=10)
        self.assertGreater(psi, 0.25)

    def test_constant_distribution_identical(self):
        expected = np.full(100, 3.14)
        actual = np.full(100, 3.14)
        psi = population_stability_index(expected, actual)
        self.assertEqual(psi, 0.0)

    def test_constant_distribution_shifted(self):
        expected = np.full(100, 0.0)
        actual = np.full(100, 1.0)
        psi = population_stability_index(expected, actual)
        self.assertGreater(psi, 0.25)

    def test_empty_or_nan_inputs(self):
        self.assertTrue(np.isnan(population_stability_index([], [])))
        self.assertTrue(np.isnan(population_stability_index([np.nan, np.nan], [1.0, 2.0])))
        self.assertTrue(np.isnan(population_stability_index([1.0, 2.0], [np.nan, np.nan])))

    def test_handles_nan_and_inf_cleanly(self):
        rng = np.random.RandomState(42)
        expected = rng.normal(0, 1, 1000)
        actual = rng.normal(0, 1, 1000)
        # inject nans and infs
        expected[0] = np.nan
        expected[1] = np.inf
        actual[0] = np.nan
        actual[1] = -np.inf
        psi = population_stability_index(expected, actual)
        self.assertFalse(np.isnan(psi))
        self.assertLess(psi, 0.10)

    def test_invalid_bins(self):
        with self.assertRaises(ValueError):
            population_stability_index([1, 2, 3], [1, 2, 3], bins=0)


class TestFeatureDriftReport(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(42)
        n = 1000
        self.train_df = pd.DataFrame({
            "f_stable": rng.normal(0, 1, n),
            "f_moderate": rng.normal(0, 1, n),
            "f_alert": rng.normal(0, 1, n),
            "f_constant": np.zeros(n),
        })
        self.live_df = pd.DataFrame({
            "f_stable": rng.normal(0.02, 1, n),
            "f_moderate": rng.normal(0.35, 1, n),
            "f_alert": rng.normal(2.0, 1, n),
            "f_constant": np.zeros(n),
        })

    def test_report_structure_and_status(self):
        features = ["f_stable", "f_moderate", "f_alert", "f_constant", "f_missing"]
        report = feature_drift_report(self.train_df, self.live_df, features=features)

        self.assertListEqual(list(report.columns), ["feature", "psi", "status"])
        self.assertEqual(len(report), 5)

        status_map = dict(zip(report["feature"], report["status"]))
        self.assertEqual(status_map["f_stable"], "ok")
        self.assertEqual(status_map["f_alert"], "alert")
        self.assertEqual(status_map["f_constant"], "ok")
        self.assertEqual(status_map["f_missing"], "missing")

    def test_csv_export_utf8_sig(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "drift_report.csv"
            report = feature_drift_report(
                self.train_df,
                self.live_df,
                features=["f_stable"],
                output_path=out_path,
            )
            self.assertTrue(out_path.is_file())
            # Verify utf-8-sig encoding and readable
            raw = out_path.read_bytes()
            self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
            loaded = pd.read_csv(out_path, encoding="utf-8-sig")
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded["feature"].iloc[0], "f_stable")


class TestRankICSeries(unittest.TestCase):
    def test_perfect_positive_and_negative_rank_ic(self):
        dates = ["2026-01-05"] * 20 + ["2026-01-06"] * 20
        codes = [f"sh.6000{i:02d}" for i in range(20)] * 2
        # Day 1: perfect positive correlation
        # Day 2: perfect negative correlation
        x1 = np.arange(20, dtype=float)
        y1 = x1 * 2.5 + 1.0
        x2 = np.arange(20, dtype=float)
        y2 = -x2 * 3.0

        preds = pd.DataFrame({
            "date": dates,
            "code": codes,
            "pred_net_return": np.r_[x1, x2],
            "net_return": np.r_[y1, y2],
        })

        ic_s = rank_ic_series(preds, actual_col="net_return", pred_col="pred_net_return", min_obs=10)
        self.assertIsInstance(ic_s, pd.Series)
        self.assertEqual(len(ic_s), 2)
        self.assertAlmostEqual(ic_s.loc["2026-01-05"], 1.0, places=5)
        self.assertAlmostEqual(ic_s.loc["2026-01-06"], -1.0, places=5)

    def test_min_obs_filtering(self):
        preds = pd.DataFrame({
            "date": ["2026-01-05"] * 5 + ["2026-01-06"] * 15,
            "code": [f"c_{i}" for i in range(20)],
            "pred_net_return": np.arange(20, dtype=float),
            "net_return": np.arange(20, dtype=float),
        })
        ic_s = rank_ic_series(preds, min_obs=10)
        self.assertNotIn("2026-01-05", ic_s.index)
        self.assertIn("2026-01-06", ic_s.index)

    def test_missing_column_raises(self):
        df = pd.DataFrame({"date": ["2026-01-05"], "pred_net_return": [0.01]})
        with self.assertRaises(KeyError):
            rank_ic_series(df, actual_col="missing_col")

    def test_zero_variance_does_not_crash(self):
        preds = pd.DataFrame({
            "date": ["2026-01-05"] * 15,
            "code": [f"c_{i}" for i in range(15)],
            "pred_net_return": np.ones(15),  # all identical
            "net_return": np.arange(15, dtype=float),
        })
        ic_s = rank_ic_series(preds, min_obs=10)
        self.assertTrue(ic_s.empty)


class TestRollingICSummary(unittest.TestCase):
    def test_rolling_metrics_values(self):
        dates = pd.date_range("2026-01-01", periods=10, freq="B").strftime("%Y-%m-%d")
        ic_values = [0.1, 0.2, 0.3, -0.1, 0.05, 0.15, -0.05, 0.25, 0.1, 0.2]
        s = pd.Series(ic_values, index=dates, name="rank_ic")

        summary = rolling_ic_summary(s, window=5, min_periods=3)
        self.assertListEqual(list(summary.columns), ["date", "rolling_ic", "ic_ir", "pct_positive"])
        self.assertEqual(len(summary), 10)

        # Check last row (window of 5: [0.15, -0.05, 0.25, 0.1, 0.2])
        last_5 = np.array([0.15, -0.05, 0.25, 0.1, 0.2])
        expected_mean = float(np.mean(last_5))
        expected_std = float(np.std(last_5, ddof=1))
        expected_ir = expected_mean / expected_std
        expected_pos = 4.0 / 5.0

        self.assertAlmostEqual(summary["rolling_ic"].iloc[-1], expected_mean, places=5)
        self.assertAlmostEqual(summary["ic_ir"].iloc[-1], expected_ir, places=5)
        self.assertAlmostEqual(summary["pct_positive"].iloc[-1], expected_pos, places=5)

    def test_empty_series_returns_empty_dataframe(self):
        s = pd.Series(dtype=float)
        summary = rolling_ic_summary(s, window=60)
        self.assertTrue(summary.empty)
        self.assertListEqual(list(summary.columns), ["date", "rolling_ic", "ic_ir", "pct_positive"])


if __name__ == "__main__":
    unittest.main()
