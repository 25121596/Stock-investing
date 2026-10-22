"""Unit tests for ensemble.py (LR + LGB model ensemble)."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ensemble import compare_with_singles, ensemble_score, ensemble_signals


class EnsembleTests(unittest.TestCase):
    def setUp(self):
        self.predictions = pd.DataFrame({
            "date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "code": ["sh.600001", "sh.600002", "sh.600001", "sh.600002"],
            "p_win": [0.60, 0.50, 0.58, 0.52],
            "pred_net_return": [0.010, 0.008, 0.002, 0.006],
        })

    def test_intersection_method(self):
        # sh.600001 on 2024-01-02 has p_win=0.60 >= 0.55 and pred_net_return=0.010 >= 0.005 -> Selected
        # sh.600002 on 2024-01-02 has p_win=0.50 < 0.55 -> Not selected
        # sh.600001 on 2024-01-03 has pred_net_return=0.002 < 0.005 -> Not selected
        # sh.600002 on 2024-01-03 has p_win=0.52 < 0.55 -> Not selected
        scored = ensemble_score(self.predictions, method="intersection",
                                p_win_threshold=0.55, return_threshold=0.005)
        self.assertEqual(scored["selected"].sum(), 1)
        self.assertEqual(scored.loc[scored.selected, "code"].iloc[0], "sh.600001")

    def test_ensemble_signals_ordering(self):
        signals = ensemble_signals(self.predictions, method="intersection")
        self.assertEqual(len(signals), 1)
        self.assertListEqual(list(signals.columns), ["date", "code", "score"])
        self.assertEqual(signals.iloc[0]["code"], "sh.600001")

    def test_average_and_weighted_methods(self):
        scored_avg = ensemble_score(self.predictions, method="average")
        self.assertIn("selected", scored_avg.columns)
        self.assertTrue(np.all(scored_avg["ensemble_score"].between(0, 1)))

        scored_w = ensemble_score(self.predictions, method="weighted", weights=(0.7, 0.3))
        self.assertIn("selected", scored_w.columns)

    def test_compare_with_singles(self):
        comparison = compare_with_singles(self.predictions, p_win_threshold=0.55, return_threshold=0.005)
        self.assertIn("counts", comparison)
        self.assertIn("jaccard_similarity", comparison)
        self.assertEqual(comparison["counts"]["ensemble_intersection"], 1)

    def test_missing_column_error(self):
        invalid = pd.DataFrame({"date": ["2024-01-02"], "code": ["sh.600001"]})
        with self.assertRaises(ValueError):
            ensemble_score(invalid)


if __name__ == "__main__":
    unittest.main()
