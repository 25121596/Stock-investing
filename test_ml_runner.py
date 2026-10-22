import unittest

import numpy as np
import pandas as pd

from ml_features import FEATURES
from run_ml import experiment_signals, training_mask


class TemporalTrainingTests(unittest.TestCase):
    def test_train_excludes_unfinished_labels_and_future_rows(self):
        data = {"date": ["2023-12-27", "2023-12-29", "2024-01-02", "2023-12-25"],
                "label_end": ["2023-12-29", "2024-01-03", "2024-01-05", None],
                "net_return": [.01, .02, .03, np.nan], "y_win": [1, 1, 1, np.nan]}
        data.update({feature: [0.0] * 4 for feature in FEATURES})
        panel = pd.DataFrame(data)
        self.assertEqual(training_mask(panel, "2024-01-01").tolist(), [True, False, False, False])
        revised = panel.copy()
        revised.loc[2, ["net_return", "y_win"]] = [-.99, 0]
        pd.testing.assert_frame_equal(panel.loc[training_mask(panel, "2024-01-01")],
                                      revised.loc[training_mask(revised, "2024-01-01")])

    def test_signals_do_not_filter_using_future_label_availability(self):
        rows = pd.DataFrame({"date": ["2024-01-02"] * 2, "code": ["sh.600001", "sh.600002"],
                             "p_win": [.60, .50], "pred_net_return": [.01, -.01],
                             "adj_ret5": [.01, .02], "adj_ret20": [.02, .03],
                             "net_return": [np.nan, .1], "label_end": [None, "2024-01-08"]})
        legacy = pd.DataFrame(columns=["date", "code", "score", "strategy"])
        cases = {case: signals for case, _label, _group, signals in experiment_signals(rows, legacy)}
        self.assertEqual(len(cases["logistic_rank"]), 2)
        self.assertEqual(cases["logistic_selective"].code.tolist(), ["sh.600001"])
        self.assertEqual(len(cases["lgb_rank"]), 2)
        self.assertEqual(cases["lgb_selective"].code.tolist(), ["sh.600001"])
        self.assertEqual(len(cases["random_seed2026101401"]), 2)


if __name__ == "__main__":
    unittest.main()
