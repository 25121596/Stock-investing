import unittest

import pandas as pd

from engine import Config
from midterm_labels import relabel_panel


def fixture(n=45, signal_i=0):
    calendar = pd.bdate_range("2025-12-01", periods=n).strftime("%Y-%m-%d").tolist()
    rows = []
    for _i, day in enumerate(calendar):
        rows.append({"date": day, "code": "600001.SH", "open": 10.0,
                     "high": 10.1, "low": 9.9, "close": 10.0,
                     "preclose": 10.0, "volume": 100000, "tradestatus": 1,
                     "isST": 0, "adj_close": 10.0})
    panel = pd.DataFrame([{"date": calendar[signal_i], "code": "600001.SH",
                           "adj_ret1": 0.123, "eligible": True,
                           "net_return": 99, "y_win": True,
                           "label_end": "old", "label_reason": "old"}])
    return calendar, rows, panel


class MidtermLabelsTests(unittest.TestCase):
    def test_hold20_exits_on_due_session_and_keeps_candidates_features(self):
        cal, rows, panel = fixture()
        result = relabel_panel(panel, {"600001.SH": pd.DataFrame(rows)}, cal, [], Config(hold_days=20))
        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "adj_ret1"], .123)
        self.assertEqual(result.loc[0, "label_end"], cal[21])
        self.assertEqual(result.loc[0, "label_reason"], "labeled")
        self.assertFalse(bool(result.loc[0, "y_win"]))  # fees make a flat trade negative

    def test_t_plus_1_stop_sells_next_open(self):
        cal, rows, panel = fixture()
        rows[1]["close"] = 9.6
        rows[2]["open"] = 9.5
        rows[2]["preclose"] = 9.6
        result = relabel_panel(panel, {"600001.SH": pd.DataFrame(rows)}, cal, [], Config(hold_days=20))
        self.assertEqual(result.loc[0, "label_end"], cal[2])

    def test_untradable_due_day_retries_within_delay_window(self):
        cal, rows, panel = fixture()
        rows[21]["tradestatus"] = 0
        result = relabel_panel(panel, {"600001.SH": pd.DataFrame(rows)}, cal, [], Config(hold_days=20))
        self.assertEqual(result.loc[0, "label_end"], cal[22])

    def test_corporate_action_interval_is_masked(self):
        cal, rows, panel = fixture()
        action = {"code": "600001.SH", "dividOperateDate": cal[10]}
        result = relabel_panel(panel, {"600001.SH": pd.DataFrame(rows)}, cal, [action], Config(hold_days=20))
        self.assertEqual(result.loc[0, "label_reason"], "corporate_action_interval")
        self.assertTrue(pd.isna(result.loc[0, "net_return"]))

    def test_truncated_window_remains_unlabeled(self):
        cal, rows, panel = fixture(n=30)
        for row in rows[21:]:
            row["tradestatus"] = 0
        result = relabel_panel(panel, {"600001.SH": pd.DataFrame(rows)}, cal, [], Config(hold_days=20))
        self.assertEqual(result.loc[0, "label_reason"], "right_censored")
        self.assertTrue(pd.isna(result.loc[0, "y_win"]))

    def test_label_can_cross_year_boundary_without_changing_candidate(self):
        cal, rows, panel = fixture(n=35)
        panel.loc[0, "date"] = "2025-12-31"
        rows = [{**row, "date": day} for row, day in zip(rows, cal)]
        # Place the candidate on the final 2025 trading day in this synthetic calendar.
        index = cal.index("2025-12-31")
        panel.loc[0, "date"] = cal[index]
        result = relabel_panel(panel, {"600001.SH": pd.DataFrame(rows)}, cal, [], Config(hold_days=5))
        self.assertEqual(result.loc[0, "date"], cal[index])
        self.assertEqual(result.loc[0, "label_end"], cal[index + 6])
        self.assertGreater(int(result.loc[0, "label_end"][:4]), 2025)


if __name__ == "__main__":
    unittest.main()
