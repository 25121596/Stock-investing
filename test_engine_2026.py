import unittest

import pandas as pd

from engine_2026 import Replay2026


class Limits2026Tests(unittest.TestCase):
    def setUp(self):
        self.replay = Replay2026({}, ["2026-07-03", "2026-07-06"], pd.DataFrame(columns=["date", "code"]))
        self.bar = {"tradestatus": 1, "volume": 10000, "isST": 1, "open": 9.3, "preclose": 10}

    def test_st_five_percent_before_and_ten_after_effective_date(self):
        self.replay.execution_day = "2026-07-03"
        self.assertEqual(self.replay.tradable(self.bar, "sell"), (False, "limit_down_open"))
        self.replay.execution_day = "2026-07-06"
        self.assertEqual(self.replay.tradable(self.bar, "sell"), (True, ""))
        self.bar["open"] = 9.0
        self.assertEqual(self.replay.tradable(self.bar, "sell"), (False, "limit_down_open"))

    def test_st_buys_still_rejected_after_change(self):
        self.replay.execution_day = "2026-07-06"
        self.assertEqual(self.replay.tradable(self.bar, "buy"), (False, "ST_entry"))


if __name__ == "__main__":
    unittest.main()
