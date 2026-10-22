import unittest
import pandas as pd

from accessibility import AccessibleReplay, accessible_candidates, allowed_buy
from engine import Config
from midterm_labels import relabel_panel


class AccessTests(unittest.TestCase):
    def test_known_effective_date_without_removing_earlier_history(self):
        panel = pd.DataFrame({'date': ['2025-06-19', '2025-06-20', '2025-06-20'],
                              'code': ['sz.000584', 'sz.000584', 'sz.002013'], 'eligible': [True] * 3})
        self.assertEqual(accessible_candidates(panel).code.tolist(), ['sz.000584', 'sz.002013'])
        self.assertTrue(allowed_buy('sz.000584', '2025-06-19'))
        self.assertFalse(allowed_buy('sz.000584', '2025-06-20'))

    def test_entry_permission_rechecked_next_open_and_sales_remain_allowed(self):
        calendar = ['2025-06-19', '2025-06-20', '2025-06-23']
        frame = pd.DataFrame([{'date': d, 'code': 'sz.000584', 'open': 10., 'high': 10., 'low': 10.,
            'close': 10., 'preclose': 10., 'volume': 1000, 'tradestatus': 1, 'isST': 0} for d in calendar])
        panel = pd.DataFrame([{'date': calendar[0], 'code': 'sz.000584'}])
        labels = relabel_panel(panel, {'sz.000584': frame}, calendar, [], Config(hold_days=5), AccessibleReplay)
        self.assertEqual(labels.label_reason.iloc[0], 'entry_not_tradable')
        replay = AccessibleReplay({'sz.000584': frame}, calendar,
            pd.DataFrame([{'date': calendar[0], 'code': 'sz.000584', 'score': 1.}]), [], Config(hold_days=5))
        replay.run()
        self.assertEqual(len(replay.fills), 0)
        self.assertEqual(replay.rejected['retirement_buy_permission'], 1)
        replay.execution_day = calendar[1]
        self.assertTrue(replay.tradable(frame.iloc[1].to_dict(), 'sell')[0])


if __name__ == '__main__':
    unittest.main()
