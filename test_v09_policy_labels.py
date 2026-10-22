import unittest
import pandas as pd
from engine import Config
from v09_policy_labels import policy_labels
from v08_engine_verified import VerifiedV08Replay


class PolicyLabelTests(unittest.TestCase):
    def compare_case(self,closes,opens=None,one_gap=False):
        dates=pd.bdate_range('2025-03-03',periods=len(closes)).strftime('%Y-%m-%d').tolist()
        opens=opens or [10.]*len(closes)
        rows=[dict(date=d,code='sh.600006',open=o,high=max(o,c),low=min(o,c),close=c,
            preclose=closes[max(0,k-1)],volume=100000,tradestatus=1,isST=0)
            for k,(d,o,c) in enumerate(zip(dates,opens,closes))]
        frame=pd.DataFrame(rows);frames={'sh.600006':frame}
        panel=pd.DataFrame([dict(date=dates[0],code='sh.600006')])
        breadth=pd.DataFrame([dict(date=d,pool_size=200,breadth5=.5) for d in dates])
        vol={(d,'sh.600006'):.025 for d in dates}
        labels,_=policy_labels(panel,frames,dates,[],breadth,vol)
        replay=VerifiedV08Replay(frames,dates,pd.DataFrame([dict(date=dates[0],code='sh.600006',score=1.)]),[],
            Config(order_budget=7500,max_positions=5,max_exposure=.9),vol_map=vol,
            daily_limits={d:1 for d in dates},breadth_lookup={d:.5 for d in dates},enable_cash_overlay=False).run()
        if pd.notna(labels.iloc[0].net_return):
            self.assertEqual(len(replay.trades),1)
            self.assertAlmostEqual(labels.iloc[0].net_return,replay.trades[0]['net_return'],places=12)
            self.assertEqual(labels.iloc[0].label_end,replay.trades[0]['exit_date'])
        return labels,replay,dates

    def test_neutral_horizon_is_eight_market_sessions(self):
        labels,replay,dates=self.compare_case([10.]*14)
        self.assertEqual(labels.iloc[0].label_end,dates[9])
        self.assertEqual(labels.iloc[0].label_reason,'policy_horizon')

    def test_profit_lock_exits_at_later_open_not_guaranteed_floor(self):
        closes=[10.,10.9,10.3,9.9]+[9.9]*10
        opens=[10.,10.,10.9,9.9]+[9.9]*10
        labels,replay,dates=self.compare_case(closes,opens)
        self.assertEqual(labels.iloc[0].label_end,dates[3])
        self.assertLess(labels.iloc[0].net_return,0)

    def test_gap_rejection_leaves_candidate_unlabelled(self):
        labels,replay,_=self.compare_case([10.]*14,[10.,10.5]+[10.]*12)
        self.assertTrue(pd.isna(labels.iloc[0].net_return))
        self.assertEqual(labels.iloc[0].label_reason,'entry_gap_limit')
        self.assertEqual(len(labels),1)
        self.assertEqual(len(replay.fills),0)

    def test_incomplete_horizon_is_not_realized_profit(self):
        labels,replay,_=self.compare_case([10.]*3)
        self.assertTrue(pd.isna(labels.iloc[0].net_return))
        self.assertEqual(labels.iloc[0].label_reason,'right_censored')
        self.assertEqual(len(replay.positions),1)


if __name__=='__main__':unittest.main()
