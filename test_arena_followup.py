import unittest
import pandas as pd
from arena_followup import metrics,periods

class FollowupTests(unittest.TestCase):
    def test_only_profitable_trades_removed_and_open_pnl_retained(self):
        summary=dict(final_equity=1050,initial_capital=1000,execution_costs=5,average_exposure=.3)
        result=metrics(summary,pd.DataFrame({'net_pnl':[100.,-200.,120.]}))
        self.assertEqual(result['closed_pnl'],20)
        self.assertEqual(result['open_and_other_pnl'],30)
        self.assertEqual(result['top3_winner_pnl'],220)
        self.assertAlmostEqual(result['excluding_top3_contribution_return'],-.17)

    def test_loss_account_has_no_misleading_profit_share(self):
        result=metrics(dict(final_equity=990,initial_capital=1000,execution_costs=2,average_exposure=.1),pd.DataFrame({'net_pnl':[10.,-20.]}))
        self.assertIsNone(result['top1_share_account_profit'])

    def test_years_use_previous_nav_and_assign_trades_by_exit(self):
        eq=pd.DataFrame({'date':['2022-12-30','2023-01-03','2023-12-29'],'equity':[1100.,1080.,1210.]})
        trades=pd.DataFrame({'exit_date':['2023-01-03'],'net_pnl':[25.]})
        rows=periods(eq,trades,1000)
        self.assertAlmostEqual(rows[1]['return'],.1)
        self.assertEqual(rows[0]['closed_trades'],0)
        self.assertEqual(rows[1]['open_and_other_pnl_change'],85)
        self.assertAlmostEqual(sum(x['nav_change'] for x in rows),210)

    def test_empty_trades_keeps_account_gain_separate(self):
        result=metrics(dict(final_equity=1010,initial_capital=1000,execution_costs=0,average_exposure=0),pd.DataFrame())
        self.assertIsNone(result['win_rate'])
        self.assertAlmostEqual(result['excluding_top3_contribution_return'],.01)

if __name__=='__main__':unittest.main()
