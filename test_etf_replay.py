import unittest

from engine import Config
from etf_replay import ETFReplay, price_tick


class ETFTests(unittest.TestCase):
    def test_later_order_shrinks_with_equity_before_next_open(self):
        import pandas as pd
        days = pd.bdate_range('2024-01-02', periods=8).strftime('%Y-%m-%d').tolist()
        prices = [10., 10., 9.1, 8.5, 8.5, 8.5, 8.5, 8.5]
        frame = pd.DataFrame([{'date': d, 'code': 'sh.510300', 'open': p, 'high': p, 'low': p,
            'close': p, 'preclose': prices[max(0, i-1)], 'volume': 100000, 'tradestatus': 1, 'isST': 0}
            for i, (d, p) in enumerate(zip(days, prices))])
        signals = pd.DataFrame([{'date': days[i], 'code': 'sh.510300', 'score': 1.} for i in (0, 4)])
        config = Config(order_budget=15000, max_positions=1, max_exposure=.5, hold_days=2,
            stop_loss=0, stamp_tax=0, transfer_fee=0, dividend_tax=0)
        replay = ETFReplay({'sh.510300': frame}, days, signals, [], config).run()
        buys = [f for f in replay.fills if f['side']=='buy']
        self.assertEqual(len(buys), 2)
        self.assertEqual(replay.rejected['exposure_limit'], 0)
        for plan in replay.order_plans:
            self.assertLessEqual(plan['shares'] * plan['limit'], plan['equity_at_signal'] * .5)
        self.assertEqual(buys[1]['shares'], replay.order_plans[1]['shares'])

    def test_price_limit_uses_mill_tick(self):
        replay = ETFReplay({}, ['2024-01-02'], __import__('pandas').DataFrame(columns=['date', 'code']))
        bar = {'open': 2.223, 'preclose': 2.021, 'tradestatus': 1, 'volume': 1000}
        self.assertEqual(price_tick(2.021 * 1.1), 2.223)
        self.assertFalse(replay.tradable(bar, 'buy')[0])
        bar['open'] = 2.222
        self.assertTrue(replay.tradable(bar, 'buy')[0])

    def test_etf_fee_scenario_excludes_stock_taxes(self):
        config = Config(stamp_tax=0, transfer_fee=0, dividend_tax=0)
        fees = config.fees(15000, sell=True)
        self.assertEqual(fees['commission'], 5.)
        self.assertEqual(fees['stamp'], 0.)
        self.assertEqual(fees['transfer'], 0.)
