import unittest

from corporate_events import canonicalize_events


def event(**kwargs):
    return {'code': 'sh.601988', 'dividRegistDate': '2025-01-22', 'dividOperateDate': '2025-01-23',
        'dividPayDate': '2025-01-23', 'dividCashPsBeforeTax': '0.1208', 'dividStocksPs': '0',
        'dividReserveToStockPs': '', **kwargs}


class CorporateEventTests(unittest.TestCase):
    def test_one_actual_entitlement_is_paid_once_in_account(self):
        import pandas as pd
        from engine import Config, Replay
        days = ['2025-01-21', '2025-01-22', '2025-01-23']
        frame = pd.DataFrame([{'date': d, 'code': 'sh.601988', 'open': p, 'high': p, 'low': p,
            'close': p, 'preclose': p, 'volume': 100000, 'tradestatus': 1, 'isST': 0}
            for d, p in zip(days, [10., 10., 9.88])])
        signal = pd.DataFrame([{'date': days[0], 'code': 'sh.601988', 'score': 1.}])
        clean, _ = canonicalize_events([event(), event(dividPlanDate='2025-01-15')])
        replay = Replay({'sh.601988': frame}, days, signal, clean, Config(hold_days=1)).run()
        self.assertEqual(len(replay.actions), 1)
        self.assertEqual(len(replay.trades), 1)
        self.assertAlmostEqual(replay.trades[0]['net_dividends'], 400 * .1208 * .8)
        self.assertEqual(replay.summary()['quality_warning_count'], 0)

    def test_metadata_variants_are_one_cash_delivery(self):
        clean, audit = canonicalize_events([event(dividCashPsAfterTax=''),
            event(dividCashPsAfterTax='0.10872或0.1208', dividPlanDate='2025-01-15')])
        self.assertEqual(len(clean), 1)
        self.assertEqual(audit['removed_rows'], 1)
        self.assertEqual(clean[0]['dividPlanDate'], '2025-01-15')

    def test_different_economic_deliveries_remain_separate(self):
        clean, _ = canonicalize_events([event(), event(dividCashPsBeforeTax='0.1216')])
        self.assertEqual(len(clean), 2)

    def test_conflicting_payment_dates_block_instead_of_guessing(self):
        with self.assertRaises(ValueError):
            canonicalize_events([event(), event(dividPayDate='2025-01-24')])

    def test_idempotent_and_missing_dates_not_silently_merged(self):
        clean, _ = canonicalize_events([event(), event(dividCashPsBeforeTax='0.120800')])
        second, audit = canonicalize_events(clean)
        self.assertEqual(clean, second)
        self.assertEqual(audit['removed_rows'], 0)
        incomplete, audit = canonicalize_events([event(dividRegistDate=''), event(dividRegistDate='')])
        self.assertEqual(len(incomplete), 2)
        self.assertEqual(len(audit['unresolved_incomplete_groups']), 1)
