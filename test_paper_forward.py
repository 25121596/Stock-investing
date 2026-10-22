"""Prospective timing, matched opportunities, and immutable ledger guards."""
from datetime import datetime
from pathlib import Path
import tempfile
import unittest

import pandas as pd

import paper_forward as paper


class ProspectiveRecordTests(unittest.TestCase):
    schedule = ['2026-10-13', '2026-10-14', '2026-10-15', '2026-10-16', '2026-10-19']

    def test_valid_close_and_preauction_boundary(self):
        now = datetime(2026, 10, 14, 23, 50, tzinfo=paper.CST)
        self.assertEqual(paper.validate_window('2026-10-14', self.schedule, now, '2026-10-14'), '2026-10-15')
        for hour, minute in [(9, 15), (9, 30), (10, 0)]:
            late = datetime(2026, 10, 15, hour, minute, tzinfo=paper.CST)
            with self.assertRaisesRegex(ValueError, 'deadline'):
                paper.validate_window('2026-10-14', self.schedule, late, '2026-10-14')

    def test_stale_incomplete_and_missed_session_are_rejected(self):
        now = datetime(2026, 10, 14, 23, 50, tzinfo=paper.CST)
        with self.assertRaisesRegex(ValueError, 'Stale'):
            paper.validate_window('2026-10-13', self.schedule, now, '2026-10-13')
        early = datetime(2026, 10, 14, 15, 0, tzinfo=paper.CST)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            paper.validate_window('2026-10-14', self.schedule, early, '2026-10-14')
        with self.assertRaisesRegex(ValueError, 'missed'):
            paper.validate_window('2026-10-14', self.schedule, now, '2026-10-14', {'execution_date': '2026-10-13'})

    def test_weekend_uses_exchange_next_trading_day(self):
        weekend = datetime(2026, 10, 18, 12, 0, tzinfo=paper.CST)
        self.assertEqual(paper.validate_window('2026-10-16', self.schedule, weekend, '2026-10-14'), '2026-10-19')

    def test_random_ranking_is_keyed_not_shifted_by_removed_rows(self):
        pool = pd.DataFrame({'date': ['2026-10-14'] * 3, 'code': ['a', 'b', 'c']})
        original = paper.random_matched(pool, 3, paper.SEEDS[0]).set_index('code').score.to_dict()
        subset = paper.random_matched(pool.iloc[1:], 2, paper.SEEDS[0]).set_index('code').score.to_dict()
        self.assertEqual(subset, {c: original[c] for c in ('b', 'c')})
        self.assertEqual(len(paper.random_matched(pool, 1, paper.SEEDS[0])), 1)

    def test_sealed_record_mutation_and_chain_are_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp)
            (session / 'runtime').mkdir()
            (session / 'runtime/freeze.json').write_text('{}')
            (session / 'paper_forward-source.py').write_bytes(Path(paper.__file__).read_bytes())
            paper.write_json(session / 'init.json', {
                'freeze_sha256': paper.digest(session / 'runtime/freeze.json'),
                'runner_sha256': paper.digest(paper.__file__)})
            folder = session / 'days/2026-10-14'
            folder.mkdir(parents=True)
            artifact = folder / 'signals.csv'
            artifact.write_text('sealed original')
            paper.write_json(folder / 'record.json', {
                'previous_record_sha256': paper.digest(session / 'init.json'),
                'files': {'signals.csv': paper.digest(artifact)}})
            self.assertEqual(len(paper.verify_records(session)[1]), 1)
            artifact.write_text('later changed')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                paper.verify_records(session)

    def test_next_open_precommitted_size_and_account_continuation(self):
        import sys
        saved_modules = {k: sys.modules.get(k) for k in ('engine', 'engine_2026', 'accessibility', 'corporate_events', 'exploration_features')}
        for k in saved_modules:
            sys.modules.pop(k, None)
        try:
            modules = paper.load_runtime(paper.FREEZE)
            code = 'sh.600006'
            days = ['2026-10-14', '2026-10-15', '2026-10-16']
            frame = pd.DataFrame([{'date': day, 'code': code, 'open': 10., 'close': close,
                'preclose': 10., 'volume': 100000., 'tradestatus': 1, 'isST': 0}
                for day, close in zip(days, [10., 10., 10.2])])
            frames = {code: frame}
            bars = {code: frame.set_index('date').to_dict('index')}
            signals = pd.DataFrame([{'date': days[0], 'code': code, 'score': 1.}])
            first = paper.simulate(frames, bars, days[:1], signals, [], 10, modules)
            self.assertEqual(first.cash, 30000.)
            self.assertEqual(len(first.fills), 0)
            self.assertEqual(first.pending_buys[0]['shares'], 400)
            second = paper.simulate(frames, bars, days[:2], signals, [], 10, modules)
            self.assertEqual(second.fills[0]['shares'], first.pending_buys[0]['shares'])
            self.assertLess(second.cash, 30000. - 4000.)
            third = paper.simulate(frames, bars, days, signals, [], 10, modules)
            self.assertEqual(third.cash, second.cash)
            self.assertEqual(third.positions[code].trade_id, second.positions[code].trade_id)
            self.assertAlmostEqual(third.nav() - second.nav(), 400 * .2)
        finally:
            for k, mod in saved_modules.items():
                if mod is not None:
                    sys.modules[k] = mod
                else:
                    sys.modules.pop(k, None)


if __name__ == '__main__':
    unittest.main()
