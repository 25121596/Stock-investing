"""Isolated operation-window and failure-recovery tests for the daily runner."""
from datetime import datetime as RealDateTime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import run_v081_parallel_day as daily
import v081_parallel_paper as paper


class FrozenClock:
    value = RealDateTime(2026, 10, 22, 16, 0, tzinfo=paper.CST)

    @classmethod
    def now(cls, tz=None):
        value = cls.value
        return value.astimezone(tz) if tz else value


class DailyOperationTests(unittest.TestCase):
    def setUp(self):
        FrozenClock.value = RealDateTime(2026, 10, 22, 16, 0, tzinfo=paper.CST)
        self.schedule = pd.DataFrame([
            {'calendar_date': '2026-10-21', 'is_trading_day': '1'},
            {'calendar_date': '2026-10-22', 'is_trading_day': '1'},
            {'calendar_date': '2026-10-23', 'is_trading_day': '1'},
        ])
        self.init = {'model_year': 2026, 'prospective_start_not_before': '2026-10-22'}

    def _run_patches(self, session, records, *, schedule=None):
        return (
            patch.object(daily, 'datetime', FrozenClock),
            patch.object(daily.paper, 'verify_records', return_value=(self.init, records, 'head')),
            patch.object(daily, 'cached_schedule', return_value=self.schedule if schedule is None else schedule),
            patch.object(daily.paper, 'validate_window', return_value='2026-10-23'),
        )

    def test_before_close_check_and_run_skip_without_subprocess_or_record(self):
        FrozenClock.value = RealDateTime(2026, 10, 22, 15, 29, 59, tzinfo=paper.CST)
        with tempfile.TemporaryDirectory() as temp:
            for check_only in (True, False):
                session = Path(temp) / str(check_only)
                session.mkdir()
                patches = self._run_patches(session, [])
                with patches[0], patches[1], patches[2], patches[3], \
                     patch.object(daily, 'run_child') as child:
                    state = daily.run_day(session, Path(temp) / 'bridge', check_only=check_only)
                self.assertEqual(state['status'], 'waiting_for_close')
                child.assert_not_called()
                self.assertFalse((session / '.daily-operation-lock.json').exists())
                self.assertFalse((session / 'operations').exists())

    def test_nontrading_day_skips_and_check_only_never_spawns_children(self):
        weekend_table = pd.DataFrame([
            {'calendar_date': '2026-10-22', 'is_trading_day': '0'},
            {'calendar_date': '2026-10-23', 'is_trading_day': '1'},
        ])
        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp) / 'session'
            session.mkdir()
            patches = self._run_patches(session, [], schedule=weekend_table)
            with patches[0], patches[1], patches[2], patches[3], \
                 patch.object(daily, 'run_child') as child:
                state = daily.run_day(session, Path(temp) / 'bridge')
            self.assertEqual(state, {'status': 'non_trading_day', 'date': '2026-10-22'})
            child.assert_not_called()
            self.assertFalse((session / '.daily-operation-lock.json').exists())

            with patch.object(daily, 'datetime', FrozenClock), \
                 patch.object(daily.paper, 'verify_records', return_value=(self.init, [], 'head')), \
                 patch.object(daily, 'cached_schedule', return_value=self.schedule), \
                 patch.object(daily.paper, 'validate_window', return_value='2026-10-23'), \
                 patch.object(daily, 'run_child') as child:
                state = daily.run_day(session, Path(temp) / 'bridge', check_only=True)
            self.assertEqual(state['status'], 'ready')
            child.assert_not_called()
            self.assertFalse((session / '.daily-operation-lock.json').exists())

    def test_cross_year_missing_calendar_and_missed_signal_day_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp) / 'session'
            session.mkdir()
            with patch.object(daily, 'datetime', FrozenClock), \
                 patch.object(daily.paper, 'verify_records', return_value=({**self.init, 'model_year': 2025}, [], 'head')):
                with self.assertRaisesRegex(ValueError, 'across years'):
                    daily.run_day(session, Path(temp) / 'bridge', check_only=True)

            missing_today = self.schedule[self.schedule.calendar_date.ne('2026-10-22')]
            patches = self._run_patches(session, [], schedule=missing_today)
            with patches[0], patches[1], patches[2], patches[3]:
                with self.assertRaisesRegex(ValueError, 'does not cover today uniquely'):
                    daily.run_day(session, Path(temp) / 'bridge', check_only=True)

            missed = [(Path(temp) / 'old-day', {'signal_date': '2026-10-20',
                                                  'execution_date': '2026-10-21'})]
            patches = self._run_patches(session, missed)
            with patches[0], patches[1], patches[2], patches[3]:
                with self.assertRaisesRegex(ValueError, 'signal day was missed'):
                    daily.run_day(session, Path(temp) / 'bridge', check_only=True)

    def test_same_day_sealed_input_is_reused_without_rewriting_signals(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            session, bridge = temp / 'session', temp / 'bridge'
            session.mkdir()
            day = session / 'days/2026-10-22'
            (day / 'input-snapshot').mkdir(parents=True)
            (day / 'signals.csv').write_text('immutable sealed signals', encoding='utf-8')
            original = (day / 'signals.csv').read_bytes()
            sealed = {'signal_date': '2026-10-22', 'execution_date': '2026-10-23'}
            records = [(day, sealed)]
            patches = self._run_patches(session, records)
            commands = []
            with patches[0], patches[1], patches[2], patches[3], \
                 patch.object(daily, 'run_child', side_effect=lambda command, cwd: commands.append(command)):
                result = daily.run_day(session, bridge)
            self.assertEqual(result['status'], 'completed')
            self.assertTrue(result['dataset'].endswith('input-snapshot'))
            flattened = [' '.join(map(str, command)) for command in commands]
            self.assertFalse(any('update_paper_data.py' in command for command in flattened))
            self.assertFalse(any('v081_parallel_paper.py record' in command for command in flattened))
            self.assertEqual((day / 'signals.csv').read_bytes(), original)
            self.assertFalse((session / '.daily-operation-lock.json').exists())
            operation = json.loads(next((session / 'operations').glob('*.json')).read_text(encoding='utf-8'))
            self.assertEqual(operation['status'], 'completed')

    def test_download_failure_blocks_record_and_bridge_logs_failure_and_cleans_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            session = temp / 'session'
            session.mkdir()
            patches = self._run_patches(session, [])
            commands = []

            def fail_download(command, cwd):
                commands.append(command)
                raise RuntimeError('synthetic fresh data download failed')

            with patches[0], patches[1], patches[2], patches[3], \
                 patch.object(daily, 'run_child', side_effect=fail_download):
                with self.assertRaisesRegex(RuntimeError, 'download failed'):
                    daily.run_day(session, temp / 'bridge')

            rendered = [' '.join(map(str, command)) for command in commands]
            self.assertEqual(len(commands), 1)
            self.assertIn('update_paper_data.py', rendered[0])
            self.assertFalse(any('v081_parallel_paper.py record' in command for command in rendered))
            self.assertFalse(any('v081_parallel_gateway.py' in command for command in rendered))
            self.assertFalse((session / '.daily-operation-lock.json').exists())
            logs = list((session / 'operations').glob('*.json'))
            self.assertEqual(len(logs), 1)
            operation = json.loads(logs[0].read_text(encoding='utf-8'))
            self.assertEqual(operation['status'], 'blocked')
            self.assertIn('synthetic fresh data download failed', operation['error'])


if __name__ == '__main__':
    unittest.main()
