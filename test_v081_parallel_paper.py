"""Synthetic invariants for the independent V0.8.1 parallel paper batch."""
from datetime import datetime as RealDateTime
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import v081_parallel_paper as paper
import v08_paper_forward_v3 as legacy


def capture_import_state():
    return list(sys.path), dict(sys.modules)


def restore_import_state(state):
    original_path, original_modules = state
    frozen_sources = (Path(paper.FREEZE) / 'sources').resolve()
    for name, module in list(sys.modules.items()):
        source = getattr(module, '__file__', None)
        in_frozen_sources = False
        if source:
            try:
                in_frozen_sources = Path(source).resolve().is_relative_to(frozen_sources)
            except (OSError, ValueError):
                pass
        if in_frozen_sources:
            if name in original_modules:
                sys.modules[name] = original_modules[name]
            else:
                sys.modules.pop(name, None)
    sys.path[:] = original_path


class ParallelPolicyTests(unittest.TestCase):
    def test_candidate_signal_and_random_accounts_are_matched_to_baseline(self):
        rows = []
        for i in range(101):
            rows.append({'date': '2026-10-22', 'code': f'sh.{600000+i:06d}',
                         'pred_net_return_v2': .01 if i < 4 else -.01,
                         'pred_score_v08': float(101-i)})
        panel = pd.DataFrame(rows)
        breadth = pd.DataFrame([{'date': '2026-10-22', 'pool_size': 101, 'breadth5': .58}])
        signals = paper.make_signals(panel, breadth)
        self.assertEqual(signals['v08_verified'].code.tolist(), signals['v081_breadth58'].code.tolist())
        for seed in paper.SEEDS:
            base = signals[f'v08_verified_random_{seed}'].set_index('code').score.to_dict()
            candidate = signals[f'v081_breadth58_random_{seed}'].set_index('code').score.to_dict()
            self.assertEqual(candidate, base)
            self.assertEqual(len(candidate), len(signals['v08_verified']))
        self.assertEqual(signals['cash'].empty, True)

    def test_breadth_gate_is_strict_at_58_percent_and_only_removes_buys(self):
        proposed = [{'code': 'sh.600001', 'signal_date': '2026-10-22', 'shares': 300, 'limit': 10.0}]
        kept, cancelled = paper.apply_candidate_gate(proposed, .58, True)
        self.assertEqual((kept, cancelled), (proposed, 0))
        kept, cancelled = paper.apply_candidate_gate(proposed, .579999, True)
        self.assertEqual((kept, cancelled), ([], 1))
        baseline, cancelled = paper.apply_candidate_gate(proposed, .40, False)
        self.assertEqual((baseline, cancelled), (proposed, 0))
        with self.assertRaisesRegex(ValueError, 'prior-close breadth'):
            paper.apply_candidate_gate(proposed, float('nan'), True)

    def test_random_control_is_stable_when_unrelated_candidates_are_removed(self):
        pool = pd.DataFrame({'date': ['2026-10-22']*4, 'code': ['a', 'b', 'c', 'd']})
        whole = legacy.random_matched(pool, 4, paper.SEEDS[0]).set_index('code').score.to_dict()
        subset = legacy.random_matched(pool.iloc[1:], 3, paper.SEEDS[0]).set_index('code').score.to_dict()
        self.assertEqual(subset, {code: whole[code] for code in ('b', 'c', 'd')})

    def test_candidate_cancels_new_buy_but_keeps_existing_stop_exit_state(self):
        import_state = capture_import_state()
        for name in ('engine', 'engine_2026', 'accessibility', 'corporate_events', 'exploration_features'):
            sys.modules.pop(name, None)
        old_panel = legacy._SIM_PANEL
        try:
            modules = legacy._base_load_runtime(paper.FREEZE)
            days = ['2026-10-22', '2026-10-23']
            codes = ['sh.600006', 'sh.600009']
            frames, bars = {}, {}
            for code in codes:
                closes = [10., 9.4] if code == codes[0] else [10., 10.]
                frame = pd.DataFrame([{'date': day, 'code': code, 'open': 10., 'high': 10.,
                    'low': close, 'close': close, 'preclose': 10., 'volume': 100000.,
                    'tradestatus': 1, 'isST': 0} for day, close in zip(days, closes)])
                frames[code] = frame
                bars[code] = frame.set_index('date').to_dict('index')
            context = []
            for day, positives in zip(days, (70, 57)):
                for i in range(101):
                    context.append({'date': day, 'code': codes[0] if i == 0 else f'x{i:03d}',
                        'adj_ret5': .01 if i < positives else -.01, 'ret_vol20': .02})
            panel = pd.DataFrame(context)
            signals = pd.DataFrame([
                {'date': days[0], 'code': codes[0], 'score': 2.},
                {'date': days[1], 'code': codes[1], 'score': 1.},
            ])
            spec = paper.accounts()['v081_breadth58']
            replay = paper.simulate(frames, bars, days, signals, [], spec, modules, panel)
            self.assertIn(codes[0], replay.positions)
            self.assertTrue(replay.positions[codes[0]].stop_pending)
            self.assertEqual(replay.pending_buys, [])
            last = replay.gate_decisions[-1]
            self.assertEqual(last['action'], 1)
            self.assertEqual(last['proposed_orders'][0]['code'], codes[1])
            self.assertEqual(last['accepted_orders'], [])
            self.assertTrue(legacy.v08_sell_due(replay, replay.positions[codes[0]], days))
        finally:
            legacy._SIM_PANEL = old_panel
            restore_import_state(import_state)

    def test_window_requires_timezone_latest_close_deadline_and_contiguous_accounting(self):
        schedule = ['2026-10-21', '2026-10-22', '2026-10-23', '2026-10-26']
        now = RealDateTime(2026, 10, 22, 16, 0, tzinfo=paper.CST)
        self.assertEqual(paper.validate_window('2026-10-22', schedule, now, '2026-10-22'), '2026-10-23')
        with self.assertRaisesRegex(ValueError, 'Timezone-aware'):
            paper.validate_window('2026-10-22', schedule, RealDateTime(2026, 10, 22, 16), '2026-10-22')
        with self.assertRaisesRegex(ValueError, 'Missed pre-auction'):
            paper.validate_window('2026-10-22', schedule,
                                  RealDateTime(2026, 10, 23, 9, 15, tzinfo=paper.CST), '2026-10-22')
        with self.assertRaisesRegex(ValueError, 'trading-day record was missed'):
            paper.validate_window('2026-10-22', schedule, now, '2026-10-22',
                                  {'execution_date': '2026-10-21'})


class ParallelIntegrityTests(unittest.TestCase):
    def test_two_day_record_day_end_to_end_rebuilds_accounts_and_exports_status(self):
        import exploration_features

        import_state = capture_import_state()
        for name in ('engine', 'engine_2026', 'accessibility', 'corporate_events', 'exploration_features'):
            sys.modules.pop(name, None)
        modules = legacy._base_load_runtime(paper.FREEZE)
        dates = ['2026-10-22', '2026-10-23']

        class FixtureClock(RealDateTime):
            values = []
            index = 0

            @classmethod
            def now(cls, tz=None):
                value = cls.values[min(cls.index, len(cls.values) - 1)]
                cls.index += 1
                return value.astimezone(tz) if tz else value

        def as_beijing(day):
            return RealDateTime.fromisoformat(day + 'T16:00:00+08:00')

        FixtureClock.values = [as_beijing(dates[0])]
        FixtureClock.index = 0
        root = Path(paper.ROOT)
        source_names = ['v08_paper_forward_v3.py', 'v08_inference_verified.py',
                        'v08_engine_verified.py', 'exploration_features.py']
        source_hashes = {name: paper.digest(root / name) for name in source_names}

        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            freeze = temp / 'fixture-freeze'
            (freeze / 'sources').mkdir(parents=True)
            (freeze / 'freeze.json').write_text('{}', encoding='utf-8')
            (freeze / '2026-lgb.txt').write_text('synthetic model identity', encoding='utf-8')
            sentinel = temp / 'prior-batch-seal.json'
            sentinel.write_text('unchanged prior batch', encoding='utf-8')
            session = temp / 'parallel-session'
            data_roots = [temp / f'dataset-{day}' for day in dates]
            schedule = ['2026-10-21', '2026-10-22', '2026-10-23', '2026-10-26']
            for day, folder in zip(dates, data_roots):
                folder.mkdir()
                paper.write_new(folder / 'manifest.json', {'end': day})
                pd.DataFrame({'calendar_date': schedule, 'is_trading_day': ['1'] * len(schedule)}).to_csv(
                    folder / 'exchange-calendar.csv', index=False)

            codes = [f'sh.{600006+i:06d}' for i in range(101)]
            code_a, code_b = codes[0], codes[1]

            def data_for(cutoff):
                calendar = dates[:dates.index(cutoff) + 1]
                frames = {}
                for code in codes:
                    frame = pd.DataFrame([{'date': day, 'code': code, 'open': 10., 'high': 10.,
                        'low': 10., 'close': 10., 'preclose': 10., 'volume': 100000.,
                        'tradestatus': 1, 'isST': 0} for day in calendar])
                    frames[code] = frame
                manifest = {'end': cutoff, 'failures': [], 'complete': True}
                return manifest, frames, calendar, [], schedule

            def fake_infer(frames, calendar, cutoff, runtime, loaded_modules):
                positive_count = 66 if cutoff == dates[0] else 51
                panel = pd.DataFrame([{'date': cutoff, 'code': code,
                    'adj_ret5': .01 if i < positive_count else -.01, 'ret_vol20': .02,
                    'pred_net_return_v2': .01 if i == (0 if cutoff == dates[0] else 1) else -.01,
                    'pred_score_v08': float(101-i)} for i, code in enumerate(codes)])
                breadth = exploration_features.breadth_days(panel)
                main_code = code_a if cutoff == dates[0] else code_b
                main = pd.DataFrame([{'date': cutoff, 'code': main_code, 'score': 1.}])
                result = {}
                for name, spec in paper.accounts().items():
                    if spec['policy'] == 'cash':
                        result[name] = pd.DataFrame(columns=['date', 'code', 'score'])
                    elif spec['seed'] is None:
                        result[name] = main.copy()
                    else:
                        result[name] = legacy.random_matched(panel, 1, spec['seed'])
                return panel, breadth, result

            def fake_load_dataset(folder, cutoff):
                return data_for(cutoff)

            try:
                with patch.object(paper, 'FREEZE', freeze), \
                     patch.object(legacy, 'verify_freeze', return_value={'source_hashes': source_hashes}), \
                     patch.object(paper, 'protected_hashes', return_value={str(sentinel): paper.digest(sentinel)}), \
                     patch.object(paper, 'runtime_versions', return_value={'synthetic': '1'}), \
                     patch.object(paper, 'datetime', FixtureClock):
                    paper.initialize(session)

                FixtureClock.values = [as_beijing(dates[0]), as_beijing(dates[0]),
                                       as_beijing(dates[1]), as_beijing(dates[1])]
                FixtureClock.index = 0
                with patch.object(paper, 'FREEZE', freeze), \
                     patch.object(legacy, 'verify_freeze', return_value={'source_hashes': source_hashes}), \
                     patch.object(paper, 'runtime_versions', return_value={'synthetic': '1'}), \
                     patch.object(paper, 'datetime', FixtureClock), \
                     patch.object(paper, 'load_runtime', return_value=modules), \
                     patch.object(paper, 'load_dataset', side_effect=fake_load_dataset), \
                     patch.object(paper, 'infer', side_effect=fake_infer), \
                     patch.object(pd.DataFrame, 'to_markdown', side_effect=ImportError('tabulate unavailable')) as markdown:
                    paper.record_day(session, data_roots[0])
                    paper.record_day(session, data_roots[1])
                    init, records, _ = paper.verify_records(session)
                    markdown.assert_not_called()

                self.assertEqual([record['signal_date'] for _, record in records], dates)
                self.assertEqual(len(init['accounts']), 9)
                self.assertTrue((session / 'status.md').is_file())
                self.assertTrue((session / 'status.html').is_file())
                self.assertIn('已封存信号日：**2**', (session / 'status.md').read_text(encoding='utf-8'))

                first_account = paper.read_json(records[0][0] / 'v08_verified/account.json')
                candidate_account = paper.read_json(records[1][0] / 'v081_breadth58/account.json')
                self.assertEqual(first_account['account'], 'v08_verified')
                self.assertEqual(candidate_account['account'], 'v081_breadth58')
                for day, expected_code in zip(dates, (code_a, code_b)):
                    signals = pd.read_csv(records[dates.index(day)][0] / 'signals.csv', dtype={'date': str, 'code': str})
                    base_code = signals[(signals.account == 'v08_verified') & (signals.date == day)].code.tolist()
                    candidate_code = signals[(signals.account == 'v081_breadth58') & (signals.date == day)].code.tolist()
                    self.assertEqual(base_code, [expected_code])
                    self.assertEqual(candidate_code, base_code)

                latest = records[-1][0]
                baseline_orders = paper.read_json(latest / 'v08_verified/next-open-attempts.json')
                candidate_orders = paper.read_json(latest / 'v081_breadth58/next-open-attempts.json')
                self.assertEqual([order['code'] for order in baseline_orders['buy_attempts']], [code_b])
                self.assertEqual(candidate_orders['buy_attempts'], [])
                gate = pd.read_csv(latest / 'v081_breadth58/gate-decisions.csv')
                self.assertEqual(int(gate.iloc[-1].action), 1)
                self.assertEqual(int(gate.iloc[-1].cancelled_orders), 1)
                for name, spec in init['accounts'].items():
                    account = paper.read_json(latest / name / 'account.json')
                    self.assertEqual(account['account'], name)
                    self.assertEqual(account['spec'], spec)
            finally:
                restore_import_state(import_state)

    def test_baseline_replay_matches_original_v08_engine_on_synthetic_bars(self):
        import_state = capture_import_state()
        for name in ('engine', 'engine_2026', 'accessibility', 'corporate_events', 'exploration_features'):
            sys.modules.pop(name, None)
        old_panel = legacy._SIM_PANEL
        try:
            # The legacy loader accepts the frozen runtime directory itself;
            # the new batch wrapper accepts a session directory containing it.
            modules = legacy._base_load_runtime(paper.FREEZE)
            code, days = 'sh.600006', ['2026-10-22', '2026-10-23']
            frame = pd.DataFrame([{'date': d, 'code': code, 'open': 10., 'high': 10.1,
                'low': 9.9, 'close': 10., 'preclose': 10., 'volume': 100000.,
                'tradestatus': 1, 'isST': 0} for d in days])
            frames = {code: frame}
            bars = {code: frame.set_index('date').to_dict('index')}
            panel_rows = []
            for day in days:
                for i in range(101):
                    panel_rows.append({'date': day, 'code': code if i == 0 else f'x{i:03d}',
                        'adj_ret5': .01 if i < 59 else -.01, 'ret_vol20': .02})
            panel = pd.DataFrame(panel_rows)
            signals = pd.DataFrame([{'date': days[0], 'code': code, 'score': 1.}])
            spec = paper.accounts()['v08_verified']
            got = paper.simulate(frames, bars, days, signals, [], spec, modules, panel)
            legacy._SIM_PANEL = panel
            expected = legacy.simulate(frames, bars, days, signals, [], 10, modules)
            self.assertEqual(paper.object_hash(paper.account_state(got)),
                             paper.object_hash(legacy.account_state(expected)))
            self.assertEqual(paper.object_hash(paper.ledger(got)),
                             paper.object_hash(legacy.ledger(expected)))
        finally:
            legacy._SIM_PANEL = old_panel
            restore_import_state(import_state)

    def test_initialization_freezes_code_model_runtime_and_protects_prior_batch(self):
        root = Path(paper.ROOT)
        names = ['v08_paper_forward_v3.py', 'v08_inference_verified.py',
                 'v08_engine_verified.py', 'exploration_features.py']
        source_hashes = {name: paper.digest(root / name) for name in names}
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            freeze = temp / 'freeze'
            (freeze / 'sources').mkdir(parents=True)
            (freeze / 'freeze.json').write_text('{}', encoding='utf-8')
            (freeze / '2026-lgb.txt').write_text('synthetic frozen model', encoding='utf-8')
            sentinel = temp / 'prior-sealed-record.json'
            sentinel.write_text('original prior batch', encoding='utf-8')
            session = temp / 'parallel-session'
            with patch.object(paper, 'FREEZE', freeze), \
                 patch.object(legacy, 'verify_freeze', return_value={'source_hashes': source_hashes}), \
                 patch.object(paper, 'protected_hashes', return_value={str(sentinel): paper.digest(sentinel)}), \
                 patch.object(paper, 'runtime_versions', return_value={'synthetic': '1'}):
                paper.initialize(session)
                paper.verify_records(session)
                (session / 'runtime/2026-lgb.txt').write_text('changed model', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Frozen runtime source/model content changed'):
                    paper.verify_records(session)
                (session / 'runtime/2026-lgb.txt').write_text('synthetic frozen model', encoding='utf-8')
                sentinel.write_text('modified old sealed record', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Protected original artifact changed'):
                    paper.verify_records(session)

    def test_sealed_signal_hash_and_record_chain_reject_later_edits(self):
        root = Path(paper.ROOT)
        names = ['v08_paper_forward_v3.py', 'v08_inference_verified.py',
                 'v08_engine_verified.py', 'exploration_features.py']
        source_hashes = {name: paper.digest(root / name) for name in names}
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            freeze = temp / 'freeze'
            (freeze / 'sources').mkdir(parents=True)
            (freeze / 'freeze.json').write_text('{}', encoding='utf-8')
            (freeze / '2026-lgb.txt').write_text('synthetic frozen model', encoding='utf-8')
            sentinel = temp / 'prior-sealed-record.json'
            sentinel.write_text('original prior batch', encoding='utf-8')
            session = temp / 'parallel-session'
            with patch.object(paper, 'FREEZE', freeze), \
                 patch.object(legacy, 'verify_freeze', return_value={'source_hashes': source_hashes}), \
                 patch.object(paper, 'protected_hashes', return_value={str(sentinel): paper.digest(sentinel)}), \
                 patch.object(paper, 'runtime_versions', return_value={'synthetic': '1'}):
                paper.initialize(session)
                init = paper.read_json(session / 'init.json')
                day = init['prospective_start_not_before']
                following = '2026-10-23'
                folder = session / 'days' / day
                (folder / 'input-snapshot').mkdir(parents=True)
                (folder / 'input-snapshot/exchange-calendar.csv').write_text(
                    'calendar_date,is_trading_day\n2026-10-21,1\n2026-10-22,1\n2026-10-23,1\n', encoding='utf-8')
                (folder / 'signals.csv').write_text('account,code,score\nv08_verified,sh.600006,1.0\n', encoding='utf-8')
                record = {'signal_date': day, 'execution_date': following,
                    'sealed_beijing': '2026-10-22T16:00:00+08:00',
                    'previous_record_sha256': paper.digest(session / 'init.json'),
                    'account_ids': list(init['accounts']), 'prospective_record': True}
                record['files'] = paper.file_hashes(folder)
                paper.write_new(folder / 'record.json', record)
                self.assertEqual(len(paper.verify_records(session)[1]), 1)
                (folder / 'signals.csv').write_text('tampered', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'artifact changed'):
                    paper.verify_records(session)
                (folder / 'signals.csv').write_text('account,code,score\nv08_verified,sh.600006,1.0\n', encoding='utf-8')
                record['previous_record_sha256'] = '0' * 64
                (folder / 'record.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Broken date or record hash chain'):
                    paper.verify_records(session)


if __name__ == '__main__':
    unittest.main()
