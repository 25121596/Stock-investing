"""Read-only integrity and replay verifier for the V0.8.1 parallel paper batch."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fail(message):
    raise ValueError(message)


def same_frame(left, right, keys, label):
    left, right = left.copy(), right.copy()
    if set(left.columns) != set(right.columns):
        fail(f'{label}: columns differ')
    cols = sorted(left.columns)
    left, right = left[cols], right[cols]
    for col in keys:
        left[col] = left[col].astype(str)
        right[col] = right[col].astype(str)
    left = left.sort_values(keys, kind='stable').reset_index(drop=True)
    right = right.sort_values(keys, kind='stable').reset_index(drop=True)
    if len(left) != len(right):
        fail(f'{label}: row count differs')
    for col in cols:
        if pd.api.types.is_numeric_dtype(left[col]) and pd.api.types.is_numeric_dtype(right[col]):
            a = pd.to_numeric(left[col], errors='coerce').to_numpy(dtype=float)
            b = pd.to_numeric(right[col], errors='coerce').to_numpy(dtype=float)
            if not np.array_equal(np.isnan(a), np.isnan(b)) or not np.allclose(a, b, rtol=0, atol=1e-12, equal_nan=True):
                fail(f'{label}: numeric column differs: {col}')
        elif not left[col].fillna('').astype(str).equals(right[col].fillna('').astype(str)):
            fail(f'{label}: column differs: {col}')


def verify(session):
    session = Path(session).resolve()
    runner_path = ROOT / 'v081_parallel_paper.py'
    spec = importlib.util.spec_from_file_location('v081_parallel_paper_under_audit', runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    init, records, _ = runner.verify_records(session)
    if len(init['accounts']) != 9:
        fail('Expected nine independent accounts')
    expected_ids = list(runner.accounts())
    if list(init['accounts']) != expected_ids:
        fail('Account ids/order differ from the frozen V0.8/V0.8.1 design')
    if init.get('current_adopted') != 'V0.8' or init.get('independent_batch') is not True:
        fail('Adoption or independent-batch status is invalid')
    if not init.get('no_real_brokerage_orders'):
        fail('Batch does not declare the no-broker boundary')
    old_session = ROOT / 'results/v08-paper-forward-v3'
    old_init = read_json(old_session / 'init.json')
    old_runtime = old_session / 'runtime'
    if digest(old_runtime / 'freeze.json') != old_init['freeze_sha256']:
        fail('Original V0.8 runtime freeze changed')
    runner.legacy.verify_freeze(old_runtime)

    # verify_records protects source/runtime and the chain; this pass also
    # confirms human-facing zero-record pages cannot imply achieved returns.
    status_text = (session / 'status.md').read_text(encoding='utf-8')
    status_html = (session / 'status.html').read_text(encoding='utf-8')
    if not records:
        if '没有未来预测、成交或可评估业绩' not in status_text or '无业绩' not in status_html:
            fail('Zero-record status overstates prospective evidence')
        if any((session / 'days').iterdir()):
            fail('Zero-record session has day artifacts')
        return {'session': str(session), 'sealed_days': 0, 'accounts': 9,
                'claim': 'Initialized only; no prospective signal, fill, or evaluable performance.'}

    modules = runner.load_runtime(session)
    reconstructed_signals = {name: [] for name in expected_ids}
    prior_panels = []
    checked_accounts = 0
    for folder, record in records:
        snapshot = folder / 'input-snapshot'
        manifest, frames, calendar, raw_events, _schedule = runner.load_dataset(snapshot, record['signal_date'])
        events, audit = modules[2].canonicalize_events(raw_events)
        if audit.get('unresolved_incomplete_groups'):
            fail(f'{record["signal_date"]}: unresolved corporate action identity')
        panel, breadth, signals = runner.infer(frames, calendar, record['signal_date'], session / 'runtime', modules)
        saved_panel = pd.read_csv(folder / 'scores-and-features.csv', dtype={'date': str, 'code': str}, float_precision='round_trip')
        same_frame(saved_panel, panel, ['date', 'code'], f'{record["signal_date"]} inference panel')
        saved_breadth = pd.read_csv(folder / 'breadth.csv', float_precision='round_trip')
        same_frame(saved_breadth, breadth, ['date'], f'{record["signal_date"]} breadth')
        saved_signals = pd.read_csv(folder / 'signals.csv', dtype={'date': str, 'code': str}, float_precision='round_trip')
        for name in expected_ids:
            observed = saved_signals[saved_signals.account.eq(name)][['date', 'code', 'score']].reset_index(drop=True)
            same_frame(observed, signals[name], ['date', 'code'], f'{record["signal_date"]}/{name} signals')
            reconstructed_signals[name].append(signals[name])
        # The three random controls in each policy arm are paired by identical
        # frozen seed and candidate count; verify both pair identity and count.
        for seed in runner.SEEDS:
            left = signals[f'v08_verified_random_{seed}']
            right = signals[f'v081_breadth58_random_{seed}']
            same_frame(left, right, ['date', 'code'], f'{record["signal_date"]}/paired random {seed}')

        prior_panels.append(panel)
        context = pd.concat(prior_panels, ignore_index=True)
        days = [d for d in calendar if records[0][1]['signal_date'] <= d <= record['signal_date']]
        bars = {code: frame.set_index('date').to_dict('index') for code, frame in frames.items()}
        for name, account_spec in init['accounts'].items():
            joined = pd.concat(reconstructed_signals[name], ignore_index=True)
            replay = runner.simulate(frames, bars, days, joined, events, account_spec, modules,
                                     context[context.date.le(record['signal_date'])])
            if replay.warnings or replay.review_warnings:
                fail(f'{record["signal_date"]}/{name}: replay has quality warnings')
            sealed = read_json(folder / name / 'account.json')
            if runner.object_hash(runner.account_state(replay)) != runner.object_hash(sealed['state']):
                fail(f'{record["signal_date"]}/{name}: account state replay differs')
            if runner.object_hash(runner.ledger(replay)) != runner.object_hash(sealed['ledger']):
                fail(f'{record["signal_date"]}/{name}: ledger replay differs')
            if sealed.get('account') != name or sealed.get('spec') != account_spec:
                fail(f'{record["signal_date"]}/{name}: account identity/config mismatch')
            gate = pd.read_csv(folder / name / 'gate-decisions.csv', float_precision='round_trip')
            if len(gate) != len(days):
                fail(f'{record["signal_date"]}/{name}: missing daily gate decisions')
            if account_spec['gate_breadth58']:
                for row in gate.itertuples(index=False):
                    breadth_value = getattr(row, 'breadth')
                    proposed = ast.literal_eval(getattr(row, 'proposed_orders')) if isinstance(getattr(row, 'proposed_orders'), str) else getattr(row, 'proposed_orders')
                    accepted = ast.literal_eval(getattr(row, 'accepted_orders')) if isinstance(getattr(row, 'accepted_orders'), str) else getattr(row, 'accepted_orders')
                    expected_cancel = bool(proposed and pd.notna(breadth_value) and breadth_value < runner.THRESHOLD)
                    if expected_cancel != bool(getattr(row, 'action')) or len(proposed) - len(accepted) != int(getattr(row, 'cancelled_orders')):
                        fail(f'{record["signal_date"]}/{name}: 58% gate decision inconsistent')
            checked_accounts += 1

        comparison = pd.read_csv(folder / 'account-comparison.csv')
        if set(comparison.account) != set(expected_ids):
            fail(f'{record["signal_date"]}: account comparison incomplete')
    return {'session': str(session), 'sealed_days': len(records), 'accounts': 9,
            'replayed_account_states': checked_accounts,
            'claim': 'Sealed prospective records verified; performance still requires actual simulated fills.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, default=ROOT / 'results/v081-parallel-paper-20261022-v1')
    args = parser.parse_args()
    result = verify(args.session)
    print(json.dumps({'ok': True, **result}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError, AssertionError) as exc:
        raise SystemExit(f'BLOCKED: {exc}')
