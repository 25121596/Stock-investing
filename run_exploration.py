"""New mechanism studies: breadth gating and weekly risk-aware stock ranking."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from accessibility import AccessibleReplay
from engine import Config
from exploration_features import breadth_days, extend_features, friday_mask
from run import load
from run_midterm import enrich
from validate_expanded import prepare_bars, save_ledger

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / 'data/forward-2026-10-13'
DONOR = ROOT / 'results/midterm-access-v3'
PERIODS = (('2022-2025', '2022-01-01', '2025-12-31'), ('2026', '2026-01-01', '2026-10-13'))
SEEDS = (2026101401, 2026101402, 2026101403)
SOURCES = ('run_exploration.py', 'exploration_features.py', 'exploration_report.py', 'exploration-plan-v1.md',
           'engine.py', 'engine_2026.py', 'accessibility.py', 'accessibility_sources.json', 'run_midterm.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def ranked(rows, score):
    result = rows[['date', 'code']].copy()
    result['score'] = np.asarray(score, dtype=float)
    if result.duplicated(['date', 'code']).any() or not np.isfinite(result.score).all():
        raise ValueError('Invalid signals')
    return result


def stable_random(panel, seed):
    result = panel[['date', 'code']].copy()
    result['score'] = [int.from_bytes(hashlib.blake2b(f'{seed}|{d}|{c}'.encode(), digest_size=8).digest(), 'big') / 2**64
                       for d, c in zip(result.date, result.code)]
    return result


def matched_candidates(policy_signals, random_pool):
    counts = policy_signals.groupby('date').size().rename('desired_count')
    rows = random_pool.merge(counts, left_on='date', right_index=True, how='inner', validate='many_to_one')
    if rows.empty:
        return random_pool.iloc[:0].copy()
    rows = rows.sort_values(['date', 'score', 'code'], ascending=[True, False, True])
    rows = rows[rows.groupby('date').cumcount() < rows.desired_count]
    actual = rows.groupby('date').size().sort_index()
    if not actual.equals(counts.sort_index()):
        raise AssertionError('Random policy candidate counts do not match daily opportunities')
    return rows[['date', 'code', 'score']].sort_values(['date', 'code']).reset_index(drop=True)


def policies(panel, predictions):
    if panel.duplicated(['date', 'code']).any() or predictions.duplicated(['date', 'code']).any():
        raise ValueError('Duplicate panel/predictions')
    coverage = predictions[['date', 'code']].merge(panel[['date', 'code']], how='left', indicator=True)
    if not coverage._merge.eq('both').all():
        raise ValueError('Prediction candidates missing from source panel')
    panel = panel[panel.date.ge(predictions.date.min()) & panel.date.le(predictions.date.max())].copy()
    breadth = breadth_days(panel)
    allowed_days = set(breadth.loc[breadth.allow_entry, 'date'])
    chosen = predictions[predictions.pred_net_return >= .005]
    base = ranked(chosen, chosen.pred_net_return)
    gated = base[base.date.isin(allowed_days)].copy()
    weekly = panel[friday_mask(panel.date) & panel.ret_vol60.gt(0) & panel.adj_ret60.notna()].copy()
    trend = weekly[weekly.adj_ret60.gt(0) & weekly.close_ma60.gt(0)].copy()
    momentum = weekly[weekly.adj_ret60.gt(0)].copy()
    return [
        ('base_lgb10', '原10日选择性收益模型', 10, base),
        ('breadth_lgb10', '市场广度≥50%才新增买入', 10, gated),
        ('weekly_lowvol', '周五选过去60日低波动', 20, ranked(weekly, -weekly.ret_vol60)),
        ('weekly_lowvol_trend', '周五低波动且60日趋势为正', 20, ranked(trend, -trend.ret_vol60)),
        ('weekly_risk_momentum', '周五按60日风险调整强弱排序', 20,
         ranked(momentum, momentum.adj_ret60 / (momentum.ret_vol60 * np.sqrt(60.)))),
    ], breadth


def simulate(frames, bars, days, signals, dividends, config):
    replay = AccessibleReplay({}, days, signals, dividends, config)
    replay.frames = frames
    replay.bars = bars
    return replay.run()


def assert_baseline(expected, actual):
    fields = ('total_return', 'max_drawdown', 'execution_costs', 'closed_trades', 'win_rate',
              'open_positions', 'quality_warning_count', 'final_equity', 'rejected', 'config')
    for field in fields:
        a, b = expected[field], actual[field]
        if isinstance(a, float):
            if abs(a - b) > 1e-9:
                raise AssertionError(f'Baseline differs: {field}: {a} vs {b}')
        elif a != b:
            raise AssertionError(f'Baseline differs: {field}: {a} vs {b}')
    return {'passed': True, 'fields': list(fields)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'results/exploration-v1')
    parser.add_argument('--dedupe-events', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Choose a new empty output directory; keep previous results')
    from exploration_report import make_report
    manifest, frames, calendar, dividends = load(DATASET)
    raw_dividends = dividends
    event_audit = None
    if args.dedupe_events:
        from corporate_events import canonicalize_events
        dividends, event_audit = canonicalize_events(raw_dividends)
    previous = json.loads((DONOR / 'summary.json').read_text(encoding='utf-8'))
    if len(frames) != 300 or calendar[-1] != '2026-10-13' or not previous['metadata']['restricted_small_account_access']:
        raise ValueError('Unexpected data/permission scope')
    if previous['metadata']['manifest_sha256'] != digest(DATASET / 'manifest.json'):
        raise ValueError('Source data differ from frozen baseline')
    for name in ('engine.py', 'engine_2026.py', 'accessibility.py', 'accessibility_sources.json'):
        if previous['hashes'][name] != digest(ROOT / name):
            raise ValueError(f'Execution/permission input changed: {name}')
    output.mkdir(parents=True)
    source_names = SOURCES + (('corporate_events.py', 'exploration-events-fix.md') if args.dedupe_events else ())
    hashes = {name: digest(ROOT / name) for name in source_names}
    (output / 'sources').mkdir()
    for name in source_names:
        shutil.copyfile(ROOT / name, output / 'sources' / name)
    shutil.copyfile(ROOT / 'exploration-plan-v1.md', output / 'protocol.md')
    write_json(output / 'started.json', {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'hashes': hashes,
        'donor_summary_sha256': digest(DONOR / 'summary.json'), 'data_manifest_sha256': digest(DATASET / 'manifest.json')})
    if event_audit is not None:
        write_json(output / 'event-normalization.json', event_audit)
    panel = pd.read_csv(DONOR / 'features.csv', dtype={'date': str, 'code': str}, float_precision='round_trip')
    predictions = pd.read_csv(DONOR / 'predictions-h10.csv', dtype={'date': str, 'code': str}, float_precision='round_trip')
    extended = extend_features(panel, frames)
    if not extended[list(panel.columns)].equals(panel):
        raise AssertionError('Feature extension changed original candidate values')
    panel = extended
    selected, breadth = policies(panel, predictions)
    breadth.to_csv(output / 'breadth.csv', index=False, encoding='utf-8-sig')
    panel[['date', 'code', 'adj_ret60', 'ret_vol60']].to_csv(output / 'new-features.csv', index=False, encoding='utf-8-sig')
    available = panel[panel.date.ge('2022-01-01') & panel.date.le('2026-10-13')]
    random_pools = {seed: stable_random(available, seed) for seed in SEEDS}
    configs, cases, match_checks = {}, [], []
    for name, label, hold, signals in selected:
        configs[name] = Config(hold_days=hold)
        cases.append((name, name, label, 'policy', signals, None))
        cases.append((f'stress_{name}', name, f'{label}·滑点加倍', 'stress', signals, None))
        for seed in SEEDS:
            random_signals = matched_candidates(signals, random_pools[seed])
            cases.append((f'matched_{name}_{seed}', name, f'{label}·同日同候选数随机{seed}', 'matched_random', random_signals, seed))
            match_checks.append({'policy': name, 'seed': seed, 'signal_dates': int(signals.date.nunique()),
                'candidate_rows': len(signals), 'random_candidate_rows': len(random_signals), 'daily_counts_match': True})
    configs['cash'] = Config(hold_days=10)
    cases.append(('cash', 'cash', '不交易现金（不计利息）', 'cash', ranked(panel.iloc[:0], []), None))
    if len(cases) != 26:
        raise AssertionError('Wrong finite comparison count')
    bars = prepare_bars(frames)
    results, replication = [], []
    for period, start, end in PERIODS:
        days = [d for d in calendar if start <= d <= end]
        for name, policy, label, group, signals, seed in cases:
            config = replace(configs[policy])
            if group == 'stress':
                config.slippage *= 2
            chosen = signals[signals.date.ge(start) & signals.date.le(end)]
            case = f'{period}_{name}'
            replay = simulate(frames, bars, days, chosen, dividends, config)
            result = enrich(replay, chosen, config.hold_days, period, case, label, group, days)
            result.update(policy=policy, seed=seed, signal_candidates=len(chosen))
            save_ledger(output / case, replay)
            if policy == 'base_lgb10' and group == 'policy':
                expected = next(r for r in previous['results'] if r['case'] == f'h10_{period}_lgb_selective')
                unnormalized = simulate(frames, bars, days, chosen, raw_dividends, replace(config)).summary() if args.dedupe_events else result
                replication.append({'period': period, **assert_baseline(expected, unnormalized),
                    'normalized_baseline_return_change': result['total_return'] - expected['total_return']})
            if group == 'policy':
                result['yearly'] = []
                for year in range(int(start[:4]), int(end[:4]) + 1):
                    year_days = [d for d in days if d.startswith(str(year))]
                    year_signals = chosen[chosen.date.str.startswith(str(year))]
                    fresh = simulate(frames, bars, year_days, year_signals, dividends, replace(config))
                    result['yearly'].append({'year': year, **fresh.summary()})
                    save_ledger(output / 'yearly' / f'{year}_{name}', fresh)
            results.append(result)
            print(f'CASE {case}: net={result["total_return"]:.2%}, DD={result["max_drawdown"]:.2%}, '
                  f'trades={result["closed_trades"]}, warnings={result["quality_warning_count"]}', flush=True)
    if len(results) != 52 or hashes != {name: digest(ROOT / name) for name in source_names}:
        raise AssertionError('Incomplete grid or changed sources')
    metadata = {'dataset': str(DATASET), 'data_manifest_sha256': digest(DATASET / 'manifest.json'),
        'source_features_sha256': digest(DONOR / 'features.csv'),
        'source_predictions_sha256': digest(DONOR / 'predictions-h10.csv'), 'candidate_rows': len(panel),
        'restricted_small_account_access': True, 'opportunity_matches': match_checks,
        'corporate_events_normalized': args.dedupe_events,
        'event_rows_removed': event_audit['removed_rows'] if event_audit else 0,
        'breadth_observed_days': len(breadth), 'breadth_gate_days': int(breadth.allow_entry.sum()),
        'added_feature_nonfinite_rows': int((~np.isfinite(panel[['adj_ret60', 'ret_vol60']].to_numpy())).any(axis=1).sum()),
        'interpretation': 'Previously viewed historical years; fixed mechanism study, not new independent test'}
    payload = {'version': 'exploration-v2' if args.dedupe_events else 'exploration-v1', 'protocol': {'main_policy_count': 5, 'cases': 52,
        'random_seeds': list(SEEDS), 'cost_pressure': 'slippage doubled, other fees unchanged',
        'random_match': 'same signal dates and candidate counts; actual fills and exposure may differ'},
        'metadata': metadata, 'results': results, 'hashes': hashes, 'replication': replication}
    write_json(output / 'summary.json', payload)
    columns = ['case', 'period', 'policy', 'group', 'hold_days', 'total_return', 'max_drawdown', 'win_rate',
        'closed_trades', 'execution_costs', 'avg_exposure', 'median_holding_days', 'quality_warning_count', 'open_positions']
    pd.DataFrame(results)[columns].to_csv(output / 'comparison.csv', index=False, encoding='utf-8-sig')
    report = make_report(output, payload)
    (ROOT / 'results/latest.txt').write_text(str(report), encoding='utf-8')
    print(f'REPORT {report}', flush=True)


if __name__ == '__main__':
    main()
