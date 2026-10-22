"""Finite 5/10/20-session study. Viewed historical years stay exploratory."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from engine import Config
from engine_2026 import Replay2026
from ml_features import FEATURES, build_panel
from run import load
from run_ml import training_mask, experiment_signals, rank_signals
from validate_expanded import prepare_bars, save_ledger

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / 'data/forward-2026-10-13'
HOLDS = (5, 10, 20)
YEARS = (2022, 2023, 2024, 2025, 2026)
PERIODS = (('2022-2025', '2022-01-01', '2025-12-31'),
           ('2026', '2026-01-01', '2026-10-13'))
SOURCES = ('run_midterm.py', 'midterm_labels.py', 'midterm_report.py', 'midterm-plan-v1.md',
           'engine.py', 'engine_2026.py', 'ml_features.py', 'run_ml.py')
REPLAY_CLASS = Replay2026


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def simulate(frames, bars, days, signals, dividends, config):
    replay = REPLAY_CLASS({}, days, signals, dividends, config)
    replay.frames = frames
    replay.bars = {code: bars[code] for code in frames}
    return replay.run()


def audit_labels(panel, frames, bars, calendar, dividends, config):
    eligible = panel[panel.net_return.notna()]
    chosen = eligible.sample(n=min(120, len(eligible)), random_state=20261015 + config.hold_days)
    index = {day: i for i, day in enumerate(calendar)}
    durations = eligible.label_end.map(index) - eligible.date.map(index) - 1
    full = eligible[durations >= config.hold_days]
    chosen = pd.concat([chosen, full.sample(n=min(20, len(full)), random_state=20261016)]).drop_duplicates(['date', 'code'])
    for row in chosen.itertuples():
        start = index[row.date]
        days = calendar[start:min(len(calendar), start + config.hold_days + 18)]
        signals = pd.DataFrame([{'date': row.date, 'code': row.code, 'score': 1.}])
        replay = simulate({row.code: frames[row.code]}, bars, days, signals, dividends, replace(config))
        if len(replay.trades) != 1 or replay.warnings:
            raise AssertionError(f'Label audit unresolved: {config.hold_days}, {row.date}, {row.code}, {replay.warnings}')
        trade = replay.trades[0]
        if abs(trade['net_return'] - row.net_return) > 1e-10 or trade['exit_date'] != row.label_end:
            raise AssertionError(f'Label audit differs: {row} vs {trade}')
    return {'hold_days': config.hold_days, 'sampled': len(chosen), 'matched': len(chosen),
            'full_horizon_labels_available': len(full), 'scope': 'fixed seeded single-trade sample, not proof of all labels'}


def train_models(panel, holding, output):
    folder = output / 'models' / f'h{holding}'
    folder.mkdir(parents=True)
    forecasts, training = [], []
    for year in YEARS:
        start, end = f'{year}-01-01', min(f'{year}-12-31', '2026-10-13')
        train = panel.loc[training_mask(panel, start)]
        evaluation = panel[panel.date.ge(start) & panel.date.le(end)].copy()
        finite = np.isfinite(evaluation[FEATURES].to_numpy(dtype=float)).all(axis=1)
        evaluation = evaluation.loc[finite].copy()
        if len(train) < 1000 or evaluation.empty or train.y_win.nunique() != 2:
            raise ValueError(f'Insufficient h{holding}/{year} input')
        assert train.label_end.max() < start and train.date.max() < start
        classifier = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=300, random_state=20261014))
        regressor = lgb.LGBMRegressor(n_estimators=150, learning_rate=.03, num_leaves=7,
            max_depth=3, min_child_samples=300, reg_lambda=10, random_state=20261014,
            n_jobs=3, deterministic=True, force_col_wise=True, verbosity=-1)
        with threadpool_limits(limits=3):
            classifier.fit(train[FEATURES], train.y_win.astype(int))
            regressor.fit(train[FEATURES], train.net_return.clip(-.2, .2))
            evaluation['p_win'] = classifier.predict_proba(evaluation[FEATURES])[:, 1]
            evaluation['pred_net_return'] = regressor.predict(evaluation[FEATURES])
        joblib.dump(classifier, folder / f'{year}-logistic.joblib')
        regressor.booster_.save_model(str(folder / f'{year}-lgb.txt'))
        labeled = evaluation[evaluation.net_return.notna()]
        observed = labeled.y_win.astype(int)
        rank_ic = labeled.pred_net_return.corr(labeled.net_return, method='spearman')
        stats = {'hold_days': holding, 'year': year, 'train_rows': len(train),
            'train_label_end_max': train.label_end.max(), 'train_date_max': train.date.max(),
            'train_win_rate': float(train.y_win.mean()), 'eval_rows': len(evaluation),
            'eval_labeled_rows': len(labeled), 'nonfinite_features_excluded': int((~finite).sum()),
            'logistic_auc': float(roc_auc_score(observed, labeled.p_win)) if observed.nunique() == 2 else None,
            'logistic_brier': float(brier_score_loss(observed, labeled.p_win)),
            'lgb_mse': float(mean_squared_error(labeled.net_return, labeled.pred_net_return)),
            'lgb_rank_ic': float(rank_ic) if np.isfinite(rank_ic) else None}
        training.append(stats)
        forecasts.append(evaluation[['date', 'code', 'p_win', 'pred_net_return', 'adj_ret5', 'adj_ret20']])
        print(f'TRAIN h{holding}/{year}: {len(train)} train, {len(evaluation)} predict, AUC={stats["logistic_auc"]:.4f}', flush=True)
    predictions = pd.concat(forecasts, ignore_index=True).sort_values(['date', 'code']).reset_index(drop=True)
    predictions.to_csv(output / f'predictions-h{holding}.csv', index=False, encoding='utf-8-sig')
    return predictions, training


def enrich(replay, signals, holding, period, case, label, group, days):
    summary = replay.summary()
    summary['continuous_yearly'] = summary.pop('yearly')
    equity = pd.DataFrame(replay.equity)
    index = {day: i for i, day in enumerate(days)}
    held = [index[t['exit_date']] - index[t['entry_date']] for t in replay.trades]
    natural = [(pd.Timestamp(t['exit_date']) - pd.Timestamp(t['entry_date'])).days for t in replay.trades]
    turnover = sum(float(fill['notional']) for fill in replay.fills)
    closed_ids = {t['trade_id'] for t in replay.trades}
    closed_costs = sum(sum(float(f[key]) for key in ('commission', 'transfer', 'stamp', 'slippage'))
                       for f in replay.fills if f['trade_id'] in closed_ids)
    return {**summary, 'case': case, 'label': label, 'group': group, 'hold_days': holding, 'period': period,
        'signal_dates': int(signals.date.nunique()), 'buy_fills': sum(f['side'] == 'buy' for f in replay.fills),
        'avg_exposure': float(((equity.equity - equity.cash - equity.receivables) / equity.equity).mean()),
        'mean_holding_days': float(np.mean(held)) if held else None,
        'median_holding_days': float(np.median(held)) if held else None,
        'mean_holding_calendar_days': float(np.mean(natural)) if natural else None,
        'held_longer_than_plan': sum(n > holding for n in held),
        'exit_reason_counts': dict(Counter(t['exit_reason'] for t in replay.trades)),
        'turnover_notional': turnover, 'two_way_turnover_to_capital': turnover / replay.config.capital,
        'cost_per_closed_trade': closed_costs / len(held) if held else None,
        'cost_per_traded_notional': summary['execution_costs'] / turnover if turnover else None,
        'quality_warnings': replay.warnings}


def replay_cases(predictions, legacy, frames, bars, calendar, dividends, holding, output):
    cases = experiment_signals(predictions, legacy)
    for name in ('logistic_selective', 'lgb_selective'):
        _case, label, _group, signals = next(c for c in cases if c[0] == name)
        cases.append((f'stress_{name}', f'{label}·滑点加倍', 'stress', signals))
    cases.append(('cash', '不交易现金（不计利息）', 'cash', rank_signals(predictions.iloc[:0], [])))
    assert len(cases) == 28
    results = []
    for period, start, end in PERIODS:
        days = [day for day in calendar if start <= day <= end]
        for name, label, group, all_signals in cases:
            signals = all_signals[all_signals.date.ge(start) & all_signals.date.le(end)]
            config = Config(hold_days=holding)
            if group == 'stress':
                config.slippage *= 2
            case = f'h{holding}_{period}_{name}'
            replay = simulate(frames, bars, days, signals, dividends, config)
            result = enrich(replay, signals, holding, period, case, label, group, days)
            save_ledger(output / case, replay)
            if group == 'model':
                result['yearly'] = []
                for year in YEARS:
                    if not start[:4] <= str(year) <= end[:4]:
                        continue
                    year_days = [d for d in days if d.startswith(str(year))]
                    year_signals = signals[signals.date.str.startswith(str(year))]
                    fresh = simulate(frames, bars, year_days, year_signals, dividends, replace(config))
                    result['yearly'].append({'year': year, **fresh.summary()})
                    save_ledger(output / 'yearly' / f'h{holding}_{year}_{name}', fresh)
            results.append(result)
            print(f'CASE {case}: net={result["total_return"]:.2%}, trades={result["closed_trades"]}, '
                  f'actualhold={result["median_holding_days"]}, warnings={result["quality_warning_count"]}', flush=True)
    return results


def main():
    global REPLAY_CLASS
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'results/midterm-v1')
    parser.add_argument('--features-from', type=Path)
    parser.add_argument('--restrict-access', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Choose a new empty output folder; previous results are preserved')
    from midterm_labels import relabel_panel
    from midterm_report import make_report
    manifest, frames, calendar, dividends = load(DATASET)
    if len(frames) != 300 or manifest['failures'] or calendar[-1] != '2026-10-13':
        raise ValueError('Need the complete fixed 300-stock source through 2026-10-13')
    output.mkdir(parents=True, exist_ok=True)
    source_names = SOURCES + (('accessibility.py', 'accessibility_sources.json', 'midterm-access-fix.md')
                              if args.restrict_access else ())
    hashes = {name: digest(ROOT / name) for name in source_names}
    source_folder = output / 'sources'
    source_folder.mkdir()
    for name in source_names:
        shutil.copyfile(ROOT / name, source_folder / name)
    shutil.copyfile(ROOT / 'midterm-plan-v1.md', output / 'protocol.md')
    write_json(output / 'started.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
        'hashes': hashes, 'manifest_sha256': digest(DATASET / 'manifest.json'),
        'interpretation': 'All evaluated years previously viewed; exploratory only'})
    if args.features_from:
        cached_path = args.features_from.resolve()
        source_start = json.loads((cached_path.parent / 'started.json').read_text(encoding='utf-8'))
        if source_start['manifest_sha256'] != digest(DATASET / 'manifest.json'):
            raise ValueError('Feature donor source data differ')
        donor_summary = json.loads((cached_path.parent / 'summary.json').read_text(encoding='utf-8'))
        if donor_summary['hashes']['ml_features.py'] != digest(ROOT / 'ml_features.py'):
            raise ValueError('Feature donor builder differs')
        features = pd.read_csv(cached_path, dtype={'date': str, 'code': str}, float_precision='round_trip')
        print(f'REUSE past-only features, SHA256={digest(cached_path)}', flush=True)
    else:
        print('BUILD past-only feature panel; discard all original 3-day labels', flush=True)
        panel = build_panel(frames, calendar, dividends, Config(hold_days=3))
        features = panel[['date', 'code', *FEATURES, 'eligible']].sort_values(['date', 'code']).reset_index(drop=True)
        del panel
    before_access = len(features)
    if args.restrict_access:
        from accessibility import AccessibleReplay, accessible_candidates
        REPLAY_CLASS = AccessibleReplay
        features = accessible_candidates(features)
        print(f'ACCESSIBILITY: removed {before_access - len(features)} known retirement-period candidates; earlier history retained', flush=True)
    if features.duplicated(['date', 'code']).any() or not features.eligible.all():
        raise ValueError('Invalid feature donor/candidate panel')
    features.to_csv(output / 'features.csv', index=False, encoding='utf-8-sig')
    bars = prepare_bars(frames)
    old_legacy = pd.read_csv(ROOT / 'data/pilot-2020-01-02-2025-12-31-n300-s20201009/signals.csv', dtype={'date': str, 'code': str})
    new_legacy = pd.read_csv(DATASET / '.legacy-2026/signals.csv', dtype={'date': str, 'code': str})
    legacy = pd.concat([old_legacy, new_legacy], ignore_index=True)
    if legacy.duplicated(['date', 'code', 'strategy']).any():
        raise ValueError('Overlapping legacy signal cache')
    metadata = {'dataset': str(DATASET), 'manifest_sha256': digest(DATASET / 'manifest.json'),
        'codes': len(frames), 'calendar_start': calendar[0], 'calendar_end': calendar[-1],
        'raw_bar_count': sum(len(f) for f in frames.values()), 'candidate_rows': len(features),
        'restricted_small_account_access': args.restrict_access,
        'removed_retirement_period_rows': before_access - len(features),
        'feature_donor_sha256': digest(args.features_from) if args.features_from else None,
        'features': FEATURES, 'versions': {'sklearn': sklearn.__version__, 'lightgbm': lgb.__version__},
        'legacy_signal_sha256': [digest(ROOT / 'data/pilot-2020-01-02-2025-12-31-n300-s20201009/signals.csv'),
                                  digest(DATASET / '.legacy-2026/signals.csv')], 'label_coverage': []}
    results, training, audits = [], [], []
    for holding in HOLDS:
        config = Config(hold_days=holding)
        print(f'LABEL h{holding}', flush=True)
        labels = relabel_panel(features, frames, calendar, dividends, config, replay_class=REPLAY_CLASS)
        if not labels[['date', 'code', *FEATURES, 'eligible']].equals(features):
            raise AssertionError('Labeling changed the inference candidate set/features')
        labels.to_csv(output / f'panel-h{holding}.csv', index=False, encoding='utf-8-sig')
        reasons = pd.crosstab(labels.date.str[:4], labels.label_reason)
        coverage = {'hold_days': holding, 'rows': len(labels), 'labeled_rows': int(labels.net_return.notna().sum()),
            'reasons': {str(k): int(v) for k, v in labels.label_reason.value_counts().items()},
            'reasons_by_year': {str(y): {str(k): int(v) for k, v in row.items()} for y, row in reasons.iterrows()}}
        metadata['label_coverage'].append(coverage)
        audit = audit_labels(labels, frames, bars, calendar, dividends, config)
        audits.append(audit)
        print(f'LABEL AUDIT h{holding}: {audit["matched"]}/{audit["sampled"]}', flush=True)
        predictions, records = train_models(labels, holding, output)
        training.extend(records)
        results.extend(replay_cases(predictions, legacy, frames, bars, calendar, dividends, holding, output))
        del labels, predictions
    if len(results) != 168:
        raise AssertionError('Incomplete finite scenario grid')
    if hashes != {name: digest(ROOT / name) for name in source_names}:
        raise AssertionError('Sources changed during the experiment')
    payload = {'version': 'midterm-access-v3' if args.restrict_access else 'midterm-v1', 'protocol': {'hold_days': list(HOLDS), 'base_config': asdict(Config()),
        'scope': '2022-2025 continuous and 2026 fresh account; all previously viewed exploratory years',
        'cases_per_hold_per_period': 28, 'total_cases': 168,
        'thresholds': {'p_win': .55, 'pred_net_return': .005}, 'no_live_orders': True},
        'metadata': metadata, 'training': training, 'label_audits': audits, 'results': results, 'hashes': hashes}
    write_json(output / 'summary.json', payload)
    columns = ['case', 'period', 'hold_days', 'group', 'label', 'total_return', 'max_drawdown', 'win_rate',
        'profit_factor', 'closed_trades', 'execution_costs', 'avg_exposure', 'quality_warning_count', 'open_positions',
        'mean_holding_days', 'median_holding_days', 'two_way_turnover_to_capital', 'cost_per_traded_notional']
    pd.DataFrame(results)[columns].to_csv(output / 'comparison.csv', index=False, encoding='utf-8-sig')
    report = make_report(output, payload)
    (ROOT / 'results/latest.txt').write_text(str(report), encoding='utf-8')
    print(f'REPORT {report}', flush=True)


if __name__ == '__main__':
    main()
