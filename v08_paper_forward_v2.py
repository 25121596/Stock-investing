"""Manual, append-only prospective paper records. Never sends brokerage orders."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FREEZE = ROOT / 'results/v08-codex-frozen-v2'
SESSION = ROOT / 'results/v08-paper-forward-v2'
CST = timezone(timedelta(hours=8))
SEEDS = (2026101401, 2026101402, 2026101403)
MAIN = {'v08_verified': 10}
# The old freeze's source_features_sha256 identifies features.csv, not Python.
# Pin the code hash verified against the original midterm training snapshot.
FEATURE_CODE_SHA256 = 'be3bbab3adf49dfd61a091b661749af3781c22890e5e36f3d48a39bb02ac9831'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def runtime_versions():
    import lightgbm
    return {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__, 'lightgbm': lightgbm.__version__}


def normal(value):
    if isinstance(value, dict):
        return {str(k): normal(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normal(v) for v in value]
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            raise ValueError('Nonfinite account value')
        return round(float(value), 8)
    if isinstance(value, np.integer):
        return int(value)
    return value


def object_hash(value):
    return hashlib.sha256(json.dumps(normal(value), sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def verify_freeze(folder):
    frozen = read_json(folder / 'freeze.json')
    if digest(folder / '2026-lgb.txt') != frozen['model_sha256']:
        raise ValueError('Frozen model changed')
    for name, expected in frozen['source_hashes'].items():
        if digest(folder / 'sources' / name) != expected:
            raise ValueError(f'Frozen source changed: {name}')
    if digest(folder / 'sources/ml_features-frozen.py') != FEATURE_CODE_SHA256:
        raise ValueError('Frozen feature source changed')
    return frozen


def load_runtime(folder):
    """Use the frozen engines and features, never the mutable working copies."""
    verify_freeze(folder)
    sources = folder / 'sources'
    sys.path.insert(0, str(sources))
    for name in ('engine', 'engine_2026', 'accessibility', 'corporate_events', 'exploration_features'):
        existing = sys.modules.get(name)
        if existing is not None and Path(existing.__file__).resolve().parent != sources.resolve():
            raise ValueError('Run paper_forward.py in a fresh Python process')
    import engine
    import accessibility
    import corporate_events
    import exploration_features
    spec = importlib.util.spec_from_file_location('paper_frozen_features', sources / 'ml_features-frozen.py')
    features = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(features)
    # Inference has no labels. In particular it must never request future bars.
    features._label = lambda *a, **k: (None, None, None, 'prospective_inference_only')
    return engine, accessibility, corporate_events, exploration_features, features


def accounts():
    result = {key: {'policy': key, 'hold_days': hold, 'seed': None} for key, hold in MAIN.items()}
    for key, hold in MAIN.items():
        for seed in SEEDS:
            result[f'{key}_random_{seed}'] = {'policy': key, 'hold_days': hold, 'seed': seed}
    result['cash'] = {'policy': 'cash', 'hold_days': 10, 'seed': None}
    return result


def initialize(session):
    if session.exists():
        raise ValueError('Session already exists; do not reset a running account')
    frozen = verify_freeze(FREEZE)
    session.mkdir(parents=True)
    shutil.copytree(FREEZE, session / 'runtime')
    shutil.copyfile(__file__, session / 'paper_forward-source.py')
    now = datetime.now(timezone.utc)
    initial = {'version': 'v08-paper-forward-v2', 'created_utc': now.isoformat(),
        'created_beijing': now.astimezone(CST).isoformat(), 'accounts': accounts(),
        'initial_cash_per_account': 30000, 'freeze_sha256': digest(session / 'runtime/freeze.json'),
        'runner_sha256': digest(__file__), 'historical_data_last_day': '2026-10-19',
        'inference_feature_source_sha256': FEATURE_CODE_SHA256,
        'model_year': 2026, 'random_seeds': list(SEEDS),
        'runtime_versions': runtime_versions(),
        'record_deadline': '09:15 Asia/Shanghai on the following exchange trading day',
        'no_automatic_runs': True, 'no_broker_connection': True,
        'timestamp_limit': 'Local clock and hash chain are an audit trail, not an external timestamp authority'}
    write_json(session / 'init.json', initial)
    (session / 'days').mkdir()
    write_status(session, initial, [])
    print(f'INITIALIZED {session}: 1 V0.8 policy + 3 random controls + cash; no prediction record yet')


def verify_records(session):
    init = read_json(session / 'init.json')
    if digest(session / 'runtime/freeze.json') != init['freeze_sha256']:
        raise ValueError('Session rule freeze changed')
    if digest(__file__) != init['runner_sha256'] or digest(session / 'paper_forward-source.py') != init['runner_sha256']:
        raise ValueError('Paper recorder source changed; do not silently change a running version')
    if 'runtime_versions' in init and init['runtime_versions'] != runtime_versions():
        raise ValueError('Inference library versions changed; keep the running environment frozen')
    previous = digest(session / 'init.json')
    records = []
    for folder in sorted((session / 'days').iterdir()):
        if not folder.is_dir() or folder.name.startswith('.'):
            continue
        record = read_json(folder / 'record.json')
        if record['previous_record_sha256'] != previous:
            raise ValueError('Broken paper record chain')
        for name, expected in record['files'].items():
            if digest(folder / name) != expected:
                raise ValueError(f'Sealed paper artifact changed: {folder.name}/{name}')
        previous = digest(folder / 'record.json')
        records.append((folder, record))
    return init, records, previous


def validate_window(cutoff, schedule, now, earliest_date, previous=None):
    if now.tzinfo is None:
        raise ValueError('Timezone-aware clock required')
    trading = sorted(set(schedule))
    if cutoff not in trading:
        raise ValueError('Signal day is not an exchange trading day')
    if cutoff < earliest_date or cutoff[:4] != '2026':
        raise ValueError('Date precedes prospective start or exceeds frozen model year')
    complete = [d for d in trading if datetime.combine(date.fromisoformat(d), time(15, 30), CST) <= now]
    if not complete or complete[-1] != cutoff:
        raise ValueError('Stale or incomplete close: dataset must end on the latest completed session')
    future = [d for d in trading if d > cutoff]
    if not future:
        raise ValueError('A verified following trading day is required')
    next_day = future[0]
    if now >= datetime.combine(date.fromisoformat(next_day), time(9, 15), CST):
        raise ValueError('Missed pre-auction deadline; never backfill a prospective prediction')
    if previous is not None and previous['execution_date'] != cutoff:
        raise ValueError('A trading-day record was missed; no retrospective stop orders or signals allowed')
    return next_day


def load_dataset(folder, cutoff):
    manifest = read_json(folder / 'manifest.json')
    if manifest['failures'] or manifest['end'] != cutoff or manifest['requested_count'] != 300:
        raise ValueError('Incomplete data snapshot or changed stock pool')
    expected = pd.read_csv(ROOT / 'data/forward-2026-10-13/universe.csv', dtype=str).code.tolist()
    if [item['code'] for item in manifest['inventory']] != expected:
        raise ValueError('The frozen 300-stock universe changed')
    frames, events = {}, []
    for item in manifest['inventory']:
        code = item['code']
        price, actions = folder / f'{code}.csv', folder / f'{code}.dividends.csv'
        if digest(price) != item['sha256'] or digest(actions) != item['dividends_sha256']:
            raise ValueError(f'Dataset hash mismatch: {code}')
        frame = pd.read_csv(price, dtype={'date': str, 'code': str}, float_precision='round_trip')
        if (frame.empty or frame.date.duplicated().any() or not frame.code.eq(code).all()
                or frame.date.max() > cutoff or not frame.date.is_monotonic_increasing
                or not pd.to_numeric(frame.adjustflag).eq(3).all()
                or not np.isfinite(frame[['open', 'high', 'low', 'close', 'adj_close']]).all().all()
                or not frame.close.gt(0).all() or not frame.adj_close.gt(0).all()):
            raise ValueError(f'Invalid price bars: {code}')
        frames[code] = frame
        events.extend(pd.read_csv(actions, dtype=str).fillna('').to_dict('records'))
    calendar = pd.read_csv(folder / 'calendar.csv', dtype=str).calendar_date.tolist()
    for name, key in [('calendar.csv', 'calendar_sha256'), ('universe.csv', 'universe_sha256')]:
        if manifest.get(key) != digest(folder / name):
            raise ValueError(f'Dataset identity file hash mismatch: {name}')
    if not calendar or calendar != sorted(set(calendar)) or calendar[-1] != cutoff:
        raise ValueError('Invalid or truncated trading calendar')
    exchange = folder / 'exchange-calendar.csv'
    if not exchange.is_file():
        raise ValueError('Missing next-session exchange calendar; run update_paper_data.py first')
    if manifest.get('exchange_calendar_sha256') != digest(exchange):
        raise ValueError('Exchange calendar hash mismatch')
    schedule = pd.read_csv(exchange, dtype=str)
    schedule = schedule[schedule.is_trading_day.eq('1')].calendar_date.tolist()
    return manifest, frames, calendar, events, schedule


def random_matched(pool, count, seed):
    result = pool[['date', 'code']].copy()
    result['score'] = [int.from_bytes(hashlib.blake2b(f'{seed}|{d}|{c}'.encode(), digest_size=8).digest(), 'big') / 2**64
                       for d, c in zip(result.date, result.code)]
    return result.sort_values(['score', 'code'], ascending=[False, True]).head(count).reset_index(drop=True)


def infer(frames, calendar, cutoff, runtime, modules):
    engine, access, _, exploration, feature = modules
    import lightgbm as lgb
    # 21 calendar sessions suffice for the model's market aggregates. Stock
    # histories remain complete for the 250-valid-bar eligibility requirement.
    panel = feature.build_panel(frames, calendar[-21:], [], engine.Config(hold_days=10))
    panel = access.accessible_candidates(panel[panel.date.eq(cutoff)].copy())
    panel = exploration.extend_features(panel, frames)
    model = lgb.Booster(model_file=str(runtime / '2026-lgb.txt'))
    if model.feature_name() != feature.FEATURES:
        raise ValueError('Model feature order mismatch')
    panel['pred_net_return'] = model.predict(panel[feature.FEATURES], num_threads=1) if len(panel) else np.array([])
    if not np.isfinite(panel.pred_net_return).all():
        raise ValueError('Nonfinite model prediction')
    selected = panel[panel.pred_net_return.ge(.005)][['date', 'code', 'pred_net_return']].rename(columns={'pred_net_return': 'score'})
    breadth = exploration.breadth_days(panel)
    allow = bool(len(breadth) and breadth.allow_entry.iloc[0])
    weekly = panel[panel.ret_vol60.gt(0) & panel.adj_ret60.notna()].copy()
    if date.fromisoformat(cutoff).weekday() != 4:
        weekly = weekly.iloc[:0]
    lowvol = weekly[['date', 'code']].copy()
    lowvol['score'] = -weekly.ret_vol60.to_numpy()
    signals = {'base_lgb10': selected, 'breadth_lgb10': selected if allow else selected.iloc[:0], 'weekly_lowvol': lowvol}
    empty = pd.DataFrame(columns=['date', 'code', 'score'])
    for name, spec in accounts().items():
        if spec['seed'] is not None:
            signals[name] = random_matched(panel, len(signals[spec['policy']]), spec['seed'])
    signals['cash'] = empty
    return panel, breadth, signals


def account_state(replay):
    return normal({'cash': replay.cash, 'nav': replay.nav(), 'marks': replay.marks,
        'positions': {code: asdict(p) for code, p in replay.positions.items()},
        'receivables': replay.receivables, 'pending_buys': replay.pending_buys,
        'entitlements': [{'event': c['event'], 'trade_id': c['position'].trade_id,
            'shares': c['shares'], 'posted': c['posted']} for c in replay.entitlements],
        'next_id': replay.next_id, 'rejected': dict(replay.rejected)})


def ledger(replay):
    return normal({key: getattr(replay, key) for key in ('equity', 'fills', 'trades', 'actions', 'warnings')})


def simulate(frames, bars, calendar, signals, events, hold, modules):
    engine, access, *_ = modules
    replay = access.AccessibleReplay({}, calendar, signals, events, engine.Config(hold_days=hold))
    replay.frames, replay.bars = frames, bars
    return replay.run()


def record_day(session, dataset):
    init, prior, previous_hash = verify_records(session)
    # Do not accept a caller-supplied timestamp or a historical date override.
    now = datetime.now(timezone.utc)
    cutoff = read_json(dataset / 'manifest.json')['end']
    manifest, frames, calendar, raw_events, schedule = load_dataset(dataset, cutoff)
    earliest = (date.fromisoformat(init['historical_data_last_day']) + timedelta(days=1)).isoformat()
    following = validate_window(cutoff, schedule, now, earliest, prior[-1][1] if prior else None)
    runtime = session / 'runtime'
    modules = load_runtime(runtime)
    events, audit = modules[2].canonicalize_events(raw_events)
    if audit['unresolved_incomplete_groups']:
        raise ValueError('Unresolved corporate-action identity; no new record permitted')
    panel, breadth, signals = infer(frames, calendar, cutoff, runtime, modules)
    first = prior[0][1]['signal_date'] if prior else cutoff
    days = [d for d in calendar if first <= d <= cutoff]
    bars = {code: frame.set_index('date').to_dict('index') for code, frame in frames.items()}
    destination = session / 'days' / cutoff
    if destination.exists():
        raise ValueError('Duplicate signal day; sealed records cannot be overwritten')
    staging = session / 'days' / f'.{cutoff}-pending'
    if staging.exists():
        raise ValueError('Interrupted attempt exists; inspect it before trying again')
    staging.mkdir()
    all_signals = []
    result_rows = []
    for name, spec in init['accounts'].items():
        old = [pd.read_csv(folder / 'signals.csv', dtype={'date': str, 'code': str}, float_precision='round_trip')
               for folder, _ in prior]
        old = [frame[frame.account.eq(name)][['date', 'code', 'score']] for frame in old]
        previous_signals = pd.concat(old, ignore_index=True) if old else pd.DataFrame(columns=['date', 'code', 'score'])
        chosen = signals[name].copy()
        chosen['account'] = name
        all_signals.append(chosen)
        joined = pd.concat([previous_signals, signals[name]], ignore_index=True)
        # Reconstruct every prior account day using only its already sealed
        # signals. Changes to old prices/events must not rewrite its ledger.
        for folder, previous in prior:
            old_days = [d for d in days if d <= previous['signal_date']]
            history = joined[joined.date.le(previous['signal_date'])]
            restored = simulate(frames, bars, old_days, history, events, spec['hold_days'], modules)
            sealed = read_json(folder / name / 'account.json')
            if object_hash(account_state(restored)) != object_hash(sealed['state']) or object_hash(ledger(restored)) != object_hash(sealed['ledger']):
                raise ValueError(f'Prior account changed: {previous["signal_date"]}/{name}; inspect data/event revisions')
        replay = simulate(frames, bars, days, joined, events, spec['hold_days'], modules)
        summary = replay.summary()
        if replay.warnings:
            write_json(staging / 'quality-block.json', {'account': name, 'warnings': replay.warnings})
            raise ValueError(f'Account quality warning: {name}; preserve attempt and investigate')
        account = staging / name
        account.mkdir()
        write_json(account / 'account.json', {'state': account_state(replay), 'ledger': ledger(replay), 'summary': summary})
        # These are precommitted next-open attempts; risk checks may reject them.
        write_json(account / 'next-open-attempts.json', {
            'execution_date': following, 'buy_attempts': replay.pending_buys,
            'sell_attempts': [{'code': p.code, 'shares': p.shares,
                'reason': 'stop_close_next_open' if p.stop_pending else 'fixed_horizon'}
                for p in replay.positions.values()
                if v08_sell_due(replay, p, days)]})
        for key, rows in ledger(replay).items():
            pd.DataFrame(rows).to_csv(account / f'{key}.csv', index=False, encoding='utf-8-sig')
        result_rows.append({'account': name, 'equity': summary['final_equity'],
            'net_return': summary['total_return'], 'closed_trades': summary['closed_trades'],
            'win_rate': summary['win_rate'], 'open_positions': summary['open_positions'],
            'fees': summary['execution_costs'], 'candidate_count': len(signals[name]),
            'next_buy_attempts': len(replay.pending_buys)})
    pd.concat(all_signals, ignore_index=True).to_csv(staging / 'signals.csv', index=False, encoding='utf-8-sig')
    panel.to_csv(staging / 'scores-and-features.csv', index=False, encoding='utf-8-sig')
    breadth.to_csv(staging / 'breadth.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(result_rows).to_csv(staging / 'account-comparison.csv', index=False, encoding='utf-8-sig')
    shutil.copyfile(dataset / 'manifest.json', staging / 'data-manifest.json')
    write_json(staging / 'event-normalization.json', audit)
    # Saving a hash alone is insufficient if the original dataset is later lost.
    # Retain the exact input snapshot for future reconstruction and audit.
    shutil.copytree(dataset, staging / 'input-snapshot')
    sealed = datetime.now(timezone.utc)
    validate_window(cutoff, schedule, sealed, earliest, prior[-1][1] if prior else None)
    files = {str(p.relative_to(staging)).replace('\\', '/'): digest(p) for p in sorted(staging.rglob('*')) if p.is_file()}
    write_json(staging / 'record.json', {'signal_date': cutoff, 'execution_date': following,
        'created_utc': now.isoformat(), 'sealed_utc': sealed.isoformat(),
        'sealed_beijing': sealed.astimezone(CST).isoformat(),
        'previous_record_sha256': previous_hash, 'model_sha256': digest(runtime / '2026-lgb.txt'),
        'dataset_path': str(dataset.resolve()), 'data_manifest_sha256': digest(dataset / 'manifest.json'),
        'files': files, 'prospective_record': True,
        'python_version': sys.version, 'numpy_version': np.__version__, 'pandas_version': pd.__version__,
        'account_start_signal_day': first, 'same_accounts_continued': True})
    staging.rename(destination)
    write_status(session, init, prior + [(destination, read_json(destination / 'record.json'))])
    print(f'SEALED {cutoff} -> next-open attempts {following}; records={len(prior) + 1}')
    print(pd.DataFrame(result_rows).to_string(index=False))


def write_status(session, init, prior):
    text = '# 未来模拟观察状态\n\n'
    text += f'每账户初始虚拟资金3万元。一个V0.8主策略、三个固定随机对照和现金对照。初始化时间：{init["created_beijing"]}。\n\n'
    text += f'已封存的未来信号日：**{len(prior)}**。'
    if not prior:
        text += '尚无未来预测记录或模拟成交，收益与胜率尚不可评估。\n'
    else:
        folder, latest = prior[-1]
        text += f'最新信号日：**{latest["signal_date"]}**；预定次开盘：{latest["execution_date"]}；封存：{latest["sealed_beijing"]}。\n\n'
        names = {'v08_verified':'V0.8核验版（无现金利息）','cash':'现金对照'}
        table = pd.read_csv(folder / 'account-comparison.csv')
        text += '| 账户 | 净值 | 累计净收益 | 已平仓 | 持仓数 | 次开盘买入尝试 |\n|---|---:|---:|---:|---:|---:|\n'
        for row in table[table.account.isin(names)].itertuples():
            text += f'| {names[row.account]} | {row.equity:.2f}元 | {row.net_return:.2%} | {row.closed_trades} | {row.open_positions} | {row.next_buy_attempts} |\n'
        text += f'\n[完整账户对照](days/{latest["signal_date"]}/account-comparison.csv)；[盘前封存记录](days/{latest["signal_date"]}/record.json)。\n'
    text += '\n本流程须手动运行；未设置定时任务或连接券商。次开盘订单只是虚拟尝试，可能因停牌、价格上限、整手、资金或仓位条件拒绝。未来收益只统计实际发生并封存的模拟成交。\n'
    (session / 'status.md').write_text(text, encoding='utf-8')


def status(session):
    init, prior, _ = verify_records(session)
    verify_freeze(session / 'runtime')
    print(f'SESSION {session}\nCreated Beijing: {init["created_beijing"]}\nSealed prospective days: {len(prior)}')
    print(f'Latest: {prior[-1][1]["signal_date"]}' if prior else 'No prospective prediction or executed paper trade yet')
    if prior:
        print(pd.read_csv(prior[-1][0] / 'account-comparison.csv').to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('init', 'record', 'status'))
    parser.add_argument('--session', type=Path, default=SESSION)
    parser.add_argument('--dataset', type=Path)
    args = parser.parse_args()
    try:
        if args.action == 'init':
            initialize(args.session.resolve())
        elif args.action == 'status':
            status(args.session.resolve())
        elif args.dataset is None:
            parser.error('record requires --dataset from update_paper_data.py')
        else:
            record_day(args.session.resolve(), args.dataset.resolve())
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f'BLOCKED: {exc}\n')



# V0.8 adapter. The append-only journal implementation above is retained;
# only inference, execution rules, and complete strategy-state serialization
# are adapted. Historical account signals are never recomputed.
_base_load_runtime = load_runtime
_base_account_state = account_state
_SIM_PANEL = None


def load_runtime(folder):
    return _base_load_runtime(folder)


def infer(frames, calendar, cutoff, runtime, modules):
    global _SIM_PANEL
    from v08_inference_verified import inference_panel, predict
    from exploration_features import breadth_days
    panel = inference_panel(frames, calendar, runtime / 'sources/ml_features-frozen.py')
    panel = predict(panel, runtime / 'sources/2026-v2.txt', runtime / '2026-lgb.txt')
    old = [pd.read_csv(p, dtype={'date':str,'code':str},float_precision='round_trip')
           for p in sorted((runtime.parent / 'days').glob('*/scores-and-features.csv'))
           if not p.parent.name.startswith('.')]
    _SIM_PANEL = pd.concat(old + [panel], ignore_index=True)
    breadth = breadth_days(panel)
    allowed = bool(len(breadth) and breadth.pool_size.iloc[0] >= 100 and breadth.breadth5.iloc[0] >= .45)
    selected = panel[panel.pred_net_return_v2.ge(.005)] if allowed else panel.iloc[:0]
    sig = selected[['date','code','pred_score_v08']].rename(columns={'pred_score_v08':'score'})
    signals = {'v08_verified':sig, 'cash':pd.DataFrame(columns=['date','code','score'])}
    for seed in SEEDS:
        signals[f'v08_verified_random_{seed}'] = random_matched(panel,len(sig),seed)
    return panel,breadth,signals


def simulate(frames,bars,calendar,signals,events,hold,modules):
    from v08_engine_verified import VerifiedV08Replay
    from exploration_features import breadth_days
    engine = modules[0]
    b = breadth_days(_SIM_PANEL)
    bm=b.set_index('date').breadth5.to_dict()
    pm=b.set_index('date').pool_size.to_dict()
    limits={d:0 if pm[d]<100 or v<.45 else (1 if v<.55 else 2) for d,v in bm.items()}
    vol=_SIM_PANEL.set_index(['date','code']).ret_vol20.to_dict()
    cfg=engine.Config(capital=30000,order_budget=7500,max_positions=5,max_exposure=.90,hold_days=10)
    rep=VerifiedV08Replay({},calendar,signals,events,cfg,vol_map=vol,daily_limits=limits,
        breadth_lookup=bm,enable_cash_overlay=False)
    rep.frames,rep.bars=frames,bars
    return rep.run()


def account_state(rep):
    result=_base_account_state(rep)
    result['peak_prices']=normal(rep.peak_prices)
    result['profit_lock_flags']={code:{'breakeven_active':bool(getattr(p,'breakeven_active',False)),
        'moonbag_active':bool(getattr(p,'moonbag_active',False))} for code,p in rep.positions.items()}
    return result


def v08_sell_due(rep,position,days):
    horizon=12 if rep.daily_limits.get(days[-1],1)>=2 else 8
    return position.stop_pending or len(days)-position.entry_index>=horizon


if __name__ == '__main__':
    main()
