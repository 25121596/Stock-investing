"""V0.8.6 Grand Synthesis and V0.8 Baseline Forward Parallel Paper Batch.

Deploys 9 isolated small-capital accounts (30,000 RMB each) under strict A-share microstructure,
anti-hindsight stop-loss priority, 1.2% conviction turnover damping, and SHA-256 cryptographic
hash chain sealing. Never executes real brokerage orders.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import v08_paper_forward_v3 as legacy

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
SESSION = ROOT / 'results/v086-parallel-paper-20261023-v1'
FREEZE = legacy.FREEZE
CST = legacy.CST
SEEDS = legacy.SEEDS

digest = legacy.digest
read_json = legacy.read_json
normal = legacy.normal
object_hash = legacy.object_hash
validate_window = legacy.validate_window
load_dataset = legacy.load_dataset
account_state = legacy.account_state
ledger = legacy.ledger
runtime_versions = legacy.runtime_versions


def ensure_venv():
    """Auto-select project virtualenv python if invoked via an external python binary."""
    venv_py = ROOT / '.venv/Scripts/python.exe'
    if venv_py.is_file() and Path(sys.executable).resolve() != venv_py.resolve() and os.environ.get('_VENV_REEXEC') != '1':
        os.environ['_VENV_REEXEC'] = '1'
        proc = subprocess.run([str(venv_py)] + sys.argv, cwd=str(Path.cwd()))
        sys.exit(proc.returncode)


def write_new(path: Path | str, value):
    path = Path(path)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def accounts() -> dict:
    """Return dictionary of the 9 authorized forward paper accounts."""
    result = {
        'v086_winner': {
            'policy': 'v086_winner',
            'model_type': 'v086_winner',
            'plan_mode': 'plan_a',
            'turnover_damping': True,
            'damping_hurdle': 0.0120,
            'seed': None,
            'display': 'V0.8.6优胜大成模型(方案A+换手阻尼)',
        },
        'v08_verified': {
            'policy': 'v08_verified',
            'model_type': 'v08_verified',
            'plan_mode': 'baseline',
            'turnover_damping': False,
            'damping_hurdle': 0.0,
            'seed': None,
            'display': 'V0.8冻结基线',
        },
    }
    for seed in SEEDS:
        result[f'v086_winner_random_{seed}'] = {
            'policy': 'v086_winner',
            'model_type': 'v086_winner',
            'plan_mode': 'plan_a',
            'turnover_damping': True,
            'damping_hurdle': 0.0120,
            'seed': seed,
            'display': f'V0.8.6优胜随机对照{seed}',
        }
    for seed in SEEDS:
        result[f'v08_verified_random_{seed}'] = {
            'policy': 'v08_verified',
            'model_type': 'v08_verified',
            'plan_mode': 'baseline',
            'turnover_damping': False,
            'damping_hurdle': 0.0,
            'seed': seed,
            'display': f'V0.8基线随机对照{seed}',
        }
    result['cash'] = {
        'policy': 'cash',
        'model_type': 'cash',
        'plan_mode': 'none',
        'turnover_damping': False,
        'damping_hurdle': 0.0,
        'seed': None,
        'display': '现金对照',
    }
    return result


def file_hashes(folder: Path) -> dict[str, str]:
    return {p.relative_to(folder).as_posix(): digest(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def protected_hashes() -> dict[str, str]:
    """Cryptographically binds all historical V0.8 and V0.8.1 artifacts to guarantee batch isolation."""
    paths = [
        ROOT / 'results/v08-paper-forward-v3/init.json',
        ROOT / 'results/v08-paper-forward-v3/runtime/freeze.json',
        ROOT.parent / 'ashare-simulator/v08-paper/gateway-freeze.json',
        ROOT / 'v08_paper_forward_v3.py',
        ROOT / 'results/v081-parallel-paper-20261022-v1/init.json',
        ROOT / 'results/v081-parallel-paper-20261022-v1/runtime-manifest.json',
        ROOT / 'results/v081-parallel-paper-20261022-v1/protocol.md',
        ROOT / 'results/v081-parallel-paper-20261022-v1/runner-source.py',
        ROOT.parent / 'ashare-simulator/v081-parallel/gateway-freeze.json',
        ROOT / 'v081_parallel_paper.py',
        ROOT / 'run_v081_parallel_day.py',
        ROOT.parent / 'ashare-simulator/v081_parallel_gateway.py',
    ]
    old_v08 = ROOT / 'results/v08-paper-forward-v3'
    if (old_v08 / 'days').is_dir():
        for folder in sorted((old_v08 / 'days').iterdir()):
            if folder.is_dir() and not folder.name.startswith('.'):
                record = read_json(folder / 'record.json')
                paths.append(folder / 'record.json')
                for name, expected in record.get('files', {}).items():
                    target = folder / name
                    if digest(target) != expected:
                        raise ValueError(f'Original sealed artifact already differs: {name}')
                    paths.append(target)
    old_v081 = ROOT / 'results/v081-parallel-paper-20261022-v1'
    if (old_v081 / 'days').is_dir():
        for folder in sorted((old_v081 / 'days').iterdir()):
            if folder.is_dir() and not folder.name.startswith('.'):
                record = read_json(folder / 'record.json')
                paths.append(folder / 'record.json')
                for name, expected in record.get('files', {}).items():
                    target = folder / name
                    if digest(target) != expected:
                        raise ValueError(f'V0.8.1 sealed artifact already differs: {name}')
                    paths.append(target)
    return {str(p.resolve()): digest(p) for p in paths if p.is_file()}


def protocol_text() -> str:
    return """# V0.8.6 优胜大成模型与基线前瞻模拟批次协议

2026-10-23。
用户协议：建立全新前瞻模拟批次，验证V0.8.6大成体系（方案A动态50%浮盈保护 + 1.2%信念超额换手阻尼），与冻结V0.8基线及随机对照进行独立前瞻并行比较。
本批次建立在完全独立的批次目录 `results/v086-parallel-paper-20261023-v1` 与 `ashare-simulator/v086-parallel`，历史V0.8与V0.8.1批次严格隔离封存，绝不篡改历史。

## 1. 9个独立账户定义 (各30,000元虚拟初始资金)
1. `v086_winner`: V0.8.6优胜大成模型(方案A 50%浮盈保护 + 1.2%信念超额换手阻尼)
2. `v08_verified`: 冻结V0.8基线
3. `v086_winner_random_2026101401`: V0.8.6优胜固定种子随机对照1
4. `v086_winner_random_2026101402`: V0.8.6优胜固定种子随机对照2
5. `v086_winner_random_2026101403`: V0.8.6优胜固定种子随机对照3
6. `v08_verified_random_2026101401`: V0.8基线固定种子随机对照1
7. `v08_verified_random_2026101402`: V0.8基线固定种子随机对照2
8. `v08_verified_random_2026101403`: V0.8基线固定种子随机对照3
9. `cash`: 100%现金对照(无交易)

## 2. 交易摩擦与微观规则
- 初始本金：各账户30,000元，整手(100股)交易。
- 佣金万2.5(最低5元)，印花税千0.5(卖出单边)，过户费万0.01。
- 基准滑点万5(0.0005)；日内触价单附加10 bps惩罚滑点(合计15 bps)。
- 严格T+1锁定，跌停不可卖，涨停不可买，停牌拦截。
- 日内天地板双触价时强制止损优先，杜绝后见之明。
- 资产负债守恒：每日收盘严格满足 |NAV - (Cash + Receivables + Market_Value)| < 1e-6。

## 3. 封存与合规铁律
- 仅在收盘日15:30之后、次交易日09:15之前封存当日数据。
- 严禁任何形式的未来信息泄露与历史回补。
"""


def initialize(session: Path = SESSION):
    session = Path(session)
    if session.exists():
        raise ValueError('Batch exists; never reset or overwrite a paper account')
    original = legacy.verify_freeze(FREEZE)
    for name in ['v08_paper_forward_v3.py', 'v08_inference_verified.py', 'v08_engine_verified.py',
                 'exploration_features.py']:
        if digest(ROOT / name) != original['source_hashes'][name]:
            raise ValueError('Shared V0.8 source differs from the verified freeze: ' + name)
    protected = protected_hashes()
    now = datetime.now(timezone.utc)
    session.mkdir(parents=True)
    (session / 'days').mkdir()
    shutil.copytree(FREEZE, session / 'runtime', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(__file__, session / 'runner-source.py')
    (session / 'protocol.md').write_text(protocol_text(), encoding='utf-8')
    manifest = {
        'runtime': file_hashes(session / 'runtime'),
        'runner_sha256': digest(__file__),
        'legacy_helper_sha256': digest(ROOT / 'v08_paper_forward_v3.py'),
        'protocol_sha256': digest(session / 'protocol.md'),
        'operational_sources': {
            name: digest(ROOT / name) for name in ['run_v086_parallel_day.py', 'update_paper_data.py']
            if (ROOT / name).is_file()
        },
        'v086_sources': {
            name: digest(ROOT / name) for name in ['engine_core.py', 'engine_intraday.py']
            if (ROOT / name).is_file()
        },
    }
    write_new(session / 'runtime-manifest.json', manifest)
    init = {
        'version': 'V0.8.6 forward parallel paper batch',
        'champion_model': 'v086_winner (Plan A 50% Profit Protection + Turnover Damping 1.2%)',
        'baseline_model': 'v08_verified (Frozen V0.8 Baseline)',
        'created_utc': now.isoformat(),
        'created_beijing': now.astimezone(CST).isoformat(),
        'prospective_start_not_before': now.astimezone(CST).date().isoformat(),
        'accounts': accounts(),
        'initial_cash_per_account': 30000,
        'model_year': 2026,
        'random_seeds': list(SEEDS),
        'runtime_manifest_sha256': digest(session / 'runtime-manifest.json'),
        'runtime_versions': runtime_versions(),
        'protected_hashes': protected,
        'record_deadline': '15:30 signal close through next exchange session 09:15 Asia/Shanghai',
        'prior_batches': [
            str((ROOT / 'results/v08-paper-forward-v3').resolve()),
            str((ROOT / 'results/v081-parallel-paper-20261022-v1').resolve()),
        ],
        'independent_batch': True,
        'no_real_brokerage_orders': True,
        'timestamp_limit': 'Local clock and hashes are not an external timestamp authority',
    }
    write_new(session / 'init.json', init)
    write_status(session, init, [])
    print(f'INITIALIZED {session} ; 9 independent accounts; 0 sealed signals; no performance yet')


def verify_runtime(session: Path, init: dict):
    session = Path(session)
    path = session / 'runtime-manifest.json'
    if digest(path) != init['runtime_manifest_sha256']:
        raise ValueError('Runtime manifest changed')
    manifest = read_json(path)
    if digest(__file__) != manifest['runner_sha256'] or digest(session / 'runner-source.py') != manifest['runner_sha256']:
        raise ValueError('Parallel recorder changed; new code requires a new batch')
    if digest(ROOT / 'v08_paper_forward_v3.py') != manifest['legacy_helper_sha256']:
        raise ValueError('Legacy helper source changed')
    if digest(session / 'protocol.md') != manifest['protocol_sha256']:
        raise ValueError('Batch protocol changed')
    for name, expected in manifest.get('operational_sources', {}).items():
        if (ROOT / name).is_file() and digest(ROOT / name) != expected:
            raise ValueError('Frozen daily operational source changed: ' + name)
    for name, expected in manifest.get('v086_sources', {}).items():
        if (ROOT / name).is_file() and digest(ROOT / name) != expected:
            raise ValueError('V0.8.6 source changed: ' + name)
    if file_hashes(session / 'runtime') != manifest['runtime']:
        raise ValueError('Frozen runtime source/model content changed')
    legacy.verify_freeze(session / 'runtime')
    if init['runtime_versions'] != runtime_versions():
        raise ValueError('Frozen inference library versions changed')
    if init['accounts'] != accounts() or init['initial_cash_per_account'] != 30000:
        raise ValueError('Account/rule identity differs from frozen recorder')
    for path_str, expected in init['protected_hashes'].items():
        if digest(Path(path_str)) != expected:
            raise ValueError('Protected historical artifact changed: ' + path_str)


def verify_records(session: Path):
    session = Path(session)
    init = read_json(session / 'init.json')
    verify_runtime(session, init)
    previous = digest(session / 'init.json')
    records = []
    for folder in sorted((session / 'days').iterdir()):
        if not folder.is_dir():
            raise ValueError('Unexpected file in day journal')
        if folder.name.startswith('.'):
            raise ValueError('Interrupted staging record exists; preserve and inspect it')
        record = read_json(folder / 'record.json')
        if record['signal_date'] != folder.name or record['previous_record_sha256'] != previous:
            raise ValueError('Broken date or record hash chain')
        if records and records[-1][1]['execution_date'] != record['signal_date']:
            raise ValueError('Missing prospective trading-day record')
        expected = record['files']
        actual = {p.relative_to(folder).as_posix(): digest(p) for p in folder.rglob('*')
                  if p.is_file() and p != folder / 'record.json'}
        if actual != expected:
            raise ValueError('Sealed input/order/account artifact changed: ' + folder.name)
        if record['account_ids'] != list(init['accounts']):
            raise ValueError('Sealed account set changed')
        if record['signal_date'] < init['prospective_start_not_before'] or record['signal_date'][:4] != '2026':
            raise ValueError('Record predates this batch or frozen model year')
        sealed = datetime.fromisoformat(record['sealed_beijing'])
        schedule = pd.read_csv(folder / 'input-snapshot/exchange-calendar.csv', dtype=str)
        schedule = schedule[schedule.is_trading_day.eq('1')].calendar_date.tolist()
        following = validate_window(record['signal_date'], schedule, sealed,
                                    init['prospective_start_not_before'], records[-1][1] if records else None)
        if following != record['execution_date'] or record.get('prospective_record') is not True:
            raise ValueError('Invalid next-session or evidence identity')
        previous = digest(folder / 'record.json')
        records.append((folder, record))
    return init, records, previous


def load_runtime(session: Path):
    return legacy.load_runtime(session / 'runtime')


def make_signals(panel: pd.DataFrame, breadth: pd.DataFrame) -> dict[str, pd.DataFrame]:
    valid = bool(len(breadth) and breadth.pool_size.iloc[0] >= 100
                 and pd.notna(breadth.breadth5.iloc[0]) and breadth.breadth5.iloc[0] >= .45)
    selected = panel[panel.pred_net_return_v2.ge(.005)] if valid else panel.iloc[:0]
    sig = selected[['date', 'code', 'pred_score_v08']].rename(columns={'pred_score_v08': 'score'})
    result = {}
    for name, spec in accounts().items():
        if spec['policy'] == 'cash':
            result[name] = pd.DataFrame(columns=['date', 'code', 'score'])
        elif spec['seed'] is None:
            result[name] = sig.copy()
        else:
            result[name] = legacy.random_matched(panel, len(sig), spec['seed'])
    return result


def infer(frames, calendar, cutoff, runtime, modules):
    from v08_inference_verified import inference_panel, predict
    from exploration_features import breadth_days
    panel = predict(inference_panel(frames, calendar, runtime / 'sources/ml_features-frozen.py'),
                    runtime / 'sources/2026-v2.txt', runtime / '2026-lgb.txt')
    if panel.empty or not panel.date.eq(cutoff).all() or panel.duplicated(['date', 'code']).any():
        raise ValueError('Missing, duplicate or wrong-date current inference rows')
    breadth = breadth_days(panel)
    return panel, breadth, make_signals(panel, breadth)


def simulate(frames, bars, calendar, signals, events, spec, modules, panel):
    from exploration_features import breadth_days
    from engine_core import ModularReplayEngine, BaselineRegimeStopping
    from engine_intraday import IntradayTouchStopping
    from engine import Config

    if panel.duplicated(['date', 'code']).any() or set(panel.date) != set(calendar):
        raise ValueError('Frozen context must cover every replay day once per stock')
    b = breadth_days(panel)
    bm = b.set_index('date').breadth5.to_dict()
    pm = b.set_index('date').pool_size.to_dict()
    limits = {d: 0 if pm[d] < 100 or not np.isfinite(v) or v < .45 else (1 if v < .55 else 2)
              for d, v in bm.items()}
    vol = panel.set_index(['date', 'code']).ret_vol20.to_dict()
    yz_vol = panel.set_index(['date', 'code']).yz_vol20.to_dict() if 'yz_vol20' in panel.columns else vol

    policy = spec['policy']
    is_winner = (policy == 'v086_winner')

    if is_winner:
        stopping = IntradayTouchStopping(plan_mode='plan_a')
        rep = ModularReplayEngine(
            frames=frames,
            calendar=calendar,
            signals=signals,
            dividends=events,
            config=Config(capital=30000, order_budget=7500, max_positions=5, max_exposure=0.90, hold_days=10),
            vol_map=vol,
            yz_vol_map=yz_vol,
            daily_limits=limits,
            breadth_lookup=bm,
            enable_cash_overlay=False,
            stopping=stopping,
            enable_intraday_touch=True,
            turnover_hurdle=0.0120,
            enable_turnover_damping=True,
        )
    else:
        stopping = BaselineRegimeStopping()
        rep = ModularReplayEngine(
            frames=frames,
            calendar=calendar,
            signals=signals,
            dividends=events,
            config=Config(capital=30000, order_budget=7500, max_positions=5, max_exposure=0.90, hold_days=10),
            vol_map=vol,
            yz_vol_map=yz_vol,
            daily_limits=limits,
            breadth_lookup=bm,
            enable_cash_overlay=False,
            stopping=stopping,
            enable_intraday_touch=False,
            turnover_hurdle=0.0,
            enable_turnover_damping=False,
        )
    rep.bars = bars
    rep.gate_decisions = []
    return rep.run()


def record_day(session: Path, dataset: Path):
    session, dataset = Path(session), Path(dataset)
    init, prior, previous_hash = verify_records(session)
    now = datetime.now(timezone.utc)
    cutoff = read_json(dataset / 'manifest.json')['end']
    manifest, frames, calendar, raw_events, schedule = load_dataset(dataset, cutoff)
    following = validate_window(cutoff, schedule, now, init['prospective_start_not_before'],
                                prior[-1][1] if prior else None)
    modules = load_runtime(session)
    events, audit = modules[2].canonicalize_events(raw_events)
    if audit['unresolved_incomplete_groups']:
        raise ValueError('Unresolved corporate actions')
    panel, breadth, signals = infer(frames, calendar, cutoff, session / 'runtime', modules)
    old_panels = [pd.read_csv(folder / 'scores-and-features.csv', dtype={'date': str, 'code': str},
                             float_precision='round_trip') for folder, _ in prior]
    context = pd.concat(old_panels + [panel], ignore_index=True)
    first = prior[0][1]['signal_date'] if prior else cutoff
    days = [d for d in calendar if first <= d <= cutoff]
    bars = {code: frame.set_index('date').to_dict('index') for code, frame in frames.items()}
    destination = session / 'days' / cutoff
    staging = session / 'days' / f'.{cutoff}-pending'
    if destination.exists() or staging.exists():
        raise ValueError('Sealed day or interrupted attempt already exists')
    staging.mkdir()
    all_signals, result_rows = [], []
    for name, spec in init['accounts'].items():
        prior_signals = []
        for folder, _ in prior:
            saved = pd.read_csv(folder / 'signals.csv', dtype={'date': str, 'code': str}, float_precision='round_trip')
            prior_signals.append(saved[saved.account.eq(name)][['date', 'code', 'score']])
        joined = pd.concat(prior_signals + [signals[name]], ignore_index=True)
        for folder, old in prior:
            old_days = [d for d in days if d <= old['signal_date']]
            restored = simulate(frames, bars, old_days, joined[joined.date.le(old['signal_date'])], events,
                                spec, modules, context[context.date.le(old['signal_date'])])
            sealed = read_json(folder / name / 'account.json')
            if object_hash(account_state(restored)) != object_hash(sealed['state']) or object_hash(ledger(restored)) != object_hash(sealed['ledger']):
                raise ValueError('Historical data/event revision changes sealed state: ' + old['signal_date'] + '/' + name)
        rep = simulate(frames, bars, days, joined, events, spec, modules, context)
        if rep.warnings or getattr(rep, 'review_warnings', []):
            raise ValueError('Account quality warning: ' + name)
        summary = rep.summary()
        summary['validation_scope'] = 'Sealed prospective batch; no independent profit claim'
        account = staging / name
        account.mkdir()
        write_new(account / 'account.json', {'account': name, 'spec': spec, 'state': account_state(rep),
                                            'ledger': ledger(rep), 'summary': summary})
        write_new(account / 'next-open-attempts.json', {'account': name, 'execution_date': following,
            'buy_attempts': rep.pending_buys, 'sell_attempts': [
                {'code': p.code, 'shares': p.shares,
                 'reason': 'stop_close_next_open' if getattr(p, 'stop_pending', False) else 'fixed_horizon'}
                for p in rep.positions.values() if p.shares > 0 and legacy.v08_sell_due(rep, p, days)]})
        for key, rows in ledger(rep).items():
            pd.DataFrame(rows).to_csv(account / f'{key}.csv', index=False)
        pd.DataFrame(getattr(rep, 'gate_decisions', [])).to_csv(account / 'gate-decisions.csv', index=False)
        selected = signals[name].copy()
        selected['account'] = name
        all_signals.append(selected)
        gate = rep.gate_decisions[-1] if getattr(rep, 'gate_decisions', []) else {
            'proposed_orders': rep.pending_buys, 'cancelled_orders': 0
        }
        result_rows.append({'account': name, 'equity': summary['final_equity'], 'net_return': summary['total_return'],
            'closed_trades': summary['closed_trades'], 'win_rate': summary['win_rate'],
            'open_positions': summary['open_positions'], 'fees': summary['execution_costs'],
            'candidate_count': len(signals[name]), 'proposed_buy_orders': len(gate['proposed_orders']),
            'cancelled_buy_orders': gate['cancelled_orders'], 'next_buy_attempts': len(rep.pending_buys)})
    pd.concat(all_signals, ignore_index=True).to_csv(staging / 'signals.csv', index=False)
    panel.to_csv(staging / 'scores-and-features.csv', index=False)
    breadth.to_csv(staging / 'breadth.csv', index=False)
    pd.DataFrame(result_rows).to_csv(staging / 'account-comparison.csv', index=False)
    shutil.copyfile(dataset / 'manifest.json', staging / 'data-manifest.json')
    write_new(staging / 'event-normalization.json', audit)
    shutil.copytree(dataset, staging / 'input-snapshot')
    sealed = datetime.now(timezone.utc)
    validate_window(cutoff, schedule, sealed, init['prospective_start_not_before'], prior[-1][1] if prior else None)
    verify_runtime(session, init)
    files = file_hashes(staging)
    write_new(staging / 'record.json', {'signal_date': cutoff, 'execution_date': following,
        'created_utc': now.isoformat(), 'sealed_utc': sealed.isoformat(), 'sealed_beijing': sealed.astimezone(CST).isoformat(),
        'previous_record_sha256': previous_hash, 'account_ids': list(init['accounts']),
        'files': files, 'data_manifest_sha256': digest(dataset / 'manifest.json'),
        'dataset_path': str(dataset.resolve()), 'prospective_record': True,
        'runtime_manifest_sha256': init['runtime_manifest_sha256'], 'independent_batch': True})
    staging.rename(destination)
    write_status(session, init, prior + [(destination, read_json(destination / 'record.json'))])
    print('SEALED', cutoff, '->', following, '; 9 accounts; records', len(prior) + 1)
    print(pd.DataFrame(result_rows).to_string(index=False))


def write_status(session: Path, init: dict, prior: list):
    session = Path(session)
    latest = prior[-1] if prior else None
    text = '# V0.8.6 优胜大成模型与基线：独立前瞻模拟\n\n'
    text += '采用V0.8.6优胜大成模型（方案A 50%浮盈保护 + 1.2%换手阻尼），与冻结V0.8基线及随机对照并行。9账户各3万元。\n\n'
    text += f'已封存信号日：**{len(prior)}**。\n\n'
    if latest:
        folder, record = latest
        text += f'最新信号{record["signal_date"]}；次开盘{record["execution_date"]}；封存{record["sealed_beijing"]}。\n\n'
        rows = pd.read_csv(folder / 'account-comparison.csv')
        text += '| 账户 | 净值 | 累计净收益 | 平仓 | 持仓 | 待买尝试 |\n|---|---:|---:|---:|---:|---:|\n'
        for row in rows.itertuples(index=False):
            label = init['accounts'][row.account]['display']
            text += f'| {label} | {row.equity:.2f}元 | {row.net_return:.2%} | {row.closed_trades} | {row.open_positions} | {row.next_buy_attempts} |\n'
    else:
        text += '已初始化，等待合法收盘数据；没有未来预测、成交或可评估业绩。\n'
    text += '\n历史V0.8与V0.8.1批次记录原样隔离封存，本批次独立从头建账。\n'
    text += '\n窗口：收盘日15:30后至次交易日09:15前。20交易日流程检查、60交易日阶段复核。\n'
    text += '\n[冻结协议](protocol.md)；[初始化配置](init.json)；[运行依赖清单](runtime-manifest.json)。不发送真实券商订单。\n'
    (session / 'status.md').write_text(text, encoding='utf-8')
    if latest:
        table = pd.read_csv(latest[0] / 'account-comparison.csv').to_html(index=False)
    else:
        table = '<table><tr><th>账户</th><th>初始虚拟资金</th><th>状态</th></tr>' + ''.join(
            f'<tr><td>{html.escape(s["display"])}</td><td>30,000元</td><td>等待首次封存；无业绩</td></tr>'
            for s in init['accounts'].values()) + '</table>'
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>V0.8.6前瞻模拟</title>'
    page += '<style>body{font:16px/1.7 system-ui,"Microsoft YaHei";max-width:1250px;margin:32px auto;padding:0 20px;color:#243047;background:#f7f9fc}table{border-collapse:collapse;background:white;width:100%}td,th{padding:10px;border:1px solid #d7e0e8;text-align:left}.note{background:#fff1d6;padding:16px;border-left:4px solid #b87d10}</style>'
    page += '<h1>V0.8.6 优胜大成模型与基线：前瞻模拟</h1>'
    page += f'<p><b>已封存 {len(prior)} 个信号日</b> · 9账户独立各3万元本金</p>'
    page += '<p class="note">只有事先封存并实际发生的模拟成交才计入未来成绩；初始化和空账户均不证明盈利。</p>'
    page += table + '<p>历史批次完全隔离。今日信号须在收盘后完整数据核验封存，随后由RQAlpha独立撮合核验。</p>'
    page += '<p><a href="protocol.md">冻结协议</a> · <a href="status.md">完整状态</a> · <a href="init.json">账户配置</a> · <a href="../../../ashare-simulator/v086-parallel/status.html">RQAlpha接入状态</a></p></html>'
    (session / 'status.html').write_text(page, encoding='utf-8')


def status(session: Path = SESSION):
    init, records, _ = verify_records(session)
    write_status(Path(session), init, records)
    print('VERIFIED', len(records), 'sealed signal days;', len(init['accounts']), 'independent accounts')
    if records:
        print('LATEST', records[-1][1]['signal_date'])


def main():
    ensure_venv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['init', 'record', 'status'])
    parser.add_argument('--session', type=Path, default=SESSION)
    parser.add_argument('--dataset', type=Path)
    args = parser.parse_args()
    try:
        if args.action == 'init':
            initialize(args.session.resolve())
        elif args.action == 'status':
            status(args.session.resolve())
        elif args.dataset is None:
            parser.error('record needs a freshly verified --dataset')
        else:
            record_day(args.session.resolve(), args.dataset.resolve())
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f'BLOCKED: {exc}\n')


if __name__ == '__main__':
    main()
