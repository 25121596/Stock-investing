"""V0.8.7-GLM research candidate forward parallel paper batch (built by GLM checkpoint).

Deploys 6 isolated small-capital accounts (30,000 RMB each) for prospective shadow
observation of the V0.8.7-GLM research candidate (breadth gate >= 0.58 + intraday
stop-loss-touch-only execution + close-based profit exits) against the frozen V0.8
baseline, matched random controls, and cash. SHA-256 hash chain sealing. Never
executes real brokerage orders.

Batch lineage: strictly isolated from the sealed V0.8 / V0.8.1 / V0.8.6 batches,
whose artifacts are cryptographically protected by this batch. The V0.8.6 batch
keeps running unchanged; this batch adds the GLM candidate as a shadow arm.

Disclosed divergence from the accepted backtest (results/v087-glm-v1): the frozen
paper runtime exposes pred_score_v08 (not the adaptive_score blend used in the
research evaluation), so candidate ranking here uses pred_score_v08 with the
breadth gate and SL-touch stopping exactly as accepted. Turnover damping stays
OFF per the pre-registered REJECT_DAMPING verdict (2026-10-25).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
SESSION = ROOT / 'results/v087-glm-parallel-paper-20261025-v1'
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

MADE_BY = 'GLM checkpoint 2026-10-25'


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
    """Return dictionary of the 6 authorized V0.8.7-GLM forward paper accounts."""
    result = {
        'v087_glm': {
            'policy': 'v087_glm',
            'model_type': 'v087_glm',
            'breadth_floor': 0.58,
            'intraday_mode': 'sl',
            'turnover_damping': False,
            'damping_hurdle': 0.0,
            'seed': None,
            'display': 'V0.8.7-GLM研究候选(广度门控0.58+仅日内止损触价)',
        },
        'v08_verified_ref': {
            'policy': 'v08_verified_ref',
            'model_type': 'v08_verified',
            'breadth_floor': 0.0,
            'intraday_mode': 'baseline',
            'turnover_damping': False,
            'damping_hurdle': 0.0,
            'seed': None,
            'display': '冻结V0.8基线参照',
        },
    }
    for seed in SEEDS:
        result[f'v087_glm_random_{seed}'] = {
            'policy': 'v087_glm',
            'model_type': 'v087_glm',
            'breadth_floor': 0.58,
            'intraday_mode': 'sl',
            'turnover_damping': False,
            'damping_hurdle': 0.0,
            'seed': seed,
            'display': f'V0.8.7-GLM固定种子随机对照{seed}',
        }
    result['cash'] = {
        'policy': 'cash',
        'model_type': 'cash',
        'breadth_floor': 0.0,
        'intraday_mode': 'none',
        'turnover_damping': False,
        'damping_hurdle': 0.0,
        'seed': None,
        'display': '现金对照',
    }
    return result


def file_hashes(folder: Path) -> dict[str, str]:
    return {p.relative_to(folder).as_posix(): digest(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def protected_hashes() -> dict[str, str]:
    """Binds all prior sealed batches (V0.8, V0.8.1, V0.8.6) to guarantee isolation."""
    paths = [
        ROOT / 'results/v08-paper-forward-v3/init.json',
        ROOT / 'results/v08-paper-forward-v3/runtime/freeze.json',
        ROOT.parent / 'ashare-simulator/v08-paper/gateway-freeze.json',
        ROOT / 'v08_paper_forward_v3.py',
        ROOT / 'results/v081-parallel-paper-20261022-v1/init.json',
        ROOT / 'results/v081-parallel-paper-20261022-v1/runtime-manifest.json',
        ROOT / 'results/v081-parallel-paper-20261022-v1/protocol.md',
        ROOT / 'results/v081-parallel-paper-20261022-v1/runner-source.py',
        ROOT.parent / 'ashare-simulator/v081_parallel/gateway-freeze.json',
        ROOT / 'v081_parallel_paper.py',
        ROOT / 'run_v081_parallel_day.py',
        ROOT.parent / 'ashare-simulator/v081_parallel_gateway.py',
        ROOT / 'results/v086-parallel-paper-20261023-v1/init.json',
        ROOT / 'results/v086-parallel-paper-20261023-v1/runtime-manifest.json',
        ROOT / 'results/v086-parallel-paper-20261023-v1/protocol.md',
        ROOT / 'results/v086-parallel-paper-20261023-v1/runner-source.py',
        ROOT / 'v086_parallel_paper.py',
        ROOT / 'run_v086_parallel_day.py',
    ]
    for session_name in ['v08-paper-forward-v3', 'v081-parallel-paper-20261022-v1', 'v086-parallel-paper-20261023-v1']:
        base = ROOT / 'results' / session_name / 'days'
        if base.is_dir():
            for folder in sorted(base.iterdir()):
                if folder.is_dir() and not folder.name.startswith('.'):
                    record = read_json(folder / 'record.json')
                    paths.append(folder / 'record.json')
                    for name, expected in record.get('files', {}).items():
                        target = folder / name
                        if digest(target) != expected:
                            raise ValueError(f'Sealed artifact already differs: {session_name}/{folder.name}/{name}')
                        paths.append(target)
    return {str(p.resolve()): digest(p) for p in paths if p.is_file()}


def protocol_text() -> str:
    return """# V0.8.7-GLM 研究候选前瞻影子模拟批次协议（GLM 检查点建立）

2026-10-25。
目的：以独立前瞻影子批次观察 V0.8.7-GLM 研究候选（信号日市场广度 breadth5 ≥ 0.58 才接受次日新买单 + 日内仅止损触价（含10 bps惩罚滑点）+ 止盈与浮盈保护按收盘判定次日开盘退出），与冻结 V0.8 基线参照、同引擎随机对照并行比较。
本批次建立在完全独立目录 `results/v087-glm-parallel-paper-20261025-v1`；历史 V0.8、V0.8.1、V0.8.6 批次严格隔离封存（哈希保护），绝不篡改历史，也不替换 V0.8.6 批次的任何账户。
换手阻尼依预注册复核结论（REJECT_DAMPING，2026-10-25）保持关闭。
与已通过验收的回测（results/v087-glm-v1）的差异已披露：前瞻运行时仅提供 pred_score_v08，候选排序采用该分数；广度门控与仅日内止损触价机制与回测一致。

## 1. 6个独立账户定义 (各30,000元虚拟初始资金)
1. `v087_glm`: V0.8.7-GLM研究候选(广度门控0.58 + 仅日内止损触价)
2. `v08_verified_ref`: 冻结V0.8基线参照
3. `v087_glm_random_2026101401`: 候选引擎固定种子随机对照1
4. `v087_glm_random_2026101402`: 候选引擎固定种子随机对照2
5. `v087_glm_random_2026101403`: 候选引擎固定种子随机对照3
6. `cash`: 100%现金对照(无交易)

## 2. 交易摩擦与微观规则
- 初始本金：各账户30,000元，整手(100股)交易。
- 佣金万2.5(最低5元)，印花税千0.5(卖出单边)，过户费万0.01。
- 基准滑点万5；日内触价单附加10 bps惩罚滑点。
- 严格T+1锁定，跌停不可卖，涨停不可买，停牌拦截。
- 广度门控：信号日 breadth5 < 0.58 时拒绝全部次日新买单（空位亦不豁免；卖出与既有持仓退出不受影响）。
- 资产负债守恒：每日收盘严格满足 |NAV - (Cash + Receivables + Market_Value)| < 1e-6。

## 3. 封存与合规铁律
- 仅在收盘日15:30之后、次交易日09:15之前封存当日数据；创建日(2026-10-25)之前的信号日一律不得封存。
- 严禁任何形式的未来信息泄露与历史回补。
- 本批次为研究候选观察，不代表 V0.8 已采用版本变更；不发送真实券商订单。
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
        'made_by_checkpoint': 'GLM',
        'operational_sources': {
            name: digest(ROOT / name) for name in ['run_v087_glm_day.py', 'update_paper_data.py']
            if (ROOT / name).is_file()
        },
        'glm_sources': {
            name: digest(ROOT / name) for name in ['model_v087_glm.py', 'engine_core.py', 'engine_intraday.py']
            if (ROOT / name).is_file()
        },
    }
    write_new(session / 'runtime-manifest.json', manifest)
    init = {
        'version': 'V0.8.7-GLM forward parallel paper batch',
        'made_by_checkpoint': 'GLM',
        'champion_model': 'v087_glm (BreadthGate 0.58 + SL-touch-only intraday + close-based profit exits)',
        'baseline_model': 'v08_verified_ref (Frozen V0.8 Baseline reference)',
        'accepted_research_run': str((ROOT / 'results/v087-glm-v1').resolve()),
        'damping_policy': 'OFF per pre-registered REJECT_DAMPING verdict 2026-10-25',
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
            str((ROOT / 'results/v086-parallel-paper-20261023-v1').resolve()),
        ],
        'independent_batch': True,
        'no_real_brokerage_orders': True,
        'timestamp_limit': 'Local clock and hashes are not an external timestamp authority',
    }
    write_new(session / 'init.json', init)
    write_status(session, init, [])
    print(f'INITIALIZED {session} ; 6 independent accounts; 0 sealed signals; no performance yet')


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
    for name, expected in manifest.get('glm_sources', {}).items():
        if (ROOT / name).is_file() and digest(ROOT / name) != expected:
            raise ValueError('V0.8.7-GLM source changed: ' + name)
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


def build_engine(spec: dict, frames, calendar, signals, events, cfg, vol, yz_vol, limits, bm):
    """Assemble the ModularReplayEngine for one account spec (GLM batch)."""
    from engine_core import BaselineRanker, BaselineRegimeStopping, ModularReplayEngine
    from model_v087_glm import BreadthGatekeeperGLM, TouchModeStoppingGLM

    policy = spec['policy']
    if policy == 'cash':
        rep = ModularReplayEngine(
            frames=frames, calendar=calendar, signals=signals, dividends=events, config=cfg,
            vol_map=vol, yz_vol_map=yz_vol, daily_limits=limits, breadth_lookup=bm,
            enable_cash_overlay=False, stopping=BaselineRegimeStopping(),
        )
    elif policy == 'v087_glm':
        rep = ModularReplayEngine(
            frames=frames, calendar=calendar, signals=signals, dividends=events, config=cfg,
            vol_map=vol, yz_vol_map=yz_vol, daily_limits=limits, breadth_lookup=bm,
            ranker=BaselineRanker(),
            gatekeeper=BreadthGatekeeperGLM(breadth_floor=spec.get('breadth_floor', 0.58)),
            stopping=TouchModeStoppingGLM(intraday_mode=spec.get('intraday_mode', 'sl')),
            enable_cash_overlay=False, enable_intraday_touch=True, turnover_hurdle=0.0,
            enable_turnover_damping=False,
        )
    else:  # v08_verified_ref: frozen baseline, identical to the V0.8.6 batch's baseline branch
        rep = ModularReplayEngine(
            frames=frames, calendar=calendar, signals=signals, dividends=events, config=cfg,
            vol_map=vol, yz_vol_map=yz_vol, daily_limits=limits, breadth_lookup=bm,
            enable_cash_overlay=False, stopping=BaselineRegimeStopping(),
            enable_intraday_touch=False, turnover_hurdle=0.0, enable_turnover_damping=False,
        )
    return rep


def simulate(frames, bars, calendar, signals, events, spec, modules, panel):
    from exploration_features import breadth_days
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

    rep = build_engine(
        spec, frames, calendar, signals, events,
        Config(capital=30000, order_budget=7500, max_positions=5, max_exposure=0.90, hold_days=10),
        vol, yz_vol, limits, bm,
    )
    rep.bars = bars
    rep.gate_decisions = []
    rep.run()
    # One summary gate decision per day-run so account-comparison shows breadth-gate
    # rejections (gating_abstentions) alongside proposed orders.
    rep.gate_decisions = [{'proposed_orders': rep.pending_buys, 'cancelled_orders': rep.gating_abstentions}]
    return rep


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
        'runtime_manifest_sha256': init['runtime_manifest_sha256'], 'independent_batch': True,
        'made_by_checkpoint': 'GLM'})
    staging.rename(destination)
    write_status(session, init, prior + [(destination, read_json(destination / 'record.json'))])
    print('SEALED', cutoff, '->', following, '; 6 accounts; records', len(prior) + 1)
    print(pd.DataFrame(result_rows).to_string(index=False))


def write_status(session: Path, init: dict, prior: list):
    session = Path(session)
    latest = prior[-1] if prior else None
    text = '# V0.8.7-GLM 研究候选：独立前瞻影子模拟（GLM 检查点建立）\n\n'
    text += 'V0.8.7-GLM研究候选（广度门控0.58 + 仅日内止损触价 + 收盘基准止盈），与冻结V0.8基线参照、同引擎随机对照并行。6账户各3万元。\n\n'
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
    text += '\n历史V0.8、V0.8.1、V0.8.6批次记录原样隔离封存，本批次独立从头建账，不替换任何既有账户。\n'
    text += '\n窗口：收盘日15:30后至次交易日09:15前。20交易日流程检查、60交易日阶段复核。\n'
    text += '\n研究候选身份：通过历史预注册验收（results/v087-glm-v1）不等于未来证明；本批次用于前瞻观察。\n'
    text += '\n[冻结协议](protocol.md)；[初始化配置](init.json)；[运行依赖清单](runtime-manifest.json)。不发送真实券商订单。\n'
    (session / 'status.md').write_text(text, encoding='utf-8')
    if latest:
        table = pd.read_csv(latest[0] / 'account-comparison.csv').to_html(index=False)
    else:
        table = '<table><tr><th>账户</th><th>初始虚拟资金</th><th>状态</th></tr>' + ''.join(
            f'<tr><td>{html.escape(s["display"])}</td><td>30,000元</td><td>等待首次封存；无业绩</td></tr>'
            for s in init['accounts'].values()) + '</table>'
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>V0.8.7-GLM前瞻影子模拟</title>'
    page += '<style>body{font:16px/1.7 system-ui,"Microsoft YaHei";max-width:1250px;margin:32px auto;padding:0 20px;color:#243047;background:#f7f9fc}table{border-collapse:collapse;background:white;width:100%}td,th{padding:10px;border:1px solid #d7e0e8;text-align:left}.note{background:#fff1d6;padding:16px;border-left:4px solid #b87d10}</style>'
    page += '<h1>V0.8.7-GLM 研究候选：前瞻影子模拟（GLM检查点）</h1>'
    page += f'<p><b>已封存 {len(prior)} 个信号日</b> · 6账户独立各3万元本金</p>'
    page += '<p class="note">只有事先封存并实际发生的模拟成交才计入未来成绩；初始化和空账户均不证明盈利。研究候选通过历史验收不构成未来收益承诺。</p>'
    page += table + '<p>历史批次完全隔离。今日信号须在收盘后完整数据核验封存。</p>'
    page += '<p><a href="protocol.md">冻结协议</a> · <a href="status.md">完整状态</a> · <a href="init.json">账户配置</a></p></html>'
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
