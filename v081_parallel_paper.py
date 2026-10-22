"""Frozen V0.8 / breadth-58 parallel paper batch; no real brokerage orders."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import html
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import v08_paper_forward_v3 as legacy

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
SESSION = ROOT / 'results/v081-parallel-paper-20261022-v1'
FREEZE = legacy.FREEZE
CST = legacy.CST
SEEDS = legacy.SEEDS
THRESHOLD = .58
digest, read_json, normal, object_hash = legacy.digest, legacy.read_json, legacy.normal, legacy.object_hash
validate_window = legacy.validate_window
load_dataset = legacy.load_dataset
account_state, ledger = legacy.account_state, legacy.ledger
runtime_versions = legacy.runtime_versions


def write_new(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def accounts():
    result = {}
    for key, enabled, display in [('v08_verified', False, 'V0.8基线'),
                                  ('v081_breadth58', True, 'V0.8.1广度候选（未采用）')]:
        result[key] = {'policy': key, 'gate_breadth58': enabled, 'seed': None, 'display': display}
        for seed in SEEDS:
            result[f'{key}_random_{seed}'] = {'policy': key, 'gate_breadth58': enabled,
                                             'seed': seed, 'display': f'{display}随机对照{seed}'}
    result['cash'] = {'policy': 'cash', 'gate_breadth58': False, 'seed': None, 'display': '现金对照'}
    return result


def file_hashes(folder):
    return {p.relative_to(folder).as_posix(): digest(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def protected_hashes():
    old = ROOT / 'results/v08-paper-forward-v3'
    paths = [old / 'init.json', old / 'runtime/freeze.json',
             ROOT.parent / 'ashare-simulator/v08-paper/gateway-freeze.json',
             ROOT / 'v08_paper_forward_v3.py']
    for folder in sorted((old / 'days').iterdir()):
        if folder.is_dir() and not folder.name.startswith('.'):
            record = read_json(folder / 'record.json')
            paths.append(folder / 'record.json')
            for name, expected in record['files'].items():
                if digest(folder / name) != expected:
                    raise ValueError('Original sealed artifact already differs: ' + name)
                paths.append(folder / name)
    return {str(p.resolve()): digest(p) for p in paths}


def initialize(session):
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
    shutil.copyfile(ROOT / 'docs/v081-parallel-paper-protocol.md', session / 'protocol.md')
    manifest = {'runtime': file_hashes(session / 'runtime'), 'runner_sha256': digest(__file__),
                'legacy_helper_sha256': digest(ROOT / 'v08_paper_forward_v3.py'),
                'protocol_sha256': digest(session / 'protocol.md'),
                'operational_sources': {name: digest(ROOT / name) for name in
                    ['run_v081_parallel_day.py', 'update_paper_data.py', 'verify_v081_parallel_paper.py']}}
    write_new(session / 'runtime-manifest.json', manifest)
    init = {'version': 'V0.8.1 parallel observation candidate', 'current_adopted': 'V0.8',
            'created_utc': now.isoformat(), 'created_beijing': now.astimezone(CST).isoformat(),
            'prospective_start_not_before': now.astimezone(CST).date().isoformat(),
            'accounts': accounts(), 'initial_cash_per_account': 30000, 'model_year': 2026,
            'gate_threshold': THRESHOLD, 'random_seeds': list(SEEDS),
            'runtime_manifest_sha256': digest(session / 'runtime-manifest.json'),
            'runtime_versions': runtime_versions(), 'protected_hashes': protected,
            'record_deadline': '15:30 signal close through next exchange session 09:15 Asia/Shanghai',
            'prior_batch': str(ROOT / 'results/v08-paper-forward-v3'),
            'prior_batch_gap': '2026-10-21 signal not sealed before its deadline; no backfill or balance carry-over',
            'independent_batch': True, 'no_real_brokerage_orders': True,
            'timestamp_limit': 'Local clock and hashes are not an external timestamp authority'}
    write_new(session / 'init.json', init)
    write_status(session, init, [])
    print('INITIALIZED', session, '; 9 independent accounts; 0 sealed signals; no performance yet')


def verify_runtime(session, init):
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
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen daily operational source changed: ' + name)
    if file_hashes(session / 'runtime') != manifest['runtime']:
        raise ValueError('Frozen runtime source/model content changed')
    legacy.verify_freeze(session / 'runtime')
    if init['runtime_versions'] != runtime_versions():
        raise ValueError('Frozen inference library versions changed')
    if init['accounts'] != accounts() or init['gate_threshold'] != THRESHOLD or init['initial_cash_per_account'] != 30000:
        raise ValueError('Account/rule identity differs from frozen recorder')
    for path, expected in init['protected_hashes'].items():
        if digest(Path(path)) != expected:
            raise ValueError('Protected original artifact changed: ' + path)


def verify_records(session):
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


def load_runtime(session):
    return legacy.load_runtime(session / 'runtime')


def make_signals(panel, breadth):
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


def apply_candidate_gate(orders, breadth, enabled):
    if orders and not np.isfinite(breadth):
        raise ValueError('Cannot gate pending orders without valid prior-close breadth')
    cancel = bool(enabled and orders and breadth < THRESHOLD)
    return ([] if cancel else [dict(o) for o in orders]), int(cancel)


def simulate(frames, bars, calendar, signals, events, spec, modules, panel):
    from v08_engine_verified import VerifiedV08Replay
    from exploration_features import breadth_days
    engine = modules[0]
    if panel.duplicated(['date', 'code']).any() or set(panel.date) != set(calendar):
        raise ValueError('Frozen context must cover every replay day once per stock')
    b = breadth_days(panel)
    bm = b.set_index('date').breadth5.to_dict()
    pm = b.set_index('date').pool_size.to_dict()
    limits = {d: 0 if pm[d] < 100 or not np.isfinite(v) or v < .45 else (1 if v < .55 else 2)
              for d, v in bm.items()}
    vol = panel.set_index(['date', 'code']).ret_vol20.to_dict()

    class ParallelReplay(VerifiedV08Replay):
        def gate_close(self, day):
            before = [dict(o) for o in self.pending_buys]
            after, action = apply_candidate_gate(before, bm[day], spec['gate_breadth58'])
            self.pending_buys = after
            self.gate_decisions.append({'signal_date': day, 'breadth': float(bm[day]) if np.isfinite(bm[day]) else None,
                                       'threshold': THRESHOLD, 'action': action, 'proposed_orders': before,
                                       'accepted_orders': after, 'cancelled_orders': len(before) - len(after)})

        def corporate_actions(self, day):
            if self.equity:
                self.gate_close(self.equity[-1]['date'])
            return super().corporate_actions(day)

        def run(self):
            self.gate_decisions = []
            super().run()
            self.gate_close(self.calendar[-1])
            return self

    cfg = engine.Config(capital=30000, order_budget=7500, max_positions=5, max_exposure=.90, hold_days=10)
    rep = ParallelReplay({}, calendar, signals, events, cfg, vol_map=vol, daily_limits=limits,
                         breadth_lookup=bm, enable_cash_overlay=False)
    rep.frames, rep.bars = frames, bars
    return rep.run()


def record_day(session, dataset):
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
        if rep.warnings or rep.review_warnings:
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
                 'reason': 'stop_close_next_open' if p.stop_pending else 'fixed_horizon'}
                for p in rep.positions.values() if p.shares > 0 and legacy.v08_sell_due(rep, p, days)]})
        for key, rows in ledger(rep).items():
            pd.DataFrame(rows).to_csv(account / f'{key}.csv', index=False)
        pd.DataFrame(rep.gate_decisions).to_csv(account / 'gate-decisions.csv', index=False)
        selected = signals[name].copy()
        selected['account'] = name
        all_signals.append(selected)
        gate = rep.gate_decisions[-1]
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


def write_status(session, init, prior):
    latest = prior[-1] if prior else None
    text = '# V0.8 / V0.8.1广度候选：独立前瞻模拟\n\n'
    text += f'当前采用V0.8；候选只增加58%广度过滤。9账户各3万元，模型与费用相同。\n\n已封存信号日：**{len(prior)}**。'
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
    text += '\n旧批次10月20日记录原样保留，10月21日漏封不回补，本批次不接续其收益。\n'
    text += '\n窗口：收盘日15:30后至次交易日09:15前。20交易日检查流程、60交易日阶段复核，样本不足则延长；这些不是盈利认证。\n'
    text += '\n[冻结协议](protocol.md)；[初始化与完整账户配置](init.json)；[运行依赖哈希](runtime-manifest.json)。不发送真实券商订单。\n'
    (session / 'status.md').write_text(text, encoding='utf-8')
    table = ''
    if latest:
        table = pd.read_csv(latest[0] / 'account-comparison.csv').to_html(index=False)
    else:
        table = '<table><tr><th>账户</th><th>初始虚拟资金</th><th>状态</th></tr>' + ''.join(
            f'<tr><td>{html.escape(s["display"])}</td><td>30,000元</td><td>等待首次封存；无业绩</td></tr>'
            for s in init['accounts'].values()) + '</table>'
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>V0.8与广度候选前瞻模拟</title>'
    page += '<style>body{font:16px/1.7 system-ui,"Microsoft YaHei";max-width:1250px;margin:32px auto;padding:0 20px;color:#243047;background:#f7f9fc}table{border-collapse:collapse;background:white;width:100%}td,th{padding:10px;border:1px solid #d7e0e8;text-align:left}.note{background:#fff1d6;padding:16px;border-left:4px solid #b87d10}</style>'
    page += '<h1>V0.8与58%广度候选：前瞻模拟</h1>'
    page += f'<p><b>已封存 {len(prior)} 个信号日</b> · 当前采用V0.8 · 各账户独立3万元</p>'
    page += '<p class="note">候选尚未采用。只有事先封存并实际发生的模拟成交才计入未来成绩；初始化、历史演练和空账户都不证明盈利。</p>'
    page += table + '<p>旧批次保留；漏记日期不回补。今日信号须在收盘数据核验后封存，随后由RQAlpha独立撮合核验。</p>'
    page += '<p><a href="protocol.md">冻结协议</a> · <a href="status.md">完整状态</a> · <a href="init.json">账户配置</a> · <a href="../../../ashare-simulator/v081-parallel/status.html">RQAlpha接入状态</a></p></html>'
    (session / 'status.html').write_text(page, encoding='utf-8')


def status(session):
    init, records, _ = verify_records(session)
    write_status(Path(session), init, records)
    print('VERIFIED', len(records), 'sealed signal days;', len(init['accounts']), 'independent accounts')
    if records:
        print('LATEST', records[-1][1]['signal_date'])


def main():
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
