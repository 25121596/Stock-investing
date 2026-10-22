"""One bounded daily operation for the frozen parallel paper batch."""
from __future__ import annotations

import argparse
from datetime import datetime, time, timezone
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import v081_parallel_paper as paper

ROOT = Path(__file__).resolve().parent
SIMULATOR = ROOT.parent / 'ashare-simulator'
BRIDGE = SIMULATOR / 'v081-parallel'


def cached_schedule(records):
    sources = [folder / 'input-snapshot/exchange-calendar.csv' for folder, _ in reversed(records)]
    sources += [ROOT / 'results/v08-paper-forward-v3/days/2026-10-20/input-snapshot/exchange-calendar.csv']
    for path in sources:
        if path.is_file():
            table = pd.read_csv(path, dtype=str)
            if {'calendar_date', 'is_trading_day'}.issubset(table.columns):
                return table
    raise ValueError('No integrity-checked exchange calendar is available')


def readiness(init, records, now):
    if now.tzinfo is None:
        raise ValueError('Timezone-aware clock required')
    local = now.astimezone(paper.CST)
    today = local.date().isoformat()
    if today[:4] != str(init['model_year']):
        raise ValueError('Frozen annual models cannot continue across years')
    table = cached_schedule(records)
    row = table[table.calendar_date.eq(today)]
    if len(row) != 1:
        raise ValueError('Cached exchange calendar does not cover today uniquely')
    if str(row.is_trading_day.iloc[0]) != '1':
        return {'status': 'non_trading_day', 'date': today}
    if local.time() < time(15, 30):
        return {'status': 'waiting_for_close', 'date': today, 'earliest': today + ' 15:30 Asia/Shanghai'}
    if records and records[-1][1]['signal_date'] != today and records[-1][1]['execution_date'] != today:
        raise ValueError('A prospective signal day was missed; never backfill this batch')
    schedule = table[table.is_trading_day.eq('1')].calendar_date.tolist()
    previous = records[-2][1] if records and records[-1][1]['signal_date'] == today and len(records) > 1 else (
        records[-1][1] if records and records[-1][1]['signal_date'] != today else None)
    following = paper.validate_window(today, schedule, now, init['prospective_start_not_before'], previous)
    return {'status': 'ready', 'date': today, 'next_execution_date': following,
            'already_sealed': bool(records and records[-1][1]['signal_date'] == today)}


def run_child(command, cwd):
    result = subprocess.run([str(x) for x in command], cwd=str(cwd), check=False, timeout=3600)
    if result.returncode:
        raise ValueError('Operational subprocess failed with exit code ' + str(result.returncode))


def run_day(session, bridge, *, check_only=False):
    session, bridge = Path(session).resolve(), Path(bridge).resolve()
    init, records, _ = paper.verify_records(session)
    state = readiness(init, records, datetime.now(timezone.utc))
    print(json.dumps(state, ensure_ascii=False), flush=True)
    if check_only or state['status'] != 'ready':
        return state
    # A single immutable day may be resumed only at the gateway stage. Data
    # already sealed on this day are reused byte-for-byte, never re-downloaded.
    lock = session / '.daily-operation-lock.json'
    with lock.open('x', encoding='utf-8') as stream:
        json.dump({'started_utc': datetime.now(timezone.utc).isoformat(), 'signal_date': state['date']}, stream)
    operation = {'started_utc': datetime.now(timezone.utc).isoformat(), **state, 'steps': []}
    try:
        if state['already_sealed']:
            dataset = records[-1][0] / 'input-snapshot'
        else:
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            dataset = ROOT / 'data' / f'v081-paper-{state["date"]}-{stamp}'
            run_child([sys.executable, ROOT / 'update_paper_data.py', '--end', state['date'], '--output', dataset], ROOT)
            operation['steps'].append('download_and_validate')
            manifest = paper.read_json(dataset / 'manifest.json')
            if manifest.get('failures') or not manifest.get('complete') or manifest['end'] != state['date']:
                raise ValueError('Daily data snapshot is incomplete; do not seal or remove stocks')
            run_child([sys.executable, ROOT / 'v081_parallel_paper.py', 'record', '--session', session, '--dataset', dataset], ROOT)
            operation['steps'].append('seal_all_9_accounts')
        run_child([sys.executable, ROOT / 'verify_v081_parallel_paper.py', '--session', session], ROOT)
        operation['steps'].append('verify_sealed_batch')
        simulator_python = SIMULATOR / '.venv/Scripts/python.exe'
        gateway = SIMULATOR / 'v081_parallel_gateway.py'
        run_child([simulator_python, gateway, 'sync', '--session', session, '--bridge', bridge], SIMULATOR)
        operation['steps'].append('import_frozen_plans')
        run_child([simulator_python, gateway, 'execute', '--session', session, '--bridge', bridge, '--dataset', dataset], SIMULATOR)
        operation['steps'].append('rqalpha_settle_completed_sessions_or_wait')
        run_child([simulator_python, gateway, 'sync', '--session', session, '--bridge', bridge], SIMULATOR)
        operation['steps'].append('reconcile_external_accounts_before_next_orders')
        run_child([simulator_python, gateway, 'status', '--session', session, '--bridge', bridge], SIMULATOR)
        operation['status'] = 'completed'
        operation['dataset'] = str(dataset)
        return operation
    except Exception as exc:
        operation['status'] = 'blocked'
        operation['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        operation['finished_utc'] = datetime.now(timezone.utc).isoformat()
        logs = session / 'operations'
        logs.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        paper.write_new(logs / f'{stamp}.json', operation)
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, default=paper.SESSION)
    parser.add_argument('--bridge', type=Path, default=BRIDGE)
    parser.add_argument('--check-only', action='store_true', help='Verify readiness without fetching or recording')
    args = parser.parse_args()
    try:
        result = run_day(args.session, args.bridge, check_only=args.check_only)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except (ValueError, FileNotFoundError, FileExistsError, subprocess.TimeoutExpired) as exc:
        parser.exit(2, f'BLOCKED: {exc}\n')


if __name__ == '__main__':
    main()
