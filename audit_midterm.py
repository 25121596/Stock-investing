"""Independently check exported fills and accounts, without rerunning policies."""
from pathlib import Path
import argparse
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/midterm-v1'
DATA = ROOT / 'data/forward-2026-10-13'
FEE_KEYS = ('commission', 'transfer', 'stamp', 'slippage')


def optional(path):
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def rounding(value):
    return float(Decimal(str(value)).quantize(Decimal('.01'), rounding=ROUND_HALF_UP))


def main():
    global OUT, DATA
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--dataset', type=Path, default=DATA)
    args = parser.parse_args()
    OUT, DATA = args.output.resolve(), args.dataset.resolve()
    payload = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    calendar = pd.read_csv(DATA / 'calendar.csv').calendar_date.tolist()
    index = {day: i for i, day in enumerate(calendar)}
    raw = {}
    rows = []
    for case in payload['results']:
        folder = OUT / case['case']
        fills, trades = optional(folder / 'fills.csv'), optional(folder / 'trades.csv')
        holdings, equity = optional(folder / 'holdings.csv'), optional(folder / 'equity.csv')
        errors = []
        if len(fills):
            config = case['config']
            expected_costs = 0.
            for fill in fills.itertuples():
                if fill.code not in raw:
                    raw[fill.code] = pd.read_csv(DATA / f'{fill.code}.csv').set_index('date').to_dict('index')
                source = raw[fill.code][fill.date]
                if abs(source['open'] - fill.price) > 1e-8:
                    errors.append('fill_not_raw_open')
                if abs(fill.shares * fill.price - fill.notional) > .001:
                    errors.append('notional')
                # Reconstruct using raw source open rather than a CSV parser's
                # rounded float notional; half-cent boundaries retain the exact
                # source multiplication used by the exported money convention.
                notional = fill.shares * source['open']
                expected = {'commission': rounding(max(config['min_commission'], notional * config['commission'])),
                    'transfer': rounding(notional * config['transfer_fee']),
                    'stamp': rounding(notional * config['stamp_tax']) if fill.side == 'sell' else 0.,
                    'slippage': rounding(notional * config['slippage'])}
                if any(abs(getattr(fill, key) - amount) > 1e-8 for key, amount in expected.items()):
                    errors.append('fee')
                expected_costs += sum(expected.values())
                if fill.side == 'buy':
                    if payload['metadata'].get('restricted_small_account_access'):
                        from accessibility import allowed_buy
                        if not allowed_buy(fill.code, fill.date):
                            errors.append('inaccessible_retirement_buy')
                    if fill.shares % 100:
                        errors.append('buy_not_whole_lot')
                    signal_day = calendar[index[fill.date] - 1]
                    limit = rounding(raw[fill.code][signal_day]['close'] * (1 + config['max_buy_gap']))
                    if fill.price > limit + .0001:
                        errors.append('buy_gap_limit')
            if abs(expected_costs - case['execution_costs']) > 1e-6:
                errors.append('summary_execution_costs')
        for trade in trades.itertuples():
            same = fills[fills.trade_id.eq(trade.trade_id)]
            buys, sells = same[same.side.eq('buy')], same[same.side.eq('sell')]
            proceeds = sells.notional.sum() - sells[list(FEE_KEYS)].to_numpy().sum()
            spent = buys.notional.sum() + buys[list(FEE_KEYS)].to_numpy().sum()
            if abs(proceeds + trade.net_dividends - spent - trade.net_pnl) > 1e-6:
                errors.append('trade_pnl')
            if index[trade.entry_date] != index[trade.signal_date] + 1:
                errors.append('next_session_entry')
            duration = index[trade.exit_date] - index[trade.entry_date]
            if duration < 1:
                errors.append('t_plus_1')
            if trade.exit_reason == 'fixed_horizon' and duration < case['hold_days']:
                errors.append('early_fixed_exit')
        closed = trades.net_pnl.sum() if len(trades) else 0.
        marked = holdings.marked_pnl.sum() if len(holdings) else 0.
        difference = equity.equity.iloc[-1] - 30000. - closed - marked
        if abs(difference) > 1e-6:
            errors.append('independent_nav_reconcile')
        expected_days = calendar[index[equity.date.iloc[0]]:index[equity.date.iloc[-1]] + 1]
        if equity.date.tolist() != expected_days:
            errors.append('equity_calendar')
        rows.append({'case': case['case'], 'fills': len(fills), 'trades': len(trades),
            'reconciliation_error': float(difference), 'errors': errors,
            'source_quality_warnings': case['quality_warning_count']})
    hash_check = {}
    for name, expected in payload['hashes'].items():
        hash_check[name] = hashlib.sha256((OUT / 'sources' / name).read_bytes()).hexdigest() == expected
    result = {'scope': f'All {len(rows)} exported scenarios; execution/account checks do not certify source data completeness',
        'cases': rows, 'source_snapshot_hashes_match': hash_check,
        'all_checks_passed': all(not item['errors'] for item in rows) and all(hash_check.values()),
        'derived_metric_note': 'cost_per_closed_trade uses total costs including open-position buys divided by closed trades; it is not mean closed-trade cost'}
    (OUT / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'all_checks_passed': result['all_checks_passed'], 'cases': len(rows),
        'fills': sum(item['fills'] for item in rows), 'closed_trades': sum(item['trades'] for item in rows),
        'max_reconciliation_error': max(abs(item['reconciliation_error']) for item in rows),
        'failures': [item for item in rows if item['errors']]}, ensure_ascii=False, indent=2))
    if not result['all_checks_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
