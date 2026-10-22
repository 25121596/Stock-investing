"""Parity of the prospective engine with already-viewed historical references."""
import argparse
import json
from pathlib import Path

import pandas as pd
import arena
import engine
import run_v09_rl_2022 as previous
import v081_parallel_paper as paper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    out.mkdir(exist_ok=False)
    source = Path(__file__).resolve().parent / 'results/v09-supervised-filter-20261021-v1'
    _, frames, calendar, raw = arena.load(arena.DATASET)
    events, _ = arena.canonicalize_events(raw)
    bars = arena.prepare_bars(frames)
    panel = pd.read_csv(previous.FEATURES, dtype={'date': str, 'code': str},
                        usecols=['date', 'code', 'adj_ret5', 'market_ret5', 'ret_vol20'], float_precision='round_trip')
    two = pd.read_csv(previous.PRED2, dtype={'date': str, 'code': str},
                      usecols=['date', 'code', 'pred_net_return_v2'], float_precision='round_trip')
    eight = pd.read_csv(previous.PRED8, dtype={'date': str, 'code': str},
                        usecols=['date', 'code', 'pred_score_v08'], float_precision='round_trip')
    signals = previous.prepare(panel, two, eight, [d for d in calendar if '2022-01-01' <= d <= '2026-10-13'])[0]
    records = []
    for period, start, end in [('2023-2025', '2023-01-01', '2025-12-31'), ('2026', '2026-01-01', '2026-10-13')]:
        days = [d for d in calendar if start <= d <= end]
        chosen = signals[signals.date.between(start, end)]
        context = panel[panel.date.between(start, end)]
        for name, refname in [('v08_verified', 'baseline'), ('v081_breadth58', 'fixed_breadth')]:
            rep = paper.simulate(frames, bars, days, chosen, events, paper.accounts()[name], [engine], context)
            summary = rep.summary()
            audit = arena.audit_ledger(rep, bars, events)
            folder = out / f'{period}_{name}'
            arena.save_case(folder, rep, summary, chosen)
            pd.DataFrame(rep.gate_decisions).to_csv(folder / 'gate-decisions.csv', index=False)
            reference = source / f'{period}_{refname}'
            for item in ['equity', 'fills', 'trades', 'actions', 'holdings', 'warnings', 'review_warnings', 'signals']:
                def read(path):
                    try:
                        return pd.read_csv(path, float_precision='round_trip')
                    except pd.errors.EmptyDataError:
                        return pd.DataFrame()
                pd.testing.assert_frame_equal(read(folder / f'{item}.csv'), read(reference / f'{item}.csv'),
                                              check_dtype=False, check_exact=True)
            expected = json.loads((reference / 'summary.json').read_text(encoding='utf-8'))
            for key in ['total_return', 'final_equity', 'max_drawdown', 'execution_costs', 'closed_trades', 'win_rate', 'rejected', 'config']:
                assert summary[key] == expected[key], (period, name, key)
            records.append({'period': period, 'account': name, 'reference': str(reference),
                            'passed': True, 'ledger_audit': audit, 'total_return': summary['total_return'],
                            'max_drawdown': summary['max_drawdown']})
    arena.write_json(out / 'parity.json', {'passed': True, 'evidence_class': 'Previously viewed history; engineering parity only',
        'not_prospective_performance': True, 'records': records,
        'recorder_sha256': paper.digest(Path(paper.__file__)), 'validator_sha256': paper.digest(Path(__file__))})
    print('PASSED: 4 historical accounts match prior V0.8/fixed-breadth ledger exports exactly; no future evidence claimed')


if __name__ == '__main__':
    main()
