"""Fixed-protocol label, execution, saved-inference and matched-random checks."""
from datetime import datetime, timezone
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine import Config
from run import load
from validate_expanded import prepare_bars
from corporate_events import canonicalize_events
from exploration_features import breadth_days
from run_exploration import stable_random, matched_candidates
from midterm_labels import relabel_panel
from v08_engine_verified import VerifiedV08Replay
from v08_inference_verified import inference_panel, predict

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / 'results/v08-codex-rebuild-20261020-v1'
OUT = ROOT / 'results/v08-codex-certified-20261020-v1'


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=False)
    seeds = list(range(2026102001, 2026102101))
    write(OUT / 'protocol.json', {'started_utc': datetime.now(timezone.utc).isoformat(), 'seeds': seeds,
        'pool': 'Same daily eligible pool, matched dates and candidate counts; no replacement',
        'cash_interest': False, 'random_cases_per_period': 100,
        'quality_failure': 'Keep failed draws; do not treat conditional valid subset as a formal p-value',
        'historical_selection_bias_not_removed': True, 'extra_embargo_sessions': 0})
    manifest, frames, calendar, dividends = load(ROOT / 'data/forward-2026-10-13')
    events, event_audit = canonicalize_events(dividends)
    bars = prepare_bars(frames)
    panel = pd.read_csv(ROOT / 'results/midterm-access-v3/features.csv', dtype={'date': str, 'code': str})
    labels = pd.read_csv(ROOT / 'results/midterm-access-v3/panel-h10.csv', dtype={'date': str, 'code': str, 'label_end': str})
    # Preserve both labeled and unlabelable rows, sampled equally by year.
    sample = labels.groupby(labels.date.str[:4], group_keys=False).sample(n=25, random_state=20261020)
    rebuilt = relabel_panel(sample, frames, calendar, dividends, Config(hold_days=10))
    mismatch = []
    for idx in sample.index:
        old, new = sample.loc[idx], rebuilt.loc[idx]
        a, b = old.net_return, new.net_return
        if not ((pd.isna(a) and pd.isna(b)) or (pd.notna(a) and pd.notna(b) and abs(a-b) < 1e-10)):
            mismatch.append({'date': old.date, 'code': old.code, 'old': None if pd.isna(a) else a,
                             'new': None if pd.isna(b) else b})
        if str(old.label_end) != str(new.label_end) and not (pd.isna(old.label_end) and pd.isna(new.label_end)):
            mismatch.append({'date': old.date, 'code': old.code, 'issue': 'label_end_mismatch'})
    write(OUT / 'label-audit.json', {'sampled': len(sample), 'mismatches': mismatch,
        'scope': 'Seeded sample across all years, not an exhaustive raw-label audit'})
    fresh = inference_panel(frames, calendar, ROOT / 'results/exploration-paper-frozen-v1/sources/ml_features-frozen.py')
    fresh = predict(fresh, BUNDLE / 'v2/2026.txt', BUNDLE / 'v08/2026.txt')
    p2 = pd.read_csv(BUNDLE / 'v2/predictions-h10-v2.csv', dtype={'date': str, 'code': str})
    p8 = pd.read_csv(BUNDLE / 'v08/predictions-h10-v08.csv', dtype={'date': str, 'code': str})
    original = p2.merge(p8, on=['date', 'code'], validate='one_to_one')
    parity = fresh.merge(original[['date', 'code', 'pred_net_return_v2', 'pred_score_v08']], on=['date', 'code'], suffixes=('_new', '_old'), validate='one_to_one')
    error = {key: float((parity[key+'_new'] - parity[key+'_old']).abs().max()) for key in ['pred_net_return_v2', 'pred_score_v08']}
    covered = len(parity) == len(fresh) == len(original[original.date.eq(calendar[-1])])
    write(OUT / 'inference-audit.json', {'date': calendar[-1], 'candidates': len(fresh),
        'coverage_equal': covered, 'maximum_errors': error})
    if mismatch or not covered or max(error.values()) > 1e-10:
        raise ValueError('Label or fresh-feature inference audit failed; do not freeze')
    breadth = breadth_days(panel)
    bm, pm = breadth.set_index('date').breadth5.to_dict(), breadth.set_index('date').pool_size.to_dict()
    limits = {d: 0 if pm[d] < 100 or b < .45 else (1 if b < .55 else 2) for d,b in bm.items()}
    sig = original[original.pred_net_return_v2.ge(.005) & original.date.isin({d for d,n in limits.items() if n})]
    sig = sig[['date', 'code', 'pred_score_v08']].rename(columns={'pred_score_v08': 'score'})
    vol = panel.set_index(['date', 'code']).ret_vol20.to_dict()
    cfg = Config(capital=30000, order_budget=7500, max_positions=5, max_exposure=.9, hold_days=10)
    results, failed, audits = [], [], []
    for period, start, end in [('2022-2025', '2022-01-01', '2025-12-31'), ('2026', '2026-01-01', '2026-10-13')]:
        days = [d for d in calendar if start <= d <= end]
        selected = sig[sig.date.between(start,end)]
        pool = panel[panel.date.between(start,end)]
        choices = [('actual', selected, cfg), ('slippage_x2', selected, replace(cfg, slippage=.001)),
                   ('cash', selected.iloc[:0], cfg)]
        for seed in seeds:
            choices.append((f'random_{seed}', matched_candidates(selected, stable_random(pool,seed)), cfg))
        for num, (name, signals, config) in enumerate(choices):
            try:
                rep = VerifiedV08Replay({}, days, signals, events, config, vol_map=vol,
                    daily_limits=limits, breadth_lookup=bm, enable_cash_overlay=False)
                rep.frames, rep.bars = frames, bars
                rep.run()
                summary = rep.summary()
                errors = []
                trade_lookup = {t['trade_id']: t for t in rep.trades}
                for f in rep.fills:
                    bar = bars[f['code']][f['date']]
                    if abs(f['price'] - float(bar['open'])) > 1e-10: errors.append('not_open_price')
                    fees = config.fees(f['shares'] * f['price'], sell=f['side']=='sell')
                    if any(abs(f[k] - fees[k]) > 1e-9 for k in fees): errors.append('fee_mismatch')
                    if f['side']=='buy' and f['shares'] % 100: errors.append('buy_lot_size')
                if any(t['signal_date'] >= t['entry_date'] or t['entry_date'] >= t['exit_date'] for t in rep.trades):
                    errors.append('timing_or_t1')
                if errors: raise ValueError(str(errors))
                results.append({'period':period,'case':name,**summary})
                if name in ('actual','slippage_x2','cash'):
                    dest=OUT/f'{period}_{name}'; dest.mkdir()
                    for key in ('fills','trades','actions','equity'):
                        pd.DataFrame(getattr(rep,key)).to_csv(dest/f'{key}.csv',index=False)
                    audits.append({'period':period,'case':name,'fills_checked':len(rep.fills), 'errors':errors})
            except ValueError as exc:
                failed.append({'period':period,'case':name,'error':str(exc)})
                if name in ('actual','slippage_x2','cash'): raise
            if num % 20 == 0: print(period, num, '/', len(choices), flush=True)
        actual = next(x['total_return'] for x in results if x['period']==period and x['case']=='actual')
        random = [x['total_return'] for x in results if x['period']==period and x['case'].startswith('random_')]
        print('PERIOD',period,'actual',actual,'random valid',len(random),'ge',sum(x>=actual for x in random),flush=True)
    random_summary=[]
    for period in ('2022-2025','2026'):
        actual=next(x['total_return'] for x in results if x['period']==period and x['case']=='actual')
        vals=[x['total_return'] for x in results if x['period']==period and x['case'].startswith('random_')]
        random_summary.append({'period':period,'actual':actual,'valid_draws':len(vals),
            'failed_draws':sum(x['period']==period for x in failed), 'random_mean':float(np.mean(vals)) if vals else None,
            'exceedances':sum(x>=actual for x in vals),
            'empirical_p_add_one':(1+sum(x>=actual for x in vals))/101 if len(vals)==100 else None,
            'selection_adjusted':False})
    write(OUT/'certification.json', {'results':results,'failed':failed,'fill_checks':audits,
        'random_summary':random_summary,'event_audit':event_audit,'not_future_profit_proof':True})
    print('COMPLETE', OUT, flush=True)


if __name__ == '__main__': main()
