"""Independent review replay. Does not train, tune, or overwrite Gemini results."""
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from corporate_events import canonicalize_events
from engine import Config, number
from experiment_super_v7_regime import SuperV7RegimeReplay
from exploration_features import breadth_days
from run import load
from validate_expanded import prepare_bars

ROOT = Path(__file__).resolve().parent


class ObservedReplay(SuperV7RegimeReplay):
    def corporate_actions(self, day):
        # The new run() bypasses Replay.run's price/reference diagnostics.
        for code in self.positions:
            bar = self.bars[code].get(day)
            if bar is None:
                self.review_warnings.append({'date': day, 'code': code, 'issue': 'missing_held_stock_quote'})
                continue
            reference = number(bar.get('preclose'))
            mark = self.marks.get(code, 0)
            if reference > 0 and mark > 0 and abs(reference - mark) > .025:
                known = [e for e in self.actions_by_ex.get(day, []) if e.get('code') == code]
                cash = sum(number(e.get('dividCashPsBeforeTax')) for e in known)
                ratio = sum(number(e.get('dividStocksPs')) + number(e.get('dividReserveToStockPs')) for e in known)
                if not known or abs(reference - (mark - cash) / (1 + ratio)) > .025:
                    self.review_warnings.append({'date': day, 'code': code,
                        'issue': 'unexplained_adjustment' if not known else 'corporate_action_reference_mismatch'})
        return super().corporate_actions(day)


def main():
    output = ROOT / 'results' / ('codex-review-v08-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    output.mkdir()
    sources = ['experiment_super_v08.py', 'experiment_super_v7_regime.py', 'experiment_super_v5_maxprofit.py',
               'train_model_v08.py', 'train_model_v2.py', 'train_model_v3.py', 'engine.py', 'engine_2026.py',
               'accessibility.py', 'corporate_events.py', 'verify_v08_adversarial.py']
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    (output / 'started.json').write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'hashes': hashes}, indent=2))
    manifest, frames, calendar, events = load(ROOT / 'data/forward-2026-10-13')
    cleaned, event_audit = canonicalize_events(events)
    bars = prepare_bars(frames)
    panel = pd.read_csv(ROOT / 'results/midterm-access-v3/features.csv', dtype={'date': str, 'code': str})
    a = pd.read_csv(ROOT / 'results/model-v08/predictions-h10-v08.csv', dtype={'date': str, 'code': str})
    b = pd.read_csv(ROOT / 'results/model-v2/predictions-h10-v2.csv', dtype={'date': str, 'code': str})
    merged = a.merge(b, on=['date', 'code'], validate='one_to_one')
    breadth = breadth_days(panel)
    bm = breadth.set_index('date').breadth5.to_dict()
    pm = breadth.set_index('date').pool_size.to_dict()
    limits = {d: 0 if pm[d] < 100 or v < .45 else (1 if v < .55 else 2) for d, v in bm.items()}
    signals = merged[merged.pred_net_return_v2.ge(.005) & merged.date.isin({d for d, n in limits.items() if n})]
    signals = signals[['date', 'code', 'pred_score_v08']].rename(columns={'pred_score_v08': 'score'})
    vol = panel.set_index(['date', 'code']).ret_vol20.to_dict()
    base = Config(capital=30000, order_budget=7500, max_positions=5, max_exposure=.90, hold_days=10)
    results = []
    for period, start, end in [('2022-2025', '2022-01-01', '2025-12-31'), ('2026', '2026-01-01', '2026-10-13')]:
        days = [d for d in calendar if start <= d <= end]
        for mode, selected_events, cash, slip in [('original_cash', events, True, 1), ('original_pure', events, False, 1),
                ('dedup_pure', cleaned, False, 1), ('dedup_pure_slip2', cleaned, False, 2)]:
            cfg = replace(base, slippage=base.slippage * slip)
            rep = ObservedReplay({}, days, signals[signals.date.between(start, end)], selected_events, cfg,
                vol_map=vol, daily_limits=limits, breadth_lookup=bm, enable_cash_overlay=cash)
            rep.frames, rep.bars, rep.review_warnings = frames, bars, []
            rep.run()
            summary = rep.summary()
            trades = pd.DataFrame(rep.trades)
            top = trades.nlargest(min(3, len(trades)), 'net_pnl') if len(trades) else trades
            summary.update(period=period, mode=mode, warnings=rep.review_warnings + rep.warnings,
                top3_profit=float(top.net_pnl.sum()) if len(top) else 0,
                winning_trades=sum(t['net_pnl'] > 0 for t in rep.trades),
                holdings=rep.holdings(), quality_warning_count=len(rep.review_warnings + rep.warnings))
            folder = output / f'{period}_{mode}'
            folder.mkdir()
            for name in ['equity', 'fills', 'trades', 'actions']:
                pd.DataFrame(getattr(rep, name)).to_csv(folder / f'{name}.csv', index=False)
            results.append(summary)
            print(period, mode, 'return', summary['total_return'], 'DD', summary['max_drawdown'],
                'wins/trades', summary['winning_trades'], summary['closed_trades'],
                'warnings', summary['quality_warning_count'], flush=True)
    if hashes != {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}:
        raise RuntimeError('Source changed during review')
    (output / 'review.json').write_text(json.dumps({'results': results, 'event_audit': event_audit,
        'source_hashes': hashes, 'no_retraining': True, 'all_periods_previously_viewed': True}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('OUTPUT', output, flush=True)


if __name__ == '__main__':
    main()
