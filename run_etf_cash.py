"""Fixed monthly signals and real cash/units for two audited equity ETFs."""
from dataclasses import replace
import argparse
from pathlib import Path
import hashlib
import html
import json
import shutil

import pandas as pd

from engine import Config
from etf_replay import ETFReplay
from run import load
from run_exploration import matched_candidates, stable_random
from run_midterm import enrich
from validate_expanded import prepare_bars, save_ledger

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/etf-cash-two-v1'
OUT = ROOT / 'results/etf-cash-v1'
SEEDS = (2026101401, 2026101402, 2026101403)
SOURCES = ('run_etf_cash.py', 'etf_replay.py', 'etf-cash-plan-v1.md', 'etf-budget-fix.md', 'engine.py', 'corporate_events.py', 'prepare_etf_cash.py')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signal_frame(rows):
    return pd.DataFrame(rows, columns=['date', 'code', 'score'])


def simulation(frames, bars, days, signals, events, config):
    replay = ETFReplay({}, days, signals, events, config)
    replay.frames, replay.bars = frames, bars
    return replay.run()


def make_report(payload):
    columns = {'label': '政策', 'period': '时期', 'total_return': '累计净收益', 'max_drawdown': '最大回撤',
        'win_rate': '平仓净胜率', 'closed_trades': '平仓笔数', 'execution_costs': '费用(元)',
        'avg_exposure': '平均仓位', 'open_positions': '期末持仓', 'quality_warning_count': '警告'}
    table = pd.DataFrame(payload['results'])[list(columns)].copy()
    for col in ('total_return', 'max_drawdown', 'win_rate', 'avg_exposure'):
        table[col] = table[col].map(lambda v: '—' if pd.isna(v) else f'{v:.2%}')
    table.execution_costs = table.execution_costs.map(lambda v: f'{v:.2f}')
    table.columns = list(columns.values())
    links = ''.join(f'<li>{html.escape(r["case"])}: ' + ' · '.join(
        f'<a href="{r["case"]}/{file}.csv">{file}</a>' for file in ('equity', 'trades', 'fills', 'actions', 'holdings')) + '</li>' for r in payload['results'])
    content = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>两只宽基ETF现金账户探索</title><style>body{font-family:system-ui,"Microsoft YaHei";margin:20px;line-height:1.6;color:#213547;background:#f5f7fa}'
        '.wrap{overflow-x:auto;max-width:100%;background:white}table{border-collapse:collapse;white-space:nowrap;width:100%}td,th{padding:8px;border:1px solid #ddd;text-align:right}'
        'pre{white-space:pre-wrap;overflow-wrap:anywhere}.note{padding:14px;background:#fff4d7;border-left:4px solid #ad7800}@media(max-width:600px){body{margin:10px}}</style>'
        '<h1>两只宽基ETF现金账户探索</h1><p class="note">仅限510300、510050。原5只中的其余3只因复权/事件口径问题保留审计，不纳入收益。'
        '所有时期均为历史探索，少数存续产品不代表全ETF市场。收益不是年化，买入持有包含期末市值，不能当作全部已实现利润。</p>'
        '<p>本金3万元、单笔预算15000元、最多一只、计划仓位50%；月末收盘选、次开盘尝试买，20市场日到期后等待新月末信号，无盘中或3%止损。'
        '原始价格成交，分红按登记/除息/支付日入账；后复权仅构造过去指标。佣金最低5元、双边0.05%滑点，股票印花税/过户费为零，基金分红按零所得税情景。</p>'
        '<div class="wrap">' + table.to_html(index=False, escape=True) + '</div>'
        '<p>随机只匹配两基金政策的信号日与候选数，实际持仓/成交仍可能不同。现金不计利息；滑点压力是滑点加倍。</p>'
        '<h2>协议和来源</h2><pre>' + html.escape(json.dumps(payload['metadata'], ensure_ascii=False, indent=2)) + '</pre><h2>逐笔账本</h2><ul>' + links + '</ul></html>')
    path = OUT / 'etf-report.html'
    path.write_text(content, encoding='utf-8')
    return path


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    OUT = parser.parse_args().output.resolve()
    if OUT.exists():
        raise SystemExit('Preserve existing ETF result revision')
    manifest, frames, calendar, events = load(DATA)
    if set(frames) != {'sh.510300', 'sh.510050'} or manifest['failures']:
        raise ValueError('Unexpected source scope')
    OUT.mkdir()
    (OUT / 'sources').mkdir()
    hashes = {name: digest(ROOT / name) for name in SOURCES}
    for name in SOURCES:
        shutil.copyfile(ROOT / name, OUT / 'sources' / name)
    shutil.copyfile(ROOT / 'etf-cash-plan-v1.md', OUT / 'protocol.md')
    rows = []
    for code, frame in frames.items():
        f = frame.sort_values('date').copy()
        f['ret60'] = f.adj_close.pct_change(60, fill_method=None)
        f['ma120'] = f.adj_close.rolling(120, min_periods=120).mean()
        rows.append(f[['date', 'code', 'adj_close', 'ret60', 'ma120']])
    panel = pd.concat(rows, ignore_index=True)
    cal = pd.DataFrame({'date': calendar})
    cal['month'] = cal.date.str[:7]
    month_ends = cal.groupby('month').date.max()
    # The final loaded month is incomplete. Do not infer a month end from truncation.
    month_ends = set(month_ends[month_ends.index < calendar[-1][:7]])
    valid = panel[panel.date.isin(month_ends) & panel.ret60.gt(0) & panel.adj_close.gt(panel.ma120)]
    rotation = valid.sort_values(['date', 'ret60', 'code'], ascending=[True, False, True]).groupby('date').head(1)
    rotation = rotation[['date', 'code', 'ret60']].rename(columns={'ret60': 'score'})
    trend = valid[valid.code.eq('sh.510300')][['date', 'code', 'ret60']].rename(columns={'ret60': 'score'})
    base = Config(order_budget=15000, max_positions=1, max_exposure=.5, hold_days=20,
                  stop_loss=0, stamp_tax=0, transfer_fee=0, dividend_tax=0)
    bars = prepare_bars(frames)
    results = []
    for period, start, end in (('2022-2025', '2022-01-01', '2025-12-31'), ('2026', '2026-01-01', '2026-10-13')):
        days = [d for d in calendar if start <= d <= end]
        rt = rotation[rotation.date.between(start, end)]
        tr = trend[trend.date.between(start, end)]
        cases = [('rotation', '两ETF月末趋势与相对强弱', 'policy', rt, replace(base)),
                 ('trend300', '510300月末趋势择时', 'policy', tr, replace(base))]
        for code in sorted(frames):
            cases.append((f'hold_{code}', f'{code}半仓买入持有', 'buy_hold',
                signal_frame([[days[0], code, 1.]]), replace(base, hold_days=1000000)))
        available = panel[panel.date.between(start, end)]
        for seed in SEEDS:
            cases.append((f'random_{seed}', f'两ETF同机会日随机{seed}', 'matched_random',
                matched_candidates(rt, stable_random(available, seed)), replace(base)))
        for name, label, signals in (('rotation', '两ETF月末趋势·滑点加倍', rt), ('trend300', '510300趋势·滑点加倍', tr)):
            cases.append((f'stress_{name}', label, 'stress', signals, replace(base, slippage=.001)))
        cases.append(('cash', '不交易现金', 'cash', signal_frame([]), replace(base)))
        assert len(cases) == 10
        for name, label, group, signals, config in cases:
            case = f'{period}_{name}'
            replay = simulation(frames, bars, days, signals, events, config)
            result = enrich(replay, signals, config.hold_days, period, case, label, group, days)
            save_ledger(OUT / case, replay)
            pd.DataFrame(replay.order_plans).to_csv(OUT / case / 'orders.csv', index=False, encoding='utf-8-sig')
            if group == 'policy':
                result['yearly'] = []
                for year in range(int(start[:4]), int(end[:4]) + 1):
                    yd = [d for d in days if d.startswith(str(year))]
                    ys = signals[signals.date.str.startswith(str(year))]
                    fresh = simulation(frames, bars, yd, ys, events, replace(config))
                    result['yearly'].append({'year': year, **fresh.summary()})
                    save_ledger(OUT / 'yearly' / f'{year}_{name}', fresh)
            results.append(result)
            print(f'ETF {case}: net={result["total_return"]:.2%}, DD={result["max_drawdown"]:.2%}, '
                  f'trades={result["closed_trades"]}, warnings={result["quality_warning_count"]}', flush=True)
    if hashes != {name: digest(ROOT / name) for name in SOURCES}:
        raise AssertionError('Sources changed during study')
    payload = {'version': 'etf-cash-v2', 'hashes': hashes, 'results': results,
        'metadata': {'dataset': str(DATA), 'manifest_sha256': digest(DATA / 'manifest.json'),
            'source_scope': manifest['scope'], 'cases': 20, 'restricted_small_account_access': False,
            'cash_events_checked': manifest['audits'], 'cash_conversion': manifest['event_cash_conversions'],
            'all_periods_exploratory': True, 'coarse_order_price_cap_tick': .01, 'ETF_limit_price_tick': .001}}
    (OUT / 'summary.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    pd.DataFrame(results)[['case', 'label', 'group', 'period', 'total_return', 'max_drawdown', 'win_rate', 'closed_trades', 'execution_costs', 'avg_exposure', 'open_positions', 'quality_warning_count']].to_csv(OUT / 'comparison.csv', index=False, encoding='utf-8-sig')
    report = make_report(payload)
    print(f'REPORT {report}')


if __name__ == '__main__':
    main()
