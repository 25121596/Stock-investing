"""Auditable historical arena; no training, broker orders or forward-journal writes."""
from __future__ import annotations
import argparse
import ast
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from engine import Config
from run import load
from corporate_events import canonicalize_events
from exploration_features import breadth_days
from run_exploration import stable_random, matched_candidates
from v08_engine_verified import VerifiedV08Replay
from validate_expanded import prepare_bars

ROOT = Path(__file__).resolve().parent
ARENA_DIR = ROOT / 'results/arena'
DATASET = ROOT / 'data/forward-2026-10-13'
SEEDS = tuple(range(2026102001, 2026102021))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_dependencies():
    """Snapshot transitive local imports, without unrelated concurrent projects."""
    pending=[ROOT/'arena.py'];seen=set()
    while pending:
        path=pending.pop()
        if path in seen:continue
        seen.add(path)
        tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else ([node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for name in names:
                local=ROOT/(name.split('.')[0]+'.py')
                if local.is_file() and local not in seen:pending.append(local)
    return sorted(seen)

def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def validate_rows(frame,columns,name):
    if frame.empty or frame.duplicated(['date','code']).any():
        raise ValueError(f'{name}: empty or duplicate keys')
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError(f'{name}: nonfinite values')

def build_signals(v2,v3,v08,limits):
    for frame,col,name in [(v2,'pred_net_return_v2','v2'),(v3,'pred_net_return_v3','v3'),(v08,'pred_score_v08','v08')]:
        validate_rows(frame,[col],name)
    common=v2[['date','code']].merge(v3[['date','code']],on=['date','code'],validate='one_to_one')
    common=common.merge(v08[['date','code']],on=['date','code'],validate='one_to_one')
    if not len(common)==len(v2)==len(v3)==len(v08):
        raise ValueError('Prediction universes differ; refuse selective comparison')
    joined=v2.merge(v3,on=['date','code'],validate='one_to_one').merge(v08,on=['date','code'],validate='one_to_one')
    allowed=joined.date.map(limits).fillna(0).gt(0)
    selections={}
    for name,gate,rank in [('v2','pred_net_return_v2','pred_net_return_v2'),('v3','pred_net_return_v3','pred_net_return_v3'),('v08','pred_net_return_v2','pred_score_v08')]:
        selections[name]=joined.loc[allowed & joined[gate].ge(.005),['date','code',rank]].rename(columns={rank:'score'})
    return selections,common

def audit_ledger(rep,bars,events):
    """Independent reconstruction of all daily cash, share balances and NAV."""
    cfg=rep.config;cash=cfg.capital;shares=defaultdict(int);waiting=[]
    fills,actions=defaultdict(list),defaultdict(list);entry={}
    for f in rep.fills:fills[f['date']].append(f)
    for a in rep.actions:actions[a['date']].append(a)
    for row in rep.equity:
        day=row['date'];cash+=sum(v for pay,v in waiting if pay<=day)
        waiting=[(pay,v) for pay,v in waiting if pay>day]
        for a in actions[day]:
            shares[a['code']]+=a['bonus_shares']
            if a['net_cash']:
                matched=[e for e in events if e['code']==a['code'] and e.get('dividOperateDate')==day]
                if len(matched)!=1:raise ValueError('Ambiguous dividend payment')
                pay=matched[0]['dividPayDate']
                if pay<=day:cash+=a['net_cash']
                else:waiting.append((pay,a['net_cash']))
        for f in fills[day]:
            fees=cfg.fees(f['notional'],sell=f['side']=='sell')
            if abs(f['price']-float(bars[f['code']][day]['open']))>1e-9 or abs(f['notional']-f['shares']*f['price'])>1e-6 or any(abs(f[k]-v)>1e-9 for k,v in fees.items()):
                raise ValueError('Fill price/notional/fees mismatch')
            if f['side']=='buy':
                if f['shares']%100:raise ValueError('Non-lot buy')
                entry[f['trade_id']]=day
            elif day<=entry[f['trade_id']]:raise ValueError('T+1 violation')
            sign=1 if f['side']=='buy' else -1
            shares[f['code']]+=sign*f['shares'];cash-=sign*f['notional']+sum(fees.values())
            if shares[f['code']]<0 or cash< -1e-7:raise ValueError('Negative cash/shares')
        cash*=1+rep.daily_cash_yield
        receivable=sum(v for _,v in waiting)
        nav=cash+receivable+sum(q*float(bars[c][day]['close']) for c,q in shares.items() if q)
        if max(abs(cash-row['cash']),abs(receivable-row['receivables']),abs(nav-row['equity']))>1e-6:
            raise ValueError(f'Independent ledger mismatch: {day}')
    return {'passed':True,'fills_checked':len(rep.fills),'daily_nav_checked':len(rep.equity)}

def simulate(frames,bars,days,signals,events,cfg,vol,breadth,limits,cash_rate=0.,fixed=False):
    kwargs=dict(high_budget=5000.,neutral_budget=5000.,high_max_pos=4,neutral_max_pos=4) if fixed else {}
    rep=VerifiedV08Replay({},days,signals,events,cfg,vol_map=vol,daily_limits=limits,breadth_lookup=breadth,enable_cash_overlay=bool(cash_rate),**kwargs)
    rep.daily_cash_yield=(1+cash_rate)**(1/242)-1
    rep.frames,rep.bars=frames,bars
    try:
        return rep.run()
    except (ValueError,AssertionError) as exc:
        exc.replay=rep
        raise

def save_case(folder,rep,summary,signals):
    folder.mkdir()
    for name in ['fills','trades','equity','actions','warnings','review_warnings']:
        pd.DataFrame(getattr(rep,name)).to_csv(folder/f'{name}.csv',index=False)
    pd.DataFrame(rep.holdings()).to_csv(folder/'holdings.csv',index=False)
    signals.to_csv(folder/'signals.csv',index=False)
    write_json(folder/'summary.json',summary)

def generate_dashboard(output):
    record=json.loads((output/'summary.json').read_text(encoding='utf-8'))
    body=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>历史策略竞技场</title>',
        '<style>body{font:15px system-ui;margin:32px;background:#101827;color:#e5e7eb}table{border-collapse:collapse;margin:20px 0}td,th{padding:10px;border:1px solid #475569}a{color:#7dd3fc}</style>',
        '<h1>历史策略竞技场</h1><p>已查看历史重放，不是盲测或实时交易。每账户初始3万元，无训练、无调参。</p>',
        '<p>统一风险轨道比较选股；原配置轨道比较策略产品，分开排名。随机账户匹配每日候选数与执行规则，不匹配最终成交或实际暴露。费用、T+1与质量拦截统一。</p>',
        f'<p>固定现金年利率假设：{record["cash_rate"]:.2%}，并非GC001实际成交。胜率仅指盈利平仓比例；无交易或不可定义的指标显示空值。</p>',
        '<p>随机对照仅作描述，无法消除历史已查看、多次研究及固定300股样本的选择偏差。缺少独立训练血缘认证的模型不能据此获得无泄漏保证。</p>',
        '<p>本次状态：'+('全部场次完成。' if record['complete'] else '存在被拦截场次；随机分布是有效账户的条件分布，不宣布统计优胜者。')+'</p>']
    for period in record['periods']:
        for track in ['common','native']:
            rows=sorted([r for r in record['results'] if r['period']==period and r['track']==track and r['kind'] in ['model','cash']],key=lambda r:r['total_return'],reverse=True)
            table=pd.DataFrame([{'策略':r['case'],'累计收益':f'{r["total_return"]:.2%}',
                '年化':None if r['cagr'] is None else f'{r["cagr"]:.2%}','最大回撤':f'{r["max_drawdown"]:.2%}',
                '平仓数':r['closed_trades'],'胜率':None if r['win_rate'] is None else f'{r["win_rate"]:.1%}',
                '平均暴露':f'{r["average_exposure"]:.1%}'} for r in rows])
            body.extend([f'<h2>{period} · {"统一风险" if track=="common" else "原配置"}</h2>',table.to_html(index=False,na_rep='—',escape=True)])
        body.append('<h3>统一风险净值（初始本金=100）</h3>')
        colors=['#38bdf8','#fb923c','#a78bfa','#94a3b8']
        curves=[]
        for model in ['v2','v3','v08','cash']:
            path=output/period/('common_'+model)/'equity.csv'
            if path.exists():curves.append((model,np.r_[100.,pd.read_csv(path).equity.to_numpy()/300.]))
        if curves:
            low=min(float(v.min()) for _,v in curves)-1;high=max(float(v.max()) for _,v in curves)+1
            body.append('<svg viewBox="0 0 1000 300" role="img" aria-label="统一风险累计净值曲线">')
            for (name,values),color in zip(curves,colors):
                points=' '.join(f'{40+i*930/max(1,len(values)-1):.1f},{260-(v-low)*230/(high-low):.1f}' for i,v in enumerate(values))
                body.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
            body.append('</svg><p>'+ ' / '.join(f'<span style="color:{c}">{n}</span>' for (n,_),c in zip(curves,colors))+'</p>')
    body.extend(['<h2>随机对照与双倍滑点</h2>',pd.DataFrame(record['comparisons']).to_html(index=False,na_rep='—',escape=True),
        '<p>随机比较：有效数/请求数、净收益分布、超过模型的比例（不是显著性p值）。失败场次不得上榜，须连同失败列表解释结果。</p>',
        pd.DataFrame(record['failures']).to_html(index=False,escape=True),'<a href="summary.json">参数、质量检查及独立账本审计</a></html>'])
    (output/'dashboard.html').write_text('\n'.join(body),encoding='utf-8')

def run_tournament(periods,output=None,random_count=20,cash_rate=0.):
    if not 1<=random_count<=len(SEEDS) or not 0<=cash_rate<=.2:raise ValueError('Require 1..20 fixed seeds, cash rate 0..0.2')
    output=Path(output).resolve() if output else ARENA_DIR/datetime.now(timezone.utc).strftime('run-%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True,exist_ok=False)
    sources=source_dependencies();hashes={p.name:digest(p) for p in sources}
    (output/'sources').mkdir()
    for p in sources:shutil.copyfile(p,output/'sources'/p.name)
    inputs=[DATASET/'manifest.json',DATASET/'calendar.csv',ROOT/'results/midterm-access-v3/features.csv']
    predpaths=[ROOT/f'results/model-{v}/predictions-h10-{v}.csv' for v in ['v2','v3','v08']];inputs+=predpaths
    input_hashes={str(p.relative_to(ROOT)):digest(p) for p in inputs}
    write_json(output/'started.json',{'periods':periods,'random_seeds':list(SEEDS[:random_count]),'cash_rate':cash_rate,
        'source_hashes':hashes,'input_hashes':input_hashes,'all_history_previously_viewed':True,
        'fixed_rules':'No fitting/threshold search. Common risk and native configurations ranked separately. Costs included.'})
    _,frames,calendar,raw=load(DATASET);events,event_audit=canonicalize_events(raw)
    if event_audit['unresolved_incomplete_groups']:raise ValueError('Unresolved corporate action duplicates')
    bars=prepare_bars(frames)
    panel=pd.read_csv(inputs[2],dtype={'date':str,'code':str});validate_rows(panel,['ret_vol20','adj_ret5'],'features')
    predictions=[pd.read_csv(p,dtype={'date':str,'code':str}) for p in predpaths]
    breadth=breadth_days(panel);bm=breadth.set_index('date').breadth5.to_dict();pm=breadth.set_index('date').pool_size.to_dict()
    limits={d:0 if pm[d]<100 or b<.45 else (1 if b<.55 else 2) for d,b in bm.items()}
    vol=panel.set_index(['date','code']).ret_vol20.to_dict()
    signals,pool=build_signals(*predictions,limits)
    coverage=pool.merge(panel[['date','code']],on=['date','code'],how='left',indicator=True,validate='one_to_one')
    if not coverage._merge.eq('both').all() or not set(pool.code).issubset(frames):raise ValueError('Predictions outside feature/data universe')
    cfg=Config(capital=30000,order_budget=7500,max_positions=5,max_exposure=.9,hold_days=10)
    fixed_cfg=replace(cfg,order_budget=5000,max_positions=4,max_exposure=.65)
    results=[];failures=[]
    for period,start,end in periods:
        days=[d for d in calendar if start<=d<=end]
        if len(days)<2:raise ValueError('Period requires at least two sessions')
        pdir=output/period;pdir.mkdir()
        chosen={k:v[v.date.between(start,end)].copy() for k,v in signals.items()};random_pool=pool[pool.date.between(start,end)]
        cases=[('common','cash','cash',chosen['v2'].iloc[:0],cfg,False,None)]
        for model in ['v2','v3','v08']:
            cases.extend([('common',model,'model',chosen[model],cfg,False,None),('common',model+'_slip2','stress',chosen[model],replace(cfg,slippage=.001),False,None)])
            for seed in SEEDS[:random_count]:
                cases.append(('common',f'{model}_random_{seed}','random',matched_candidates(chosen[model],stable_random(random_pool,seed)),cfg,False,seed))
        for name,model,fixed in [('v5','v2',True),('v6','v3',True),('v7','v2',False),('v08','v08',False)]:
            config=fixed_cfg if fixed else cfg
            cases.extend([('native',name,'model',chosen[model],config,fixed,None),('native',name+'_slip2','stress',chosen[model],replace(config,slippage=.001),fixed,None)])
        for track,name,kind,selected,config,fixed,seed in cases:
            case_id=f'{track}_{name}'
            try:
                rep=simulate(frames,bars,days,selected,events,replace(config),vol,bm,limits,cash_rate,fixed)
                summary=rep.summary();audit=audit_ledger(rep,bars,events);eq=pd.DataFrame(rep.equity)
                summary.update(period=period,track=track,case=name,kind=kind,seed=seed,audit=audit,
                    average_exposure=float(((eq.equity-eq.cash-eq.receivables)/eq.equity).mean()),signal_candidates=len(selected),signal_dates=int(selected.date.nunique()))
                save_case(pdir/case_id,rep,summary,selected);results.append(summary)
                print(period,case_id,f'{summary["total_return"]:.3%}',flush=True)
            except (ValueError,AssertionError) as exc:
                failure={'period':period,'case':case_id,'kind':kind,'error':str(exc)};failures.append(failure)
                write_json(pdir/(case_id+'-FAILED.json'),failure)
                if hasattr(exc,'replay'):save_case(pdir/(case_id+'-FAILED'),exc.replay,failure,selected)
                print('FAILED',failure,flush=True)
    comparisons=[]
    for period,_,_ in periods:
        for model in ['v2','v3','v08']:
            group=[r for r in results if r['period']==period and r['track']=='common']
            own=next((r for r in group if r['case']==model),None);stress=next((r for r in group if r['case']==model+'_slip2'),None)
            controls=[r['total_return'] for r in group if r['kind']=='random' and r['case'].startswith(model+'_')]
            comparisons.append({'period':period,'model':model,'valid_random':len(controls),'requested_random':random_count,
                'model_return':None if own is None else own['total_return'],'random_median':float(np.median(controls)) if controls else None,
                'random_p10':float(np.quantile(controls,.1)) if controls else None,'random_p90':float(np.quantile(controls,.9)) if controls else None,
                'fraction_random_ge_model':None if not controls or own is None else sum(x>=own['total_return'] for x in controls)/len(controls),
                'double_slippage_return':None if stress is None else stress['total_return']})
    if hashes!={p.name:digest(p) for p in sources} or input_hashes!={str(p.relative_to(ROOT)):digest(p) for p in inputs}:raise ValueError('Inputs/sources changed during run')
    summary={'periods':[p[0] for p in periods],'results':results,'failures':failures,'comparisons':comparisons,'cash_rate':cash_rate,
        'event_audit':event_audit,'all_history_previously_viewed':True,'complete':not failures,'source_hashes':hashes,'input_hashes':input_hashes,
        'scope':'Fixed 300-stock historical universe; no point-in-time market-wide representativeness or future profit guarantee.'}
    write_json(output/'summary.json',summary);pd.DataFrame(results).to_csv(output/'leaderboard.csv',index=False)
    generate_dashboard(output);print('SAVED',output,'failures',len(failures),flush=True)
    return output

def resolve_run(path=None):
    if path:return Path(path).resolve()
    runs=sorted(ARENA_DIR.glob('run-*/summary.json'))
    if not runs:raise ValueError('No new-format runs; specify --output RUN_DIRECTORY')
    return runs[-1].parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',nargs='?',choices=['run','leaderboard','dashboard'],default='run')
    parser.add_argument('--mode',choices=['all','backtest','2026'])
    parser.add_argument('--start');parser.add_argument('--end');parser.add_argument('--output',type=Path)
    parser.add_argument('--random-count',type=int,default=20);parser.add_argument('--cash-rate',type=float,default=0.)
    parser.add_argument('--no-cash',action='store_true',help='Compatibility option: zero cash yield is already default')
    args=parser.parse_args()
    if args.command!='run':
        output=resolve_run(args.output)
        if args.command=='dashboard':generate_dashboard(output)
        else:print(pd.read_csv(output/'leaderboard.csv').to_string(index=False,columns=['period','track','case','kind','total_return','max_drawdown','closed_trades']))
        return
    if bool(args.start)!=bool(args.end):parser.error('Provide both --start and --end')
    if args.start and args.mode not in [None,'backtest']:parser.error('Custom dates conflict with mode')
    if args.no_cash and args.cash_rate:parser.error('--no-cash conflicts with --cash-rate')
    if args.start:
        start=pd.Timestamp(args.start).strftime('%Y-%m-%d');end=pd.Timestamp(args.end).strftime('%Y-%m-%d')
        if start>=end:parser.error('Start must precede end')
        periods=[('custom',start,end)]
    else:
        periods=[('2022-2025','2022-01-01','2025-12-31'),('2026','2026-01-01','2026-10-13')]
        if args.mode=='backtest':periods=periods[:1]
        if args.mode=='2026':periods=periods[1:]
    run_tournament(periods,args.output,args.random_count,args.cash_rate)

if __name__=='__main__':main()
