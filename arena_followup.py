"""Diagnose blocked histories and quantify frozen arena fragility; never tune."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import socket
import urllib.request
import numpy as np
import pandas as pd
import arena
from download import read_result, FIELDS

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'results/arena/run-20261020T145952030119Z'

def read_csv(path):
    try:return pd.read_csv(path)
    except pd.errors.EmptyDataError:return pd.DataFrame()

def metrics(summary,trades):
    pnl=trades.net_pnl if len(trades) else pd.Series(dtype=float)
    winners=pnl[pnl>0].sort_values(ascending=False)
    realised=float(pnl.sum());account=summary['final_equity']-summary['initial_capital']
    result={'closed_pnl':realised,'account_pnl':account,'open_and_other_pnl':account-realised,
        'closed_trades':len(pnl),'win_rate':float(pnl.gt(0).mean()) if len(pnl) else None,
        'mean_win':float(winners.mean()) if len(winners) else None,
        'mean_loss':float(pnl[pnl<0].mean()) if pnl.lt(0).any() else None,
        'median_trade':float(pnl.median()) if len(pnl) else None,
        'execution_costs':summary['execution_costs'],'average_exposure':summary['average_exposure']}
    for k in [1,3,5]:
        contribution=float(winners.head(k).sum())
        result[f'top{k}_winner_pnl']=contribution
        result[f'top{k}_share_account_profit']=contribution/account if account>0 else None
        result[f'excluding_top{k}_contribution_return']=(account-contribution)/summary['initial_capital']
    result['interpretation']='Algebraic closed-trade contribution subtraction; NOT a rerun, NAV path, drawdown or executable counterfactual.'
    return result

def periods(eq,trades,capital,frequency='Y'):
    frame=eq.copy();frame['bucket']=frame.date.str[:4 if frequency=='Y' else 7]
    previous=capital;rows=[]
    for bucket,group in frame.groupby('bucket',sort=True):
        final=float(group.equity.iloc[-1]);selected=trades[trades.exit_date.str.startswith(bucket)] if len(trades) else trades
        closed=float(selected.net_pnl.sum()) if len(selected) else 0.
        delta=final-previous
        rows.append({'bucket':bucket,'start_equity':previous,'end_equity':final,'nav_change':delta,
            'return':final/previous-1,'closed_pnl_by_exit':closed,'open_and_other_pnl_change':delta-closed,
            'closed_trades':len(selected),'win_rate':float(selected.net_pnl.gt(0).mean()) if len(selected) else None})
        previous=final
    if abs(sum(r['nav_change'] for r in rows)-(float(eq.equity.iloc[-1])-capital))>1e-6:
        raise ValueError('Annual attribution does not reconcile')
    return rows

def market_evidence(out):
    out.mkdir()
    urls={'halt':'https://static.cninfo.com.cn/finalpage/2023-02-08/1215789769.PDF',
          'delisting':'https://static.cninfo.com.cn/finalpage/2023-03-17/1216135561.PDF'}
    downloaded=[]
    for name,url in urls.items():
        try:
            data=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=30).read()
            if not data.startswith(b'%PDF'):raise ValueError('Not PDF')
            path=out/(name+'.pdf');path.write_bytes(data)
            downloaded.append({'url':url,'file':path.name,'sha256':arena.digest(path)})
        except Exception as exc:downloaded.append({'url':url,'error':str(exc)})
    # Independently re-query the same vendor. Empty post-delisting data is evidence,
    # not authorization to synthesize prices or remove a historical constituent.
    import baostock as bs
    socket.setdefaulttimeout(20)
    queries=[]
    try:
        login=bs.login()
        if login.error_code!='0':raise ValueError(login.error_msg)
        from baostock.common import context
        if context.default_socket:context.default_socket.settimeout(20)
        for start,end in [('2023-01-30','2023-02-10'),('2023-03-16','2023-03-24')]:
            params=dict(code='sz.002013',fields=FIELDS,start_date=start,end_date=end,frequency='d',adjustflag='3')
            fresh=read_result(bs.query_history_k_data_plus(**params))
            filename=f'vendor-{start}.csv';fresh.to_csv(out/filename,index=False)
            local=pd.read_csv(arena.DATASET/'sz.002013.csv',dtype={'date':str,'code':str})
            local=local[local.date.between(start,end)]
            matches=set(fresh.date)==set(local.date)
            for row in fresh.to_dict('records'):
                prior=local[local.date.eq(row['date'])]
                if len(prior)!=1:matches=False;continue
                for col in ['open','close','preclose','tradestatus']:
                    if abs(float(row[col])-float(prior.iloc[0][col]))>1e-8:matches=False
            queries.append({'params':params,'rows':len(fresh),'dates':fresh.date.tolist(),
                'local_coverage_and_prices_match':matches,'file':filename,'sha256':arena.digest(out/filename)})
    except Exception as exc:queries.append({'error':str(exc)})
    finally:
        bs.logout()
    result={'fetched_utc':datetime.now(timezone.utc).isoformat(),'official_notices':downloaded,'vendor_queries':queries,
        'finding':'Trading halted from 2023-02-08; delisted 2023-03-17 for share-swap merger into sh.600372. Not an ordinary quote download gap.',
        'decision':'Keep original dataset and failed accounts. No synthetic fills, future-based deletion, replacement seeds or invented corporate action allocation.',
        'blocking_details':'Ratio 0.6647; 600 old shares imply 398.82 theoretical new shares. Actual fractional-share allocation, successor listing/delivery and account mapping not represented in daily-bar/dividend data. Cash election restricted to qualified dissenting shareholders cannot be assumed.'}
    arena.write_json(out/'evidence.json',result)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    summary=json.loads((BASE/'summary.json').read_text(encoding='utf-8'))
    protected=[BASE/'summary.json',arena.DATASET/'manifest.json',arena.DATASET/'sz.002013.csv',
        ROOT/'results/v08-paper-forward-v3/days/2026-10-20/record.json']
    before={str(p):arena.digest(p) for p in protected}
    arena.write_json(out/'started.json',{'base':str(BASE),'base_summary_sha256':arena.digest(BASE/'summary.json'),
        'source_sha256':arena.digest(Path(__file__)),'protected_hashes':before,
        'fixed_diagnostics':['all models, both tracks, all years','top 1/3/5 profitable trade contributions','monthly and code concentration','no fitting, threshold changes, policy selection or independent validation claim']})
    evidence=market_evidence(out/'evidence')
    # Reproduce every interrupted account from its exact saved signal set.
    _,frames,calendar,raw=arena.load(arena.DATASET);events,_=arena.canonicalize_events(raw);bars=arena.prepare_bars(frames)
    panel=pd.read_csv(ROOT/'results/midterm-access-v3/features.csv',dtype={'date':str,'code':str})
    breadth=arena.breadth_days(panel);bm=breadth.set_index('date').breadth5.to_dict();pm=breadth.set_index('date').pool_size.to_dict()
    limits={d:0 if pm[d]<100 or b<.45 else (1 if b<.55 else 2) for d,b in bm.items()}
    vol=panel.set_index(['date','code']).ret_vol20.to_dict();rechecks=[]
    cfg=arena.Config(capital=30000,order_budget=7500,max_positions=5,max_exposure=.9,hold_days=10)
    for failure in summary['failures']:
        path=BASE/failure['period']/(failure['case']+'-FAILED')/'signals.csv'
        signal=pd.read_csv(path,dtype={'date':str,'code':str});days=[d for d in calendar if '2022-01-01'<=d<='2025-12-31']
        try:
            arena.simulate(frames,bars,days,signal,events,cfg,vol,bm,limits)
            rechecks.append({'case':failure['case'],'unexpected_success':True})
        except ValueError as exc:
            held=exc.replay.holdings()
            rechecks.append({'case':failure['case'],'same_failure':str(exc)==failure['error'],
                'error':str(exc),'holdings':held,'rejected':dict(exc.replay.rejected),'signals_sha256':arena.digest(path)})
    arena.write_json(out/'failure-recheck.json',rechecks)
    if not all(x.get('same_failure') for x in rechecks):raise ValueError('Failure changed unexpectedly')
    overview=[];annual=[];monthly=[];top=[];bycode=[];all_trades=[]
    for s in summary['results']:
        if s['kind']!='model':continue
        identity={k:s[k] for k in ['period','track','case']};folder=BASE/s['period']/(s['track']+'_'+s['case'])
        trades=read_csv(folder/'trades.csv');eq=read_csv(folder/'equity.csv')
        m=metrics(s,trades);m.update(identity,total_return=s['total_return'],max_drawdown=s['max_drawdown'],open_positions=s['open_positions'])
        bymonth=trades.assign(entry_month=trades.entry_date.str[:7]).groupby('entry_month').net_pnl.sum() if len(trades) else pd.Series(dtype=float)
        bestmonth=float(bymonth.max()) if len(bymonth) else 0.
        m['best_entry_month_closed_pnl']=bestmonth
        m['best_entry_month_share_account_profit']=bestmonth/m['account_pnl'] if m['account_pnl']>0 else None
        overview.append(m)
        if len(trades):
            detailed=trades.copy()
            detailed['total_fees']=detailed.entry_fees+detailed.exit_fees
            detailed['calendar_holding_days']=(pd.to_datetime(detailed.exit_date)-pd.to_datetime(detailed.entry_date)).dt.days
            detailed['account_profit_share']=detailed.net_pnl/m['account_pnl'] if m['account_pnl']>0 else np.nan
            detailed['entry_month']=detailed.entry_date.str[:7]
            detailed['exit_month']=detailed.exit_date.str[:7]
            all_trades.extend([{**identity,**r} for r in detailed.to_dict('records')])
        annual.extend([{**identity,**r} for r in periods(eq,trades,s['initial_capital'])])
        monthly.extend([{**identity,**r} for r in periods(eq,trades,s['initial_capital'],'M')])
        top.extend([{**identity,**r} for r in trades.nlargest(min(10,len(trades)),'net_pnl').to_dict('records')])
        if len(trades):
            grouped=trades.groupby('code').agg(trades=('net_pnl','size'),net_pnl=('net_pnl','sum')).reset_index()
            bycode.extend([{**identity,**r} for r in grouped.to_dict('records')])
    for name,rows in [('overview',overview),('annual',annual),('monthly',monthly),('top-trades',top),('by-code',bycode),('all-trades',all_trades)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False,encoding='utf-8-sig')
    if before!={str(p):arena.digest(p) for p in protected}:raise ValueError('Protected historical or forward evidence changed')
    arena.write_json(out/'summary.json',{'evidence':evidence,'failure_rechecks':rechecks,'overview':overview,'annual':annual,
        'protected_hashes_unchanged':True,'interpretation':'Descriptive viewed-history diagnostics, not model selection or independent forward evidence.'})
    rows=pd.DataFrame(overview);common=rows[rows.track.eq('common')]
    cols=['period','case','total_return','closed_trades','account_pnl','top3_winner_pnl','top3_share_account_profit','excluding_top3_contribution_return','mean_win','mean_loss','execution_costs']
    html='<meta charset="utf-8"><title>竞技场脆弱性诊断</title><style>body{font:15px system-ui;margin:30px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px}</style><h1>竞技场脆弱性诊断</h1><p>最大赢家扣除仅为代数贡献分析，不是删除交易后重跑；不重新选择股票、不模拟替代持仓，不能据此计算新的回撤。</p>'
    html+='<h2>同风险模型比较（收益及比例以小数表示）</h2>'+common[cols].to_html(index=False,na_rep='—',escape=True)
    html+='<h2>逐年净值归因（连续账户，不按年重置）</h2>'+pd.DataFrame(annual).query('track == "common"').to_html(index=False,na_rep='—',escape=True)
    html+='<h2>三账户失败根因</h2><p>吸收合并导致旧代码结束报价，需要证券转换和登记结算，不能补行情或强制按最后价卖出。原失败和数据均保留，重新运行三账户均复现。</p>'
    html+='<p>完整CSV：overview、annual、monthly、top-trades、by-code；原始公告及供应商复查见evidence目录。</p>'
    (out/'report.html').write_text(html,encoding='utf-8')
    print(common[cols].to_string(index=False));print('SAVED',out,flush=True)

if __name__=='__main__':main()
