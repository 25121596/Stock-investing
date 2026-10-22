"""Rebuild cash, share quantities and daily NAV independently from exported fills."""
import json,hashlib
from collections import defaultdict
from pathlib import Path
import pandas as pd
from run import load
from corporate_events import canonicalize_events
from engine import Config

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/v09-alignment-20261020-v2'


def read(path):
    try:return pd.read_csv(path).to_dict('records')
    except pd.errors.EmptyDataError:return []


def main():
    result=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    assert len(result['results'])==32 and not result['failures']
    for name,sha in result['source_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    _,frames,_,raw=load(ROOT/'data/forward-2026-10-13');events,_=canonicalize_events(raw)
    bars={c:f.set_index('date').to_dict('index') for c,f in frames.items()}
    records=[]
    for summary in result['results']:
        folder=OUT/(summary['period']+'_'+summary['case'])
        fills=read(folder/'fills.csv');actions=read(folder/'actions.csv');equity=read(folder/'equity.csv')
        cfg=Config(**summary['config']);cash=cfg.capital;shares=defaultdict(int);waiting=[]
        by_fill=defaultdict(list);by_action=defaultdict(list)
        for row in fills:by_fill[row['date']].append(row)
        for row in actions:by_action[row['date']].append(row)
        for row in equity:
            day=row['date'];cash+=sum(amount for pay,amount in waiting if pay<=day)
            waiting=[(pay,amount) for pay,amount in waiting if pay>day]
            for a in by_action[day]:
                shares[a['code']]+=int(a['bonus_shares'])
                if a['net_cash']:
                    event=[e for e in events if e['code']==a['code'] and e.get('dividOperateDate')==day]
                    assert len(event)==1,('ambiguous_payment',a)
                    pay=event[0]['dividPayDate'];amount=a['net_cash']
                    if pay<=day:cash+=amount
                    else:waiting.append((pay,amount))
            for f in by_fill[day]:
                assert abs(f['price']-float(bars[f['code']][day]['open']))<1e-9
                assert abs(f['notional']-f['shares']*f['price'])<1e-6
                fees=cfg.fees(f['notional'],sell=f['side']=='sell')
                assert all(abs(f[k]-v)<1e-9 for k,v in fees.items())
                sign=1 if f['side']=='buy' else -1
                shares[f['code']]+=sign*int(f['shares'])
                cash-=sign*f['notional']+sum(fees.values())
                assert shares[f['code']]>=0
            marked=sum(q*float(bars[c][day]['close']) for c,q in shares.items() if q)
            receivable=sum(amount for _,amount in waiting)
            assert abs(cash-row['cash'])<1e-6,(folder,day,'cash')
            assert abs(receivable-row['receivables'])<1e-6,(folder,day,'receivables')
            assert abs(cash+marked+receivable-row['equity'])<1e-6,(folder,day,'equity')
        for trade in read(folder/'trades.csv'):
            assert trade['signal_date']<trade['entry_date']<trade['exit_date']
        records.append({'period':summary['period'],'case':summary['case'],'fills_checked':len(fills),
            'daily_cash_and_nav_checked':len(equity),'passed':True})
    a=read(OUT/'2022-2025_gate_only/fills.csv');b=read(OUT/'2022-2025_stress_gate_only/fills.csv')
    first=None
    for i,(x,y) in enumerate(zip(a,b)):
        if any(x[k]!=y[k] for k in ['date','code','side','shares','price']):
            first={'fill_index':i,'normal':x,'double_slippage':y};break
    audit={'accounts':records,'all_passed':True,'total_fills':sum(x['fills_checked'] for x in records),
        'first_stress_execution_path_divergence':first,
        'interpretation':'Fees change available cash and integer order sizing. Stress simulation is a different execution path.'}
    (OUT/'ledger-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASSED',len(records),'cash/share/NAV ledgers;',audit['total_fills'],'fills')


if __name__=='__main__':main()
