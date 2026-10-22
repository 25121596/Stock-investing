"""Independent post-export check of all V10 cash, shares, fees and daily NAV."""
import argparse,json,hashlib
from collections import defaultdict
from pathlib import Path
import pandas as pd
from run import load
from corporate_events import canonicalize_events
from engine import Config

ROOT=Path(__file__).resolve().parent
def rows(p):
    try:return pd.read_csv(p,float_precision='round_trip').to_dict('records')
    except pd.errors.EmptyDataError:return []
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();out=args.run.resolve()
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'));started=json.loads((out/'started.json').read_text(encoding='utf-8'))
    for name,h in summary['source_hashes'].items():assert sha(out/'sources'/name)==h,('source_snapshot',name)
    for name,h in started['input_hashes'].items():assert sha(ROOT/name)==h,('input',name)
    for name,h in started['protected_hashes'].items():assert sha(Path(name))==h,('prospective_record',name)
    _,frames,_,raw=load(ROOT/'data/forward-2026-10-13');events,_=canonicalize_events(raw)
    bars={c:f.set_index('date').to_dict('index') for c,f in frames.items()}
    cases=[(r,out/f'{r["period"]}_{r["case"]}') for r in summary['results']]
    cases.extend((r,out/f'{r["period"]}_random_{r["seed"]}') for r in summary['random_controls'])
    audit=[]
    for r,folder in cases:
        cfg=Config(**r['config']);cash=cfg.capital;shares=defaultdict(int);waiting=[];entry={}
        fills,actions=defaultdict(list),defaultdict(list)
        for f in rows(folder/'fills.csv'):fills[f['date']].append(f)
        for a in rows(folder/'actions.csv'):actions[a['date']].append(a)
        for e in rows(folder/'equity.csv'):
            day=e['date'];cash+=sum(v for pay,v in waiting if pay<=day);waiting=[(pay,v) for pay,v in waiting if pay>day]
            for a in actions[day]:
                shares[a['code']]+=int(a['bonus_shares'])
                if a['net_cash']:
                    matches=[x for x in events if x['code']==a['code'] and x.get('dividOperateDate')==day]
                    assert len(matches)==1,('ambiguous_payment',a)
                    pay=matches[0]['dividPayDate']
                    if pay<=day:cash+=a['net_cash']
                    else:waiting.append((pay,a['net_cash']))
            for f in fills[day]:
                assert abs(f['price']-float(bars[f['code']][day]['open']))<1e-9
                assert abs(f['notional']-f['shares']*f['price'])<1e-6
                fees=cfg.fees(f['notional'],sell=f['side']=='sell')
                assert all(abs(f[k]-v)<1e-9 for k,v in fees.items()),(folder,f,fees)
                if f['side']=='buy':
                    assert f['shares']%100==0
                    entry[f['trade_id']]=day
                else:assert entry[f['trade_id']]<day
                sign=1 if f['side']=='buy' else -1
                shares[f['code']]+=sign*int(f['shares']);cash-=sign*f['notional']+sum(fees.values())
                assert shares[f['code']]>=0 and cash>=-1e-7
            receivable=sum(v for _,v in waiting)
            nav=cash+receivable+sum(q*float(bars[c][day]['close']) for c,q in shares.items() if q)
            assert max(abs(cash-e['cash']),abs(receivable-e['receivables']),abs(nav-e['equity']))<1e-6,(folder,day)
        assert abs(nav-r['final_equity'])<1e-6
        audit.append({'folder':folder.name,'fills':sum(len(v) for v in fills.values()),'passed':True})
    for fit in summary['training']:
        path=out/'models'/fit['policy']/f'{fit["year"]}.txt'
        assert sha(path)==fit['model_sha256']
        assert fit['last_label_end']<f'{fit["year"]}-01-01'
        baseline=json.loads((ROOT/f'results/v08-codex-rebuild-20261020-v1/v08/{fit["year"]}-training.json').read_text(encoding='utf-8'))
        assert fit['training_x_sha256']==baseline['training_x_sha256'] and fit['training_rows']==baseline['training_rows']
    result={'all_valid_exported_accounts_passed':True,'accounts':audit,'total_fills':sum(x['fills'] for x in audit),
        'new_model_weights_checked':len(summary['training']),'failed_cases_preserved':summary['failures'],
        'protected_prospective_evidence_unchanged':True}
    path=out/'independent-export-audit.json'
    with path.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print('PASSED',len(audit),'exported accounts;',result['total_fills'],'fills;',len(summary['training']),'annual model files; prospective hashes unchanged')

if __name__=='__main__':main()
