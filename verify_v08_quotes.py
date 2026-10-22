"""Re-fetch selected realized-profit-source bars from BaoStock, with evidence."""
import json
from datetime import datetime, timezone
from pathlib import Path
import socket
import baostock as bs
import pandas as pd
from download import read_result, FIELDS

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/v08-codex-quotes-20261020-v1'


def main():
    OUT.mkdir(exist_ok=False)
    samples=[]
    for period in ['2022-2025','2026']:
        trades=pd.read_csv(ROOT/f'results/codex-review-v08-20261020T064309Z/{period}_original_pure/trades.csv')
        chosen=pd.concat([trades.nlargest(2,'net_pnl'),trades.nsmallest(1,'net_pnl'),trades.sample(1,random_state=20261020)])
        samples.extend(chosen.drop_duplicates('trade_id').to_dict('records'))
    socket.setdefaulttimeout(25)
    login=bs.login()
    if login.error_code!='0': raise ValueError(login.error_msg)
    result=[]
    try:
        for index,t in enumerate(samples):
            params={'code':t['code'],'fields':FIELDS,'start_date':t['entry_date'],'end_date':t['exit_date'],'frequency':'d','adjustflag':'3'}
            started=datetime.now(timezone.utc).isoformat()
            fresh=read_result(bs.query_history_k_data_plus(**params))
            old=pd.read_csv(ROOT/f'data/forward-2026-10-13/{t["code"]}.csv',dtype={'date':str,'code':str})
            old=old[old.date.between(t['entry_date'],t['exit_date'])]
            errors=[]
            if fresh.empty or set(fresh.date)!=set(old.date): errors.append('coverage')
            for row in fresh.to_dict('records'):
                expected=old[old.date.eq(row['date'])]
                if len(expected)!=1: continue
                expected=expected.iloc[0]
                for key in ['open','high','low','close','preclose','volume','amount']:
                    a=pd.to_numeric(pd.Series([row[key]]),errors='coerce').iloc[0]
                    b=expected[key]
                    if pd.isna(a) and pd.isna(b): continue
                    if pd.isna(a) or pd.isna(b) or abs(a-b) > (.5 if key=='volume' else .011):
                        errors.append(f'{row["date"]}/{key}')
                if row['code']!=t['code'] or row['adjustflag']!='3': errors.append('identity_or_flag')
            item={'params':params,'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),
                'trade_net_profit':t['net_pnl'],'rows':len(fresh),'mismatches':errors,'raw_response':fresh.to_dict('records')}
            result.append(item)
            (OUT/f'{index:02d}.json').write_text(json.dumps(item,ensure_ascii=False,indent=2),encoding='utf-8')
            print(t['code'],t['entry_date'],t['exit_date'],'rows',len(fresh),'mismatches',len(errors),flush=True)
    finally:
        bs.logout()
    (OUT/'summary.json').write_text(json.dumps({'samples':len(result),'rows':sum(x['rows'] for x in result),
        'mismatches':sum(len(x['mismatches']) for x in result),
        'scope':'Independent re-fetch from same vendor; targeted sample, not exchange-certified exhaustive validation'},indent=2))


if __name__=='__main__':main()
