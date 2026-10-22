"""Scalar policy-consistent labels, verified against the frozen V0.8 ledger."""
from collections import Counter
import math
import numpy as np
import pandas as pd
from engine import Config,tick
from accessibility import AccessibleReplay


def policy_labels(panel,frames,calendar,events,breadth,vol_lookup):
    config=Config(capital=30000,order_budget=7500,max_positions=5,max_exposure=.9,hold_days=10)
    limits={r.date:0 if r.pool_size<100 or r.breadth5<.45 else (1 if r.breadth5<.55 else 2) for r in breadth.itertuples()}
    bm=breadth.set_index('date').breadth5.to_dict()
    dates=list(calendar); indexes={d:i for i,d in enumerate(dates)}; n=len(dates)
    checker=AccessibleReplay({},dates,pd.DataFrame(columns=['date','code','score']),[],config)
    result=panel[['date','code']].copy()
    result['net_return']=np.nan;result['label_end']=None;result['label_reason']=None
    result['label_shares']=0;result['label_entry_date']=None
    event_days={}
    for event in events:
        for key in ['dividRegistDate','dividOperateDate','dividPayDate','dividStockMarketDate']:
            if event.get(key):event_days.setdefault(event['code'],set()).add(event[key])
    count=Counter()
    for ci,(code,group) in enumerate(result.groupby('code',sort=True)):
        f=frames[code].set_index('date').reindex(dates)
        opened=pd.to_numeric(f.open,errors='coerce').to_numpy();closed=pd.to_numeric(f.close,errors='coerce').to_numpy()
        reference=pd.to_numeric(f.preclose,errors='coerce').to_numpy()
        exists=f.code.notna().to_numpy()
        buy=np.zeros(n,dtype=bool);sell=np.zeros(n,dtype=bool)
        for row in frames[code].to_dict('records'):
            day=str(row['date']);k=indexes.get(day)
            if k is None:continue
            checker.execution_day=day
            buy[k]=checker.tradable(row,'buy')[0];sell[k]=checker.tradable(row,'sell')[0]
        action_prefix=np.zeros(n+1,dtype=int)
        for day in event_days.get(code,()):
            if day in indexes:action_prefix[indexes[day]+1]=1
        action_prefix=np.cumsum(action_prefix)

        def outcome(i):
            if limits.get(dates[i],0)==0:return None,None,'breadth_closed',0,None
            entry=i+1
            if entry>=n:return None,None,'missing_next_session',0,None
            if not buy[entry]:return None,None,'entry_not_tradable',0,None
            limit=tick(closed[i]*1.03)
            if opened[entry]>limit+.0001:return None,None,'entry_gap_limit',0,None
            budget=7500 if bm[dates[i]]>=.58 else 5500
            quantity=int((budget-config.min_commission)/(limit*(1+config.commission+config.transfer_fee+config.slippage)))//100*100
            if quantity<100:return None,None,'lot_size_or_budget',0,None
            entry_value=quantity*opened[entry];entry_fees=sum(config.fees(entry_value).values())
            price=opened[entry];peak=price;stop=False;breakeven=False;moon=False
            sigma=vol_lookup.get((dates[entry],code),.025)
            sl=max(.025,min(.05,1.5*sigma));tp=max(.09,min(.15,4*sigma))
            previous_close=closed[i]
            last=min(n-1,entry+27)
            for k in range(entry,last+1):
                if not exists[k] or not math.isfinite(closed[k]):return None,None,'missing_held_quote',quantity,dates[entry]
                if abs(reference[k]-previous_close)>.025:
                    return None,None,'corporate_or_reference_interval',quantity,dates[entry]
                horizon=12 if limits.get(dates[k-1],1)>=2 else 8
                if k>entry and (stop or k-entry>=horizon) and sell[k]:
                    if action_prefix[k+1]-action_prefix[i]>0:return None,None,'corporate_action_interval',quantity,dates[entry]
                    final=quantity*opened[k];fees=sum(config.fees(final,sell=True).values())
                    net=(final-fees-entry_value-entry_fees)/(entry_value+entry_fees)
                    return net,dates[k],'policy_stop_next_open' if stop else 'policy_horizon',quantity,dates[entry]
                close=closed[k];previous_close=close;peak=max(peak,close)
                gain=(close-price)/price;peakgain=(peak-price)/price
                if gain>=max(.03,1.2*sigma):breakeven=True
                if gain>=max(.08,2.5*sigma):moon=True
                if moon:
                    if gain<.035:stop=True
                elif breakeven:
                    if gain<.005:stop=True
                elif close<price*(1-sl):stop=True
                if peakgain>=.1 and (peak-close)/peak>=.02:stop=True
                if gain>=tp:stop=True
            return None,None,'right_censored' if last==n-1 else 'exit_delay_exceeded',quantity,dates[entry]

        for row in group.itertuples():
            net,end,reason,shares,entry=outcome(indexes[row.date])
            result.at[row.Index,'net_return']=net
            result.at[row.Index,'label_end']=end;result.at[row.Index,'label_reason']=reason
            result.at[row.Index,'label_shares']=shares;result.at[row.Index,'label_entry_date']=entry
            count[reason]+=1
        if (ci+1)%50==0:print('LABEL',ci+1,'stocks',sum(count.values()),'rows',flush=True)
    return result,dict(count)
