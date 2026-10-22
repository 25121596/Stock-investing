from pathlib import Path
import json, math
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'expanded-300-final'
DATA = ROOT / 'data' / 'pilot-2020-01-02-2025-12-31-n300-s20201009'
CAL = pd.read_csv(DATA / 'calendar.csv', dtype=str).calendar_date.tolist()
IDX = {d:i for i,d in enumerate(CAL)}


def tick(x): return math.floor(x * 100 + 0.5 + 1e-9) / 100

def read_optional(path):
    try: return pd.read_csv(path)
    except pd.errors.EmptyDataError: return pd.DataFrame()

def inspect_case(case):
    folder = OUT / case
    trades = pd.read_csv(folder / 'trades.csv')
    fills = pd.read_csv(folder / 'fills.csv')
    actions = read_optional(folder / 'actions.csv')
    equity = pd.read_csv(folder / 'equity.csv')
    errors = {'fill_pair':0, 'fees':0, 'signal_entry_calendar':0, 'early_fixed_exit':0,
              'share_notional':0, 'trade_pnl_reconcile':0, 'equity_calendar':0}
    delayed_fixed_exits=0
    fee_examples=[]
    bytrade={}
    for tid, group in fills.groupby('trade_id'):
        bytrade[int(tid)] = group.sort_values('date')
        if len(group) != 2 or list(group.side) != ['buy','sell']:
            # The only expected one-fill item is an end-of-period holding.
            if not (len(group)==1 and group.iloc[0].side=='buy' and int(tid) in set(pd.read_csv(folder/'holdings.csv').trade_id)):
                errors['fill_pair'] += 1
        for _, f in group.iterrows():
            notional=float(f.notional)
            exp={'commission':tick(max(5,notional*.00025)),
                 'transfer':tick(notional*.00001),
                 'stamp':tick(notional*.0005) if f.side=='sell' else 0.0,
                 'slippage':tick(notional*.0005)}
            if any(abs(float(getattr(f,k))-v)>0.011 for k,v in exp.items()): errors['fees']+=1
            if len(fee_examples)<4:
                fee_examples.append({'date':f.date,'side':f.side,'notional':notional,'recorded':{k:float(getattr(f,k)) for k in exp},'expected':exp})
            if abs(float(f.shares)*float(f.price)-notional)>.011: errors['share_notional']+=1
    for _,t in trades.iterrows():
        group=bytrade.get(int(t.trade_id),pd.DataFrame())
        if len(group)==2:
            b,s=group.iloc[0],group.iloc[1]
            if b.code!=t.code or s.code!=t.code or b.date!=t.entry_date: errors['fill_pair']+=1
            if IDX.get(t.signal_date,-100)+1 != IDX.get(t.entry_date,-200): errors['signal_entry_calendar']+=1
            net=(float(s.notional)-sum(float(s[k]) for k in ('commission','transfer','stamp','slippage'))
                 +float(t.net_dividends)-float(b.notional)-sum(float(b[k]) for k in ('commission','transfer','slippage')))
            if abs(net-float(t.net_pnl))>.011: errors['trade_pnl_reconcile']+=1
            hold=int(case.rsplit('-',1)[1][:-1])
            if t.exit_reason=='fixed_horizon':
                duration=IDX.get(t.exit_date,-100)-IDX.get(t.entry_date,-200)
                if duration<hold: errors['early_fixed_exit']+=1
                if duration>hold: delayed_fixed_exits+=1
    if equity.date.tolist()!=CAL[IDX[equity.date.iloc[0]]:IDX[equity.date.iloc[-1]]+1]: errors['equity_calendar']+=1
    return {'case':case,'fills':len(fills),'closed_trades':len(trades),'actions':len(actions),'checks':errors,
            'fixed_horizon_exit_delays':delayed_fixed_exits,
            'rejected_totals':None,'fee_examples':fee_examples,
            'buy_count':int((fills.side=='buy').sum()),'sell_count':int((fills.side=='sell').sum()),
            'open_positions':len(read_optional(folder/'holdings.csv'))}

summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
rows=[]
for strategy in ('volume','volume_trend','turtle'):
 for hold in (1,3,5):
  case=f'{strategy}-{hold}d'; r=inspect_case(case)
  sr=next(x for x in summary['results'] if x['case']==case)
  r['summary_return']=sr['total_return']; r['fees_vs_capital']=sr['execution_costs']/30000
  r['rejected']=sr['rejected']; rows.append(r)

# Find a substantial, non-zero dividend paid to a position actually present on the ex-right date.
div_case=None
for r in rows:
 if r['actions']:
  f=OUT/r['case']/'actions.csv'; df=pd.read_csv(f)
  d=df[df.net_cash>10]
  if len(d):
   a=d.iloc[0]; div_case={'case':r['case'], 'action':a.to_dict()}; break

# Independently compare one action to source event and ex-date raw quotes.
if div_case:
 a=div_case['action']; code=a['code']; exdate=a['date']
 src=pd.read_csv(DATA/f'{code}.dividends.csv',dtype=str).fillna('')
 ev=src[(src.dividOperateDate==exdate)&(src.code==code)].iloc[0].to_dict()
 div_case['source_event']={k:ev.get(k) for k in ('dividRegistDate','dividOperateDate','dividPayDate','dividCashPsBeforeTax','dividStocksPs','dividReserveToStockPs')}
 div_case['cash_recalculated']=float(a['entitled_shares'])*float(ev['dividCashPsBeforeTax'])*.8

result={'scope':'read-only audit of frozen engine and exported 300-stock ledgers',
        'cases':rows,'sample_dividend':div_case,
        'code_review':[
          {'finding':'signal timing','detail':'Signals are built from each date close, queued for the next calendar session open; fills confirm next-session entries.'},
          {'finding':'long-only direction','detail':'Only buy/sell stock positions; no short or inverse fills in engine.'},
          {'finding':'candidate cancellation','detail':'Daily candidate loop sorts by descending 20-day amount, queues the first eligible code then breaks. If its next-open order is rejected, lower-ranked same-day candidates are discarded rather than tried as fallback. This is a real throughput/selection limitation, not a sign inversion or direct ledger P&L error.'},
          {'finding':'stops','detail':'A close-based 3% threshold sets stop_pending, then tries a next-open sell; rejects from suspension/limit-down are retried. Loss can exceed 3%.'},
          {'finding':'period costs','detail':'Current 0.05% stamp tax and 0.001% transfer fee are applied in all historical years; this mildly understates earlier stamp tax and overstates pre-2022 Shenzhen transfer fee. Fee timing is mechanically correct in exported fills.'},
          {'finding':'corpora action caveat','detail':'Cash/bonus entitlement follows record and operate dates; short-horizon cash is modeled at flat 20% withholding. This is an approximation, not a likely source of 50-80% equity loss.'}
        ]}
path=OUT/'audit_next_step.json'
path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({'output':str(path),'cases':[{k:v for k,v in r.items() if k not in ('fee_examples',)} for r in rows], 'sample_dividend':div_case},ensure_ascii=False,indent=2))

