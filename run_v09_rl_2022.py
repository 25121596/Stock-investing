"""Preregistered 2022-only RL fit, then frozen historical evaluation."""
import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import arena
from v09_rl_2022 import RLReplay,QPolicy,VotePolicy,FollowPolicy,SEEDS,EPOCHS

ROOT=Path(__file__).resolve().parent
BUNDLE=ROOT/'results/v08-codex-rebuild-20261020-v1'
FEATURES=ROOT/'results/midterm-access-v3/features.csv'
PRED2=BUNDLE/'v2/predictions-h10-v2.csv'
PRED8=BUNDLE/'v08/predictions-h10-v08.csv'
PERIODS=[('2022_train','2022-01-01','2022-12-31'),('2023','2023-01-01','2023-12-31'),
         ('2024','2024-01-01','2024-12-31'),('2025','2025-01-01','2025-12-31'),
         ('2023-2025','2023-01-01','2025-12-31'),('2026','2026-01-01','2026-10-13')]

def utc():return datetime.now(timezone.utc).isoformat()
def object_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()

def read_2022_prefix(path,usecols=None):
    """Read date-sorted CSV up to 2022; never parse later rows as numbers."""
    lines=[];previous=''
    with Path(path).open(encoding='utf-8-sig',newline='') as handle:
        header=handle.readline();lines.append(header)
        if header.split(',')[0]!='date':raise ValueError('Expected first date column')
        for line in handle:
            day=line.split(',',1)[0]
            if day> '2022-12-31':break
            if day<previous:raise ValueError(f'Unsorted date input: {path}')
            previous=day
            if day>='2022-01-01':lines.append(line)
    return pd.read_csv(io.StringIO(''.join(lines)),dtype={'date':str,'code':str},
                       usecols=usecols,float_precision='round_trip')

def training_data():
    manifest=json.loads((arena.DATASET/'manifest.json').read_text(encoding='utf-8'))
    if manifest['failures']:raise ValueError('Incomplete source manifest')
    frames={};raw=[]
    for row in manifest['inventory']:
        code=row['code'];path=arena.DATASET/f'{code}.csv';dp=arena.DATASET/f'{code}.dividends.csv'
        if arena.digest(path)!=row['sha256'] or arena.digest(dp)!=row['dividends_sha256']:
            raise ValueError(f'Changed source data: {code}')
        f=read_2022_prefix(path)
        if f.date.duplicated().any() or (len(f) and (f.code.nunique()!=1 or f.code.iloc[0]!=code)):
            raise ValueError('Invalid training dates/codes')
        if ((f.high<f.low)|(f.close<=0)|(f.adj_close<=0)).any():raise ValueError('Invalid training OHLC')
        frames[code]=f
        # Event identities are allowed, but no future action is passed to training.
        rows=pd.read_csv(dp,dtype=str).fillna('').to_dict('records')
        raw.extend(e for e in rows if any('2022-01-01'<=e.get(k,'')<='2022-12-31'
                   for k in ['dividRegistDate','dividOperateDate','dividPayDate','dividStockMarketDate']))
    events,event_audit=arena.canonicalize_events(raw)
    with (arena.DATASET/'calendar.csv').open(encoding='utf-8-sig',newline='') as h:
        days=[r['calendar_date'] for r in csv.DictReader(h) if '2022-01-01'<=r['calendar_date']<='2022-12-31']
    panel=read_2022_prefix(FEATURES,['date','code','adj_ret5','market_ret5','ret_vol20'])
    two=read_2022_prefix(PRED2,['date','code','pred_net_return_v2'])
    eight=read_2022_prefix(PRED8,['date','code','pred_score_v08'])
    return frames,days,events,event_audit,panel,two,eight

def prepare(panel,two,eight,days):
    if panel.duplicated(['date','code']).any():raise ValueError('Duplicate context rows')
    b=arena.breadth_days(panel);bm=b.set_index('date').breadth5.to_dict();pm=b.set_index('date').pool_size.to_dict()
    limits={d:0 if pm[d]<100 or x<.45 else (1 if x<.55 else 2) for d,x in bm.items()}
    market=panel.groupby('date').market_ret5
    if market.nunique().gt(1).any():raise ValueError('Ambiguous market context')
    m=market.first().to_dict();context={d:{'breadth':bm[d],'market_ret5':m[d]} for d in days}
    if not np.isfinite([[c['breadth'],c['market_ret5']] for c in context.values()]).all():raise ValueError('Missing context')
    vol=panel.set_index(['date','code']).ret_vol20.to_dict()
    for f,col in [(two,'pred_net_return_v2'),(eight,'pred_score_v08')]:arena.validate_rows(f,[col],col)
    joined=two.merge(eight,on=['date','code'],validate='one_to_one')
    if not len(joined)==len(two)==len(eight):raise ValueError('Prediction universe changed')
    allow=joined.date.map(limits).fillna(0).gt(0)&joined.pred_net_return_v2.ge(.005)
    signals=joined.loc[allow,['date','code','pred_score_v08']].rename(columns={'pred_score_v08':'score'})
    return signals,vol,bm,limits,context

def replay(frames,bars,days,signals,events,cfg,inputs,policy,training=False,epsilon=0.):
    _,vol,bm,limits,context=inputs
    rep=RLReplay({},days,signals,events,cfg,vol_map=vol,breadth_lookup=bm,daily_limits=limits,
        enable_cash_overlay=False,policy=policy,context=context,training=training,epsilon=epsilon)
    rep.frames,rep.bars=frames,bars
    try:return rep.run()
    except (ValueError,AssertionError) as exc:exc.replay=rep;raise

def check_passive(base,passive):
    for name in ['equity','fills','trades','actions','warnings','review_warnings','pending_buys','rejected']:
        if getattr(base,name)!=getattr(passive,name):raise ValueError(f'Action-0 parity failed: {name}')
    if base.holdings()!=passive.holdings():raise ValueError('Action-0 holdings mismatch')
    return {'passed':True,'days':len(base.equity),'fills':len(base.fills),'fields':['equity','fills','trades','actions','pending','rejections','holdings']}

def acceptance_check(rows,period):
    needed=['baseline','stress_baseline','vote','stress_vote']
    if not all(n in rows for n in needed):return {'period':period,'passed':False,'missing_or_failed':True}
    b,sb,v,sv=[rows[n] for n in needed]
    c={'period':period,'higher_net_return':v['total_return']>b['total_return']+1e-12,
       'drawdown_not_worse':v['max_drawdown']>=b['max_drawdown']-1e-12,
       'higher_stressed_return':sv['total_return']>sb['total_return']+1e-12}
    c['passed']=all(c[k] for k in ['higher_net_return','drawdown_not_worse','higher_stressed_return'])
    return c

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    started=time.monotonic();out=args.output.resolve();out.mkdir(exist_ok=False);(out/'sources').mkdir();(out/'models').mkdir()
    sources=set(arena.source_dependencies())|{ROOT/n for n in ['run_v09_rl_2022.py','v09_rl_2022.py','test_v09_rl_2022.py']}
    source_hashes={f.name:arena.digest(f) for f in sources}
    for f in sources:shutil.copyfile(f,out/'sources'/f.name)
    shutil.copyfile(ROOT/'docs/v09-rl-2022-protocol.md',out/'protocol.md')
    protected=[ROOT/'results/v08-paper-forward-v3/days/2026-10-20/record.json',ROOT/'results/v08-paper-forward-v3/runtime/freeze.json',ROOT.parent/'ashare-simulator/v08-paper/gateway-freeze.json']
    protected_hashes={str(f):arena.digest(f) for f in protected}
    raw_inputs=sorted(arena.DATASET.glob('*.csv'))+[arena.DATASET/'manifest.json']
    input_hashes={str(f.relative_to(ROOT)):arena.digest(f) for f in [FEATURES,PRED2,PRED8]+raw_inputs}
    arena.write_json(out/'started.json',{'utc':utc(),'source_hashes':source_hashes,'protected_hashes':protected_hashes,
        'input_hashes':input_hashes,'protocol_sha256':arena.digest(out/'protocol.md'),'seeds':SEEDS,'epochs':EPOCHS,
        'versions':{'numpy':np.__version__,'pandas':pd.__version__},'historical_data_previously_viewed':True})
    frames,days,events,event_audit,panel,two,eight=training_data();bars=arena.prepare_bars(frames)
    inputs=prepare(panel,two,eight,days);signals,vol,bm,limits,context=inputs
    material=out/'training-inputs';material.mkdir()
    pd.concat(frames.values(),ignore_index=True).to_csv(material/'prices-2022.csv',index=False)
    panel.to_csv(material/'context-vol-2022.csv',index=False);signals.to_csv(material/'signals-2022.csv',index=False)
    arena.write_json(material/'events-2022.json',events);arena.write_json(material/'calendar-2022.json',days)
    upstream={}
    for model in ['v2','v08']:
        record=json.loads((BUNDLE/f'{model}/2022-training.json').read_text(encoding='utf-8'))
        if record['training_last_label_end']>='2022-01-01':raise ValueError('Upstream training crosses 2022 boundary')
        upstream[model]={'last_label_end':record['training_last_label_end'],'weights_sha256':arena.digest(BUNDLE/f'{model}/2022.txt')}
    arena.write_json(out/'training-scope.json',{'first':days[0],'last':days[-1],'sessions':len(days),
        'max_price_date':max(f.date.max() for f in frames.values() if len(f)),'max_signal_date':signals.date.max(),
        'max_context_date':max(context),'stocks':len(frames),'signals':len(signals),'upstream':upstream,
        'materialized_input_hashes':{f.name:arena.digest(f) for f in material.iterdir()},'events_audit':event_audit})
    cfg=arena.Config(capital=30000,order_budget=7500,max_positions=5,max_exposure=.9,hold_days=10)
    base=arena.simulate(frames,bars,days,signals,events,replace(cfg),vol,bm,limits)
    passive=replay(frames,bars,days,signals,events,replace(cfg),inputs,FollowPolicy())
    parity=check_passive(base,passive);parity['audit']=arena.audit_ledger(passive,bars,events)
    arena.write_json(out/'passive-parity-2022.json',parity)
    print('SCOPE',days[0],days[-1],len(days),'PASSIVE PARITY OK',flush=True)
    policies=[];learning=[];training_audits=0
    for seed in SEEDS:
        policy=QPolicy(seed)
        for epoch in range(EPOCHS):
            epsilon=.30-.25*epoch/(EPOCHS-1)
            rep=replay(frames,bars,days,signals,events,replace(cfg),inputs,policy,True,epsilon)
            arena.audit_ledger(rep,bars,events);training_audits+=1
            s=rep.summary();counts=pd.Series([r['action'] for r in rep.decisions]).value_counts().to_dict()
            learning.append({'seed':seed,'epoch':epoch+1,'epsilon':epsilon,'reward':rep.episode_reward,
                'total_return':s['total_return'],'max_drawdown':s['max_drawdown'],'closed_trades':s['closed_trades'],
                **{f'action_{a}':int(counts.get(a,0)) for a in range(4)}})
            if (epoch+1)%25==0:print('TRAIN',seed,epoch+1,'/200','states',len(policy.values),'elapsed',round(time.monotonic()-started),flush=True)
        if policy.update_count!=(len(days)-1)*EPOCHS:raise ValueError('Reward/update count mismatch')
        path=out/'models'/f'q-{seed}.json';arena.write_json(path,policy.dump())
        loaded=QPolicy.restore(json.loads(path.read_text(encoding='utf-8')))
        if object_hash(policy.dump())!=object_hash(loaded.dump()):raise ValueError('Q reload mismatch')
        policies.append(loaded)
    pd.DataFrame(learning).to_csv(out/'learning.csv',index=False)
    frozen={p.name:arena.digest(p) for p in (out/'models').iterdir()}
    arena.write_json(out/'training-complete.json',{'utc':utc(),'training_cutoff':'2022-12-31','model_hashes':frozen,
        'epochs_per_seed':EPOCHS,'seeds':SEEDS,'training_ledger_audits':training_audits,'ensemble':'fixed majority; lower action breaks ties',
        'freeze_before_later_data_load':True})
    print('FROZEN ALL POLICIES; first loading later evaluation prices',flush=True)
    _,frames,calendar,raw=arena.load(arena.DATASET);events,event_audit=arena.canonicalize_events(raw);bars=arena.prepare_bars(frames)
    panel=pd.read_csv(FEATURES,dtype={'date':str,'code':str},usecols=['date','code','adj_ret5','market_ret5','ret_vol20'],float_precision='round_trip')
    two=pd.read_csv(PRED2,dtype={'date':str,'code':str},usecols=['date','code','pred_net_return_v2'],float_precision='round_trip')
    eight=pd.read_csv(PRED8,dtype={'date':str,'code':str},usecols=['date','code','pred_score_v08'],float_precision='round_trip')
    all_days=[d for d in calendar if '2022-01-01'<=d<='2026-10-13'];inputs=prepare(panel,two,eight,all_days)
    all_signals,vol,bm,limits,context=inputs
    results=[];failures=[];parities=[];policy_before=[object_hash(q.dump()) for q in policies]
    for period,start,end in PERIODS:
        days=[d for d in calendar if start<=d<=end];signals=all_signals[all_signals.date.between(start,end)]
        for stress in [False,True]:
            conf=replace(cfg,slippage=.001 if stress else .0005);baseline=None
            cases=[('baseline',FollowPolicy())]+[(f'seed_{q.seed}',q) for q in policies]+[('vote',VotePolicy(policies))]
            for name,policy in cases:
                case=('stress_' if stress else '')+name;folder=out/f'{period}_{case}'
                try:
                    rep=replay(frames,bars,days,signals,events,replace(conf),inputs,policy)
                    audit=arena.audit_ledger(rep,bars,events);s=rep.summary()
                    if name=='baseline':
                        baseline=arena.simulate(frames,bars,days,signals,events,replace(conf),vol,bm,limits)
                        parities.append({'period':period,'stress':stress,**check_passive(baseline,rep)})
                    decisions=rep.decisions;counts={a:sum(d['action']==a for d in decisions) for a in range(4)}
                    fallback=0
                    for d in decisions:
                        state=tuple(d['state'])
                        if isinstance(policy,QPolicy):fallback+=not any(policy.visits.get(state,np.zeros(4))[a]>=10 for a in range(4))
                    s.update(period=period,policy=name,case=case,stress=stress,audit=audit,
                        action_counts=counts,decisions=len(decisions),unknown_state_days=fallback if isinstance(policy,QPolicy) else None,
                        average_exposure=float(np.mean([(r['equity']-r['cash']-r['receivables'])/r['equity'] for r in rep.equity])))
                    arena.save_case(folder,rep,s,signals);pd.DataFrame(decisions).to_csv(folder/'decisions.csv',index=False)
                    results.append(s);print('RESULT',period,case,round(s['total_return']*100,3),'DD',round(s['max_drawdown']*100,3),flush=True)
                except (ValueError,AssertionError) as exc:
                    failure={'period':period,'case':case,'policy':name,'stress':stress,'error':str(exc)};failures.append(failure)
                    arena.write_json(out/f'{period}_{case}-FAILED.json',failure)
                    if hasattr(exc,'replay'):arena.save_case(out/f'{period}_{case}-FAILED',exc.replay,failure,signals)
                    print('FAILED',failure,flush=True)
        arena.write_json(out/'checkpoint.json',{'results':results,'failures':failures})
    if policy_before!=[object_hash(q.dump()) for q in policies]:raise ValueError('Evaluation modified Q policies')
    if frozen!={p.name:arena.digest(p) for p in (out/'models').iterdir()}:raise ValueError('Frozen policy files changed')
    checks=[]
    for period in ['2023-2025','2026']:
        rows={r['case']:r for r in results if r['period']==period}
        checks.append(acceptance_check(rows,period))
    protections={str(f):arena.digest(f)==protected_hashes[str(f)] for f in protected}
    sources_unchanged=all(arena.digest(f)==source_hashes[f.name] for f in sources)
    inputs_unchanged=all(arena.digest(ROOT/f)==h for f,h in input_hashes.items())
    protocol_unchanged=arena.digest(out/'protocol.md')==arena.digest(ROOT/'docs/v09-rl-2022-protocol.md')
    if not all(protections.values()) or not sources_unchanged or not inputs_unchanged or not protocol_unchanged:raise ValueError('Protected source/input/runtime changed')
    summary={'version':'V0.9 RL2022研究原型','current_adopted':'V0.8','complete':len(results)==60 and not failures,
        'results':results,'failures':failures,'acceptance':{'checks':checks,'passed':all(c['passed'] for c in checks) and not failures},
        'seeds':SEEDS,'epochs':EPOCHS,'training_year':2022,'training_audits':training_audits,'model_hashes':frozen,
        'passive_parity_cases':parities,'policies_unchanged_in_eval':True,'protections':protections,'sources_unchanged':sources_unchanged,
        'inputs_unchanged':inputs_unchanged,'protocol_unchanged':protocol_unchanged,'elapsed_seconds':time.monotonic()-started,'completed_utc':utc()}
    arena.write_json(out/'summary.json',summary)
    flat=[{k:r[k] for k in ['period','policy','stress','total_return','max_drawdown','final_equity','closed_trades','win_rate','execution_costs','open_positions','quality_warning_count','average_exposure']} for r in results]
    pd.DataFrame(flat).to_csv(out/'comparison.csv',index=False)
    print('COMPLETE',len(results),'valid',len(failures),'failed','ACCEPTED',summary['acceptance']['passed'],flush=True)

if __name__=='__main__':main()
