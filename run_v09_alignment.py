"""Fixed, non-tuned V0.9 policy-label experiment; never changes DAY1."""
from dataclasses import replace
from datetime import datetime,timezone
import hashlib,json,shutil
import argparse
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from run import load
from engine import Config
from corporate_events import canonicalize_events
from exploration_features import breadth_days
from train_model_v2 import FEATURES_V2,enrich_features_v2
from train_model_v3 import extract_ashare_features
from train_model_v08 import FEATURES_V08,extract_microstructure_features
from validate_expanded import prepare_bars
from run_exploration import stable_random,matched_candidates
from v08_engine_verified import VerifiedV08Replay
from v09_policy_labels import policy_labels

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/v09-alignment-20261020-v1'
BUNDLE=ROOT/'results/v08-codex-rebuild-20261020-v1'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def run_case(frames,bars,days,signals,events,config,vol,bm,limits):
    rep=VerifiedV08Replay({},days,signals,events,config,vol_map=vol,daily_limits=limits,breadth_lookup=bm,enable_cash_overlay=False)
    rep.frames,rep.bars=frames,bars
    return rep.run()


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--labels-from',type=Path)
    args=parser.parse_args();OUT=args.output.resolve()
    OUT.mkdir(exist_ok=False);(OUT/'sources').mkdir()
    source_names=['run_v09_alignment.py','v09_policy_labels.py','v08_engine_verified.py','experiment_super_v7_regime.py',
        'experiment_super_v5_maxprofit.py','audit_gemini_v08_codex.py','engine.py','engine_2026.py','accessibility.py',
        'train_model_v2.py','train_model_v3.py','train_model_v08.py']
    hashes={name:digest(ROOT/name) for name in source_names}
    for name in source_names:shutil.copyfile(ROOT/name,OUT/'sources'/name)
    shutil.copyfile(ROOT/'docs/v09-policy-alignment-protocol.md',OUT/'protocol.md')
    write(OUT/'started.json',{'utc':datetime.now(timezone.utc).isoformat(),'source_hashes':hashes,
        'protocol_sha256':digest(OUT/'protocol.md'),'all_history_previously_viewed':True})
    manifest,frames,calendar,raw_events=load(ROOT/'data/forward-2026-10-13')
    events,event_audit=canonicalize_events(raw_events);bars=prepare_bars(frames)
    panel=pd.read_csv(ROOT/'results/midterm-access-v3/panel-h10.csv',dtype={'date':str,'code':str,'label_end':str})
    sixty=pd.read_csv(ROOT/'results/exploration-v1/new-features.csv',dtype={'date':str,'code':str})
    panel=enrich_features_v2(panel.merge(sixty[['date','code','adj_ret60','ret_vol60']],on=['date','code'],how='left',validate='one_to_one'))
    panel=panel.merge(extract_ashare_features(frames),on=['date','code'],how='left',validate='one_to_one')
    panel=panel.merge(extract_microstructure_features(frames),on=['date','code'],how='left',validate='one_to_one')
    if not np.isfinite(panel[FEATURES_V08].to_numpy()).all():
        # As in the original trainer, early market-history nonfinite base rows
        # remain candidates but are excluded from fitting and prediction.
        print('Nonfinite feature rows:',int((~np.isfinite(panel[FEATURES_V08].to_numpy()).all(axis=1)).sum()),flush=True)
    breadth=breadth_days(panel);bm=breadth.set_index('date').breadth5.to_dict();pm=breadth.set_index('date').pool_size.to_dict()
    limits={d:0 if pm[d]<100 or b<.45 else (1 if b<.55 else 2) for d,b in bm.items()}
    vol=panel.set_index(['date','code']).ret_vol20.to_dict()
    if args.labels_from:
        prior=args.labels_from.resolve()
        source= json.loads((prior/'started.json').read_text())
        if source['source_hashes']['v09_policy_labels.py']!=digest(ROOT/'v09_policy_labels.py'):
            raise ValueError('Cached label generator changed')
        labels=pd.read_csv(prior/'policy-labels.csv',dtype={'date':str,'code':str,'label_end':str,'label_entry_date':str})
        if not labels[['date','code']].equals(panel[['date','code']]):raise ValueError('Cached label candidate scope differs')
        counts={str(k):int(v) for k,v in labels.label_reason.value_counts().items()}
        write(OUT/'label-cache-provenance.json',{'source':str(prior),'source_started_sha256':digest(prior/'started.json'),
            'label_file_sha256':digest(prior/'policy-labels.csv'),'generator_sha256':digest(ROOT/'v09_policy_labels.py')})
    else:
        labels,counts=policy_labels(panel,frames,calendar,events,breadth,vol)
    labels.to_csv(OUT/'policy-labels.csv',index=False,encoding='utf-8-sig');write(OUT/'label-counts.json',counts)
    # An independent real-ledger oracle, not a mirror implementation test.
    complete=labels[labels.net_return.notna()]
    selected=pd.concat([g.sample(n=min(12,len(g)),random_state=20261020)
        for _,g in complete.groupby([complete.date.str[:4],complete.label_reason])])
    mismatches=[];label_checks=[]
    config=Config(capital=30000,order_budget=7500,max_positions=5,max_exposure=.9,hold_days=10)
    index={d:i for i,d in enumerate(calendar)}
    for row in selected.itertuples():
        start=index[row.date];days=calendar[start:min(len(calendar),start+30)]
        signal=pd.DataFrame([{'date':row.date,'code':row.code,'score':1.}])
        rep=run_case({row.code:frames[row.code]},{row.code:bars[row.code]},days,signal,
            [e for e in events if e['code']==row.code],replace(config),vol,bm,limits)
        ok=len(rep.trades)==1 and abs(rep.trades[0]['net_return']-row.net_return)<1e-10 and rep.trades[0]['exit_date']==row.label_end and rep.trades[0]['initial_shares']==row.label_shares
        item={'date':row.date,'code':row.code,'matched':bool(ok),'reason':row.label_reason}
        label_checks.append(item)
        if not ok:mismatches.append({**item,'expected_net':row.net_return,'actual':rep.trades})
    write(OUT/'label-oracle.json',{'checked':len(label_checks),'mismatches':mismatches,'checks':label_checks})
    if mismatches:raise ValueError('Policy labels differ from execution oracle; no training allowed')
    print('ORACLE PASSED',len(label_checks),flush=True)
    training=panel.drop(columns=['net_return','label_end','label_reason','y_win']).merge(labels,on=['date','code'],validate='one_to_one')
    old2=pd.read_csv(BUNDLE/'v2/predictions-h10-v2.csv',dtype={'date':str,'code':str})
    old8=pd.read_csv(BUNDLE/'v08/predictions-h10-v08.csv',dtype={'date':str,'code':str})
    train_records=[];predictions=[]
    for year in range(2022,2027):
        start=f'{year}-01-01';end=f'{year}-12-31' if year<2026 else '2026-10-13'
        valid=np.isfinite(training[FEATURES_V08].to_numpy()).all(axis=1)
        train=training[training.date.lt(start)&training.label_end.notna()&training.label_end.lt(start)&training.net_return.notna()&valid]
        evaluate=training[training.date.between(start,end)&valid].sort_values(['date','code']).copy()
        if train.empty:raise ValueError('No policy labels before test year')
        params2=json.loads((BUNDLE/f'v2/{year}-training.json').read_text())['parameters']
        params8=json.loads((BUNDLE/f'v08/{year}-training.json').read_text())['parameters']
        gate=lgb.LGBMRegressor(**params2)
        gate.fit(train[FEATURES_V2],train.net_return.clip(-.2,.2))
        ranked=train.sort_values(['date','code']).copy()
        relevance=ranked.groupby('date').net_return.transform(lambda s:pd.qcut(s,q=5,labels=False,duplicates='drop')).fillna(2).astype(int)
        rank=lgb.LGBMRanker(**params8)
        rank.fit(ranked[FEATURES_V08].to_numpy(),relevance.to_numpy(),group=ranked.groupby('date').size().to_numpy())
        folder=OUT/'models'/str(year);folder.mkdir(parents=True)
        gate.booster_.save_model(str(folder/'gate.txt'));rank.booster_.save_model(str(folder/'rank.txt'))
        evaluate['pred_gate_v09']=gate.predict(evaluate[FEATURES_V2]);evaluate['pred_rank_v09']=rank.booster_.predict(evaluate[FEATURES_V08].to_numpy(),num_threads=1)
        predictions.append(evaluate[['date','code','pred_gate_v09','pred_rank_v09']])
        record={'year':year,'training_rows':len(train),'last_signal':train.date.max(),'last_label_end':train.label_end.max(),
            'gate_sha256':digest(folder/'gate.txt'),'rank_sha256':digest(folder/'rank.txt'),'gate_features':FEATURES_V2,'rank_features':FEATURES_V08,
            'gate_parameters':params2,'rank_parameters':params8}
        train_records.append(record);write(folder/'training.json',record)
        print('TRAINED',year,'rows',len(train),flush=True)
    pred=pd.concat(predictions,ignore_index=True);pred.to_csv(OUT/'predictions.csv',index=False,encoding='utf-8-sig')
    joined=old2[['date','code','pred_net_return_v2']].merge(old8[['date','code','pred_score_v08']],on=['date','code'],validate='one_to_one').merge(pred,on=['date','code'],validate='one_to_one')
    chosen={}
    for name,gate,rank in [('baseline','pred_net_return_v2','pred_score_v08'),('gate_only','pred_gate_v09','pred_score_v08'),('both_aligned','pred_gate_v09','pred_rank_v09')]:
        candidates=joined[joined[gate].ge(.005)&joined.date.isin({d for d,n in limits.items() if n})]
        chosen[name]=candidates[['date','code',rank]].rename(columns={rank:'score'})
    results=[];failures=[]
    for period,start,end in [('2022-2025','2022-01-01','2025-12-31'),('2026','2026-01-01','2026-10-13')]:
        days=[d for d in calendar if start<=d<=end];pool=panel[panel.date.between(start,end)]
        cases=[('cash','cash','cash',chosen['baseline'].iloc[:0],config)]
        for policy,signals in chosen.items():
            signals=signals[signals.date.between(start,end)]
            cases.extend([(policy,policy,'policy',signals,config),('stress_'+policy,policy,'stress',signals,replace(config,slippage=.001))])
            for seed in [2026102001,2026102002,2026102003]:
                cases.append((f'random_{policy}_{seed}',policy,'random',matched_candidates(signals,stable_random(pool,seed)),config))
        for name,policy,kind,signals,cfg in cases:
            try:
                rep=run_case(frames,bars,days,signals,events,replace(cfg),vol,bm,limits);summary=rep.summary()
                eq=pd.DataFrame(rep.equity);summary['average_exposure']=float(((eq.equity-eq.cash-eq.receivables)/eq.equity).mean())
                summary.update(period=period,case=name,policy=policy,kind=kind,signal_candidates=len(signals))
                results.append(summary);folder=OUT/f'{period}_{name}';folder.mkdir()
                for key in ['trades','fills','equity','actions']:pd.DataFrame(getattr(rep,key)).to_csv(folder/f'{key}.csv',index=False)
                print('RESULT',period,name,round(summary['total_return']*100,3),'%',summary['closed_trades'],'trades',flush=True)
            except ValueError as exc:
                failures.append({'period':period,'case':name,'error':str(exc)})
                if kind!='random':raise
    verdict={}
    for policy in ['gate_only','both_aligned']:
        checks=[]
        for period in ['2022-2025','2026']:
            base=next(r for r in results if r['period']==period and r['case']=='baseline')
            model=next(r for r in results if r['period']==period and r['case']==policy)
            stressbase=next(r for r in results if r['period']==period and r['case']=='stress_baseline')
            stressmodel=next(r for r in results if r['period']==period and r['case']=='stress_'+policy)
            checks.append({'period':period,'higher_net_return':model['total_return']>base['total_return'],
                'drawdown_not_worse':model['max_drawdown']>=base['max_drawdown']-1e-12,
                'higher_stressed_return':stressmodel['total_return']>stressbase['total_return']})
        verdict[policy]={'checks':checks,'passed_fixed_criteria':all(all(x[k] for k in ['higher_net_return','drawdown_not_worse','higher_stressed_return']) for x in checks)}
    if hashes!={name:digest(ROOT/name) for name in source_names}:raise ValueError('Source changed during experiment')
    write(OUT/'summary.json',{'results':results,'failures':failures,'verdict':verdict,'train_records':train_records,'label_counts':counts,
        'source_hashes':hashes,'event_audit':event_audit,'all_history_previously_viewed':True,'day1_changed':False})
    pd.DataFrame(results)[['period','case','kind','total_return','max_drawdown','closed_trades','win_rate','execution_costs','average_exposure']].to_csv(OUT/'comparison.csv',index=False,encoding='utf-8-sig')
    print('VERDICT',verdict,flush=True)


if __name__=='__main__':main()
