"""Preregistered ranking research; no tuning or changes to prospective sessions."""
import argparse,hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
from dataclasses import replace
import lightgbm as lgb
import numpy as np
import pandas as pd
import arena
from train_model_v2 import enrich_features_v2
from train_model_v3 import extract_ashare_features
from train_model_v08 import FEATURES_V08,extract_microstructure_features
from v10_rank_objectives import FIRST,SECOND,relevance,fit_rows,parameters,passes

ROOT=Path(__file__).resolve().parent
BUNDLE=ROOT/'results/v08-codex-rebuild-20261020-v1'
PERIODS=[('2022-2025','2022-01-01','2025-12-31'),('2026','2026-01-01','2026-10-13')]

def xhash(x):return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output.resolve();out.mkdir(exist_ok=False);(out/'sources').mkdir()
    sources=set(arena.source_dependencies())|{ROOT/n for n in ['run_v10_rank_objectives.py','v10_rank_objectives.py','train_model_v2.py','train_model_v3.py','train_model_v08.py']}
    hashes={f.name:arena.digest(f) for f in sources}
    for f in sources:shutil.copyfile(f,out/'sources'/f.name)
    shutil.copyfile(ROOT/'docs/v10-ranking-protocol.md',out/'protocol.md')
    protected=[ROOT/'results/v08-paper-forward-v3/days/2026-10-20/record.json',ROOT/'results/v08-paper-forward-v3/runtime/freeze.json',ROOT.parent/'ashare-simulator/v08-paper/gateway-freeze.json']
    protected_hashes={str(p):arena.digest(p) for p in protected}
    input_paths=[ROOT/'results/midterm-access-v3/panel-h10.csv',ROOT/'results/exploration-v1/new-features.csv',BUNDLE/'v2/predictions-h10-v2.csv',BUNDLE/'v08/predictions-h10-v08.csv',arena.DATASET/'manifest.json']
    input_hashes={str(p.relative_to(ROOT)):arena.digest(p) for p in input_paths}
    arena.write_json(out/'started.json',{'utc':datetime.now(timezone.utc).isoformat(),'source_hashes':hashes,'input_hashes':input_hashes,
        'protocol_sha256':arena.digest(out/'protocol.md'),'protected_hashes':protected_hashes,'all_history_previously_viewed':True,
        'versions':{'lightgbm':lgb.__version__,'numpy':np.__version__,'pandas':pd.__version__}})
    _,frames,calendar,raw=arena.load(arena.DATASET);events,event_audit=arena.canonicalize_events(raw);bars=arena.prepare_bars(frames)
    panel=pd.read_csv(input_paths[0],dtype={'date':str,'code':str,'label_end':str})
    sixty=pd.read_csv(input_paths[1],dtype={'date':str,'code':str})
    panel=enrich_features_v2(panel.merge(sixty[['date','code','adj_ret60','ret_vol60']],on=['date','code'],how='left',validate='one_to_one'))
    panel=panel.merge(extract_ashare_features(frames),on=['date','code'],how='left',validate='one_to_one').merge(extract_microstructure_features(frames),on=['date','code'],how='left',validate='one_to_one')
    if panel[FEATURES_V08[-10:]].isna().any().any():raise ValueError('Unexpected missing new features; do not apply all-period fill')
    b=arena.breadth_days(panel);bm=b.set_index('date').breadth5.to_dict();pm=b.set_index('date').pool_size.to_dict()
    limits={d:0 if pm[d]<100 or x<.45 else (1 if x<.55 else 2) for d,x in bm.items()}
    vol=panel.set_index(['date','code']).ret_vol20.to_dict()
    old2=pd.read_csv(input_paths[2],dtype={'date':str,'code':str},float_precision='round_trip')
    old8=pd.read_csv(input_paths[3],dtype={'date':str,'code':str},float_precision='round_trip')
    joined=old2[['date','code','pred_net_return_v2']].merge(old8[['date','code','pred_score_v08']],on=['date','code'],validate='one_to_one')
    allow=joined.date.map(limits).fillna(0).gt(0)&joined.pred_net_return_v2.ge(.005)
    base_signals=joined.loc[allow,['date','code','pred_score_v08']].rename(columns={'pred_score_v08':'score'})
    cfg=arena.Config(capital=30000,order_budget=7500,max_positions=5,max_exposure=.9,hold_days=10)
    results=[];failures=[];training=[];predictions=[];parity=[];folds={}
    for year in range(2022,2027):
        train=fit_rows(panel,FEATURES_V08,year)
        end=f'{year}-12-31' if year<2026 else '2026-10-13'
        eval_df=panel[panel.date.between(f'{year}-01-01',end)&np.isfinite(panel[FEATURES_V08].to_numpy()).all(axis=1)].sort_values(['date','code']).copy()
        metadata=json.loads((BUNDLE/f'v08/{year}-training.json').read_text(encoding='utf-8'))
        x=train[FEATURES_V08].to_numpy(dtype=float)
        if len(train)!=metadata['training_rows'] or xhash(x)!=metadata['training_x_sha256'] or xhash(relevance(train))!=metadata['training_y_sha256']:
            raise ValueError('Training samples/features/old labels differ from V0.8')
        model=lgb.Booster(model_file=str(BUNDLE/f'v08/{year}.txt'))
        fresh=eval_df[['date','code']].copy();fresh['score']=model.predict(eval_df[FEATURES_V08].to_numpy(),num_threads=1)
        check=fresh.merge(old8[old8.date.str.startswith(str(year))],on=['date','code'],validate='one_to_one')
        error=float(np.max(np.abs(check.score-check.pred_score_v08)))
        if len(check)!=len(eval_df) or error>1e-10:raise ValueError('Baseline saved model failed prediction parity')
        parity.append({'year':year,'rows':len(check),'max_abs_error':error,'training_rows':len(train),'training_last_label_end':train.label_end.max()})
        folds[year]=(train,eval_df,metadata['parameters'],json.loads((BUNDLE/f'v2/{year}-training.json').read_text(encoding='utf-8'))['parameters'])
    arena.write_json(out/'baseline-parity.json',parity)
    print('BASELINE PARITY',sum(x['rows'] for x in parity),'predictions; same sample hashes',flush=True)
    def evaluate(name,signals):
        for period,start,end in PERIODS:
            days=[d for d in calendar if start<=d<=end];s=signals[signals.date.between(start,end)]
            for stress in [False,True]:
                case=('stress_' if stress else '')+name;folder=out/f'{period}_{case}'
                rep=None
                try:
                    rep=arena.simulate(frames,bars,days,s,events,replace(cfg,slippage=.001 if stress else .0005),vol,bm,limits)
                    summary=rep.summary();audit=arena.audit_ledger(rep,bars,events);eq=pd.DataFrame(rep.equity)
                    summary.update(period=period,case=case,policy=name,stress=stress,audit=audit,
                        average_exposure=float(((eq.equity-eq.cash-eq.receivables)/eq.equity).mean()),signal_candidates=len(s))
                    arena.save_case(folder,rep,summary,s);results.append(summary)
                    print('RESULT',period,case,round(summary['total_return']*100,3),'DD',round(summary['max_drawdown']*100,3),flush=True)
                except (ValueError,AssertionError) as exc:
                    failure={'period':period,'case':case,'error':str(exc)};failures.append(failure);arena.write_json(out/f'{period}_{case}-FAILED.json',failure)
                    if hasattr(exc,'replay'):arena.save_case(out/f'{period}_{case}-FAILED',exc.replay,failure,s)
                    print('FAILED',failure,flush=True)
        arena.write_json(out/'checkpoint.json',{'results':results,'failures':failures})
    def verdict(name):
        checks=[]
        for period,_,_ in PERIODS:
            rows={r['case']:r for r in results if r['period']==period}
            if not all(k in rows for k in ['baseline','stress_baseline',name,'stress_'+name]):return {'passed':False,'missing_or_failed_case':True}
            checks.append({'period':period,**passes(rows['baseline'],rows[name],rows['stress_baseline'],rows['stress_'+name])})
        return {'checks':checks,'passed':all(all(c[k] for k in ['higher_net_return','drawdown_not_worse','higher_stressed_return']) for c in checks)}
    evaluate('baseline',base_signals)
    expected=json.loads((ROOT/'results/arena/run-20261020T145952030119Z/summary.json').read_text(encoding='utf-8'))
    for r in results:
        name='v08_slip2' if r['stress'] else 'v08'
        e=next(x for x in expected['results'] if x['period']==r['period'] and x['track']=='common' and x['case']==name)
        if any(abs(r[k]-e[k])>1e-9 for k in ['total_return','max_drawdown','final_equity']):raise ValueError('Baseline account differs from audited arena')
    def train_case(name):
        parts=[]
        for year,(train,evaluate_df,rp,pp) in folds.items():
            params=parameters(name,rp,pp);x=train[FEATURES_V08].to_numpy(dtype=float)
            if name=='return_regressor':
                model=lgb.LGBMRegressor(**params);target=train.net_return.clip(-.2,.2).to_numpy()
                model.fit(x,target)
            else:
                model=lgb.LGBMRanker(**params);target=relevance(train,tail=name.startswith('tail_'))
                model.fit(x,target,group=train.groupby('date',sort=True).size().to_numpy())
            folder=out/'models'/name;folder.mkdir(parents=True,exist_ok=True)
            path=folder/f'{year}.txt';model.booster_.save_model(str(path))
            saved=lgb.Booster(model_file=str(path))
            p=evaluate_df[['date','code']].copy();p['score']=saved.predict(evaluate_df[FEATURES_V08].to_numpy(dtype=float),num_threads=1);p['policy']=name
            parts.append(p)
            record={'policy':name,'year':year,'training_rows':len(train),'last_label_end':train.label_end.max(),'last_signal':train.date.max(),
                'training_x_sha256':xhash(x),'training_y_sha256':xhash(target),'features':FEATURES_V08,'parameters':params,'model_sha256':arena.digest(path)}
            training.append(record);arena.write_json(folder/f'{year}-training.json',record)
            print('TRAINED',name,year,len(train),flush=True)
        pred=pd.concat(parts,ignore_index=True);pred.to_csv(out/f'predictions-{name}.csv',index=False)
        predictions.append(name)
        merged=joined[['date','code']].merge(pred,on=['date','code'],validate='one_to_one')
        if len(merged)!=len(joined):raise ValueError('Candidate prediction scope changed')
        signals=merged.loc[allow.to_numpy(),['date','code','score']]
        if not signals[['date','code']].reset_index(drop=True).equals(base_signals[['date','code']].reset_index(drop=True)):raise ValueError('Gate changed')
        evaluate(name,signals)
    for name in FIRST:train_case(name)
    phase1={n:verdict(n) for n in FIRST};arena.write_json(out/'phase1-verdict.json',phase1)
    if not any(v['passed'] for v in phase1.values()):
        for name in SECOND:train_case(name)
    # Shared controls: every ranker has exactly the same gate dates and counts.
    random_results=[]
    for period,start,end in PERIODS:
        days=[d for d in calendar if start<=d<=end];pool=joined[joined.date.between(start,end)];signals=base_signals[base_signals.date.between(start,end)]
        for seed in arena.SEEDS:
            s=arena.matched_candidates(signals,arena.stable_random(pool,seed));folder=out/f'{period}_random_{seed}'
            try:
                rep=arena.simulate(frames,bars,days,s,events,replace(cfg),vol,bm,limits);r=rep.summary();r.update(period=period,seed=seed,audit=arena.audit_ledger(rep,bars,events))
                arena.save_case(folder,rep,r,s);random_results.append(r)
            except (ValueError,AssertionError) as exc:
                failure={'period':period,'seed':seed,'case':'shared_random','error':str(exc)};failures.append(failure)
                arena.write_json(out/f'{period}_random_{seed}-FAILED.json',failure)
                if hasattr(exc,'replay'):arena.save_case(out/f'{period}_random_{seed}-FAILED',exc.replay,failure,s)
    verdicts={n:verdict(n) for n in predictions}
    if hashes!={f.name:arena.digest(f) for f in sources} or input_hashes!={str(p.relative_to(ROOT)):arena.digest(p) for p in input_paths} or protected_hashes!={str(p):arena.digest(p) for p in protected}:
        raise ValueError('Sources/inputs/prospective evidence changed during experiment')
    result={'results':results,'failures':failures,'verdicts':verdicts,'training':training,'random_controls':random_results,'parity':parity,
        'event_audit':event_audit,'source_hashes':hashes,'input_hashes':input_hashes,'protected_unchanged':True,'all_history_previously_viewed':True,
        'adopted_into_forward_session':False,'trial_count_new_models':len(predictions)}
    arena.write_json(out/'summary.json',result)
    pd.DataFrame(results)[['period','case','total_return','max_drawdown','closed_trades','win_rate','execution_costs','average_exposure']].to_csv(out/'comparison.csv',index=False,encoding='utf-8-sig')
    print('FINAL VERDICTS',json.dumps(verdicts),flush=True)
    print('SAVED',out,flush=True)

if __name__=='__main__':main()
