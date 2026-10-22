"""Run preregistered action-bit ablations of frozen 2022-only RL policies."""
import argparse
from dataclasses import replace
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import arena
import run_v09_rl_2022 as original
from v09_rl_2022 import QPolicy,VotePolicy,FollowPolicy,SEEDS
from v09_rl_ablation import MaskedPolicy,MODES,compare_criteria,factorial_effects

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'results/v09-rl-2022-prototype-20261021-v2'
PERIODS=original.PERIODS


def read_csv(path):
    try:return pd.read_csv(path,float_precision='round_trip')
    except pd.errors.EmptyDataError:return pd.DataFrame()


def reference_parity(folder,reference):
    checked=[]
    for name in ['equity','fills','trades','actions','holdings','warnings','review_warnings','decisions','signals']:
        a,b=[read_csv(p/f'{name}.csv') for p in [folder,reference]]
        pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=True)
        checked.append(name)
    a=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    b=json.loads((reference/'summary.json').read_text(encoding='utf-8'))
    for key in ['rejected','config','total_return','max_drawdown','final_equity','execution_costs','closed_trades','win_rate']:
        if a[key]!=b[key]:raise ValueError(f'Reference summary differs: {key}')
    return {'case':folder.name,'reference':str(reference),'fields':checked,'passed':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    started=time.monotonic();out=args.output.resolve();out.mkdir(exist_ok=False);(out/'sources').mkdir();(out/'models').mkdir()
    sources=set(arena.source_dependencies())|{ROOT/n for n in ['run_v09_rl_ablation.py','v09_rl_ablation.py',
        'run_v09_rl_2022.py','v09_rl_2022.py','test_v09_rl_ablation.py']}
    source_hashes={p.name:arena.digest(p) for p in sources}
    if len(source_hashes)!=len(sources):raise ValueError('Source basename collision')
    prior=json.loads((SOURCE/'started.json').read_text(encoding='utf-8'))
    # Shared inference/execution sources must still be those of the frozen run.
    for name,digest in prior['source_hashes'].items():
        if name in source_hashes and source_hashes[name]!=digest:raise ValueError(f'Frozen shared source changed: {name}')
    for path in sources:shutil.copyfile(path,out/'sources'/path.name)
    shutil.copyfile(ROOT/'docs/v09-rl-ablation-protocol.md',out/'protocol.md')
    protosha=arena.digest(out/'protocol.md')
    source_summary=json.loads((SOURCE/'summary.json').read_text(encoding='utf-8'))
    if not source_summary['complete'] or source_summary['failures']:raise ValueError('Frozen source experiment incomplete')
    frozen=json.loads((SOURCE/'training-complete.json').read_text(encoding='utf-8'))
    if tuple(frozen['seeds'])!=SEEDS:raise ValueError('Frozen seed list changed')
    models=[]
    for seed in SEEDS:
        name=f'q-{seed}.json';path=SOURCE/'models'/name
        if arena.digest(path)!=frozen['model_hashes'][name]:raise ValueError('Frozen model hash mismatch')
        shutil.copyfile(path,out/'models'/name)
        models.append(QPolicy.restore(json.loads(path.read_text(encoding='utf-8'))))
    model_hashes={p.name:arena.digest(p) for p in (out/'models').iterdir()}
    object_hashes=[original.object_hash(q.dump()) for q in models]
    arena.write_json(out/'model-freeze.json',{'source_run':str(SOURCE),'model_hashes':model_hashes,
        'source_freeze_sha256':arena.digest(SOURCE/'training-complete.json'),'seeds':SEEDS,'no_training':True})
    # Include each raw price/action file, calendar and cached causal inputs.
    raw_inputs=sorted(arena.DATASET.glob('*.csv'))+[arena.DATASET/'manifest.json']
    input_paths=[original.FEATURES,original.PRED2,original.PRED8,SOURCE/'summary.json',SOURCE/'started.json',SOURCE/'training-complete.json']+raw_inputs
    input_hashes={str(p):arena.digest(p) for p in input_paths}
    protected_hashes=prior['protected_hashes']
    if any(arena.digest(Path(p))!=h for p,h in protected_hashes.items()):raise ValueError('Protected runtime changed since frozen experiment')
    reference_hashes={}
    for r in source_summary['results']:
        folder=SOURCE/f'{r["period"]}_{r["case"]}'
        for name in ['equity','fills','trades','actions','holdings','warnings','review_warnings','decisions','signals','summary']:
            path=folder/f'{name}.json' if name=='summary' else folder/f'{name}.csv'
            reference_hashes[str(path)]=arena.digest(path)
    plan=[('baseline','primary',None,'baseline',FollowPolicy())]
    vote=VotePolicy(models)
    plan.extend((mode,'primary',None,mode,vote) for mode in MODES if mode!='baseline')
    plan.extend((f'seed_{q.seed}_{mode}','seed',q.seed,mode,q) for q in models for mode in MODES if mode!='baseline')
    arena.write_json(out/'started.json',{'utc':datetime.now(timezone.utc).isoformat(),'source_hashes':source_hashes,
        'input_hashes':input_hashes,'protected_hashes':protected_hashes,'reference_hashes':reference_hashes,
        'protocol_sha256':protosha,'model_hashes':model_hashes,'plan':[{'policy':n,'family':f,'seed':s,'mode':m,'mask':MODES[m]} for n,f,s,m,_ in plan],
        'expected_accounts':156,'historical_data_previously_viewed':True,'no_training':True,
        'versions':{'numpy':np.__version__,'pandas':pd.__version__}})
    print('FROZEN',len(models),'Q policies; 48 primary + 108 seed-diagnostic accounts',flush=True)
    _,frames,calendar,raw=arena.load(arena.DATASET);events,event_audit=arena.canonicalize_events(raw);bars=arena.prepare_bars(frames)
    panel=pd.read_csv(original.FEATURES,dtype={'date':str,'code':str},usecols=['date','code','adj_ret5','market_ret5','ret_vol20'],float_precision='round_trip')
    two=pd.read_csv(original.PRED2,dtype={'date':str,'code':str},usecols=['date','code','pred_net_return_v2'],float_precision='round_trip')
    eight=pd.read_csv(original.PRED8,dtype={'date':str,'code':str},usecols=['date','code','pred_score_v08'],float_precision='round_trip')
    days=[d for d in calendar if '2022-01-01'<=d<='2026-10-13'];inputs=original.prepare(panel,two,eight,days)
    all_signals,_,_,_,_=inputs
    cfg=arena.Config(capital=30000,order_budget=7500,max_positions=5,max_exposure=.9,hold_days=10)
    results=[];failures=[];parities=[]
    for period,start,end in PERIODS:
        days=[d for d in calendar if start<=d<=end];signals=all_signals[all_signals.date.between(start,end)]
        for stress in [False,True]:
            for name,family,seed,mode,policy in plan:
                case=('stress_' if stress else '')+name;folder=out/f'{period}_{case}'
                wrapper=MaskedPolicy(policy,MODES[mode]);rep=None
                try:
                    conf=replace(cfg,slippage=.001 if stress else .0005)
                    rep=original.replay(frames,bars,days,signals,events,conf,inputs,wrapper)
                    if len(wrapper.trace)!=len(rep.decisions):raise ValueError('Missing raw action trace')
                    for trace,decision in zip(wrapper.trace,rep.decisions):
                        if trace['state']!=decision['state'] or trace['action']!=decision['action']:raise ValueError('Decision/trace mismatch')
                    audit=arena.audit_ledger(rep,bars,events);record=rep.summary()
                    counts={a:sum(d['action']==a for d in rep.decisions) for a in range(4)}
                    record.update(period=period,case=case,policy=name,family=family,seed=seed,mode=mode,mask=MODES[mode],stress=stress,
                        action_counts=counts,decisions=len(rep.decisions),audit=audit,
                        average_exposure=float(np.mean([(r['equity']-r['cash']-r['receivables'])/r['equity'] for r in rep.equity])))
                    arena.save_case(folder,rep,record,signals)
                    pd.DataFrame(rep.decisions).to_csv(folder/'decisions.csv',index=False)
                    pd.DataFrame([{**d,**t} for d,t in zip(rep.decisions,wrapper.trace)]).to_csv(folder/'raw-decisions.csv',index=False)
                    if mode=='baseline' or mode=='full_rl':
                        reference_name='baseline' if mode=='baseline' else ('vote' if family=='primary' else f'seed_{seed}')
                        reference_case=('stress_' if stress else '')+reference_name
                        parities.append(reference_parity(folder,SOURCE/f'{period}_{reference_case}'))
                    if object_hashes!=[original.object_hash(q.dump()) for q in models]:raise ValueError('Q policy changed during evaluation')
                    results.append(record)
                    if family=='primary':print('PRIMARY',period,case,round(record['total_return']*100,3),'DD',round(record['max_drawdown']*100,3),'exposure',round(record['average_exposure']*100,2),flush=True)
                except (ValueError,AssertionError) as exc:
                    failure={'period':period,'case':case,'policy':name,'family':family,'seed':seed,'mode':mode,'stress':stress,'error':str(exc)}
                    failures.append(failure);arena.write_json(out/f'{period}_{case}-FAILED.json',failure)
                    failed_replay=getattr(exc,'replay',rep)
                    if folder.exists():arena.write_json(folder/'FAILURE.json',failure)
                    elif failed_replay is not None:arena.save_case(out/f'{period}_{case}-FAILED',failed_replay,failure,signals)
                    print('FAILED',failure,flush=True)
            print('PERIOD',period,'stress',stress,'valid',len(results),'failed',len(failures),flush=True)
        arena.write_json(out/'checkpoint.json',{'results':results,'failures':failures})
    verdicts={}
    for mode in MODES:
        if mode=='baseline':continue
        checks=[compare_criteria(results,mode,p) for p in ['2023-2025','2026']]
        verdicts[mode]={'checks':checks,'passed':all(c['passed'] for c in checks) and not failures}
    effects=[]
    for period,_,_ in PERIODS:
        for stress in [False,True]:
            rows={r['mode']:r for r in results if r['family']=='primary' and r['period']==period and r['stress']==stress}
            if len(rows)==4:effects.append({'period':period,'stress':stress,**factorial_effects(rows)})
    checks={'reference_parity':parities,'q_objects_unchanged':object_hashes==[original.object_hash(q.dump()) for q in models],
        'model_files_unchanged':model_hashes=={p.name:arena.digest(p) for p in (out/'models').iterdir()},
        'original_models_unchanged':all(arena.digest(SOURCE/'models'/n)==h for n,h in model_hashes.items()),
        'sources_unchanged':all(arena.digest(p)==source_hashes[p.name] for p in sources),
        'inputs_unchanged':all(arena.digest(Path(p))==h for p,h in input_hashes.items()),
        'references_unchanged':all(arena.digest(Path(p))==h for p,h in reference_hashes.items()),
        'protected_unchanged':all(arena.digest(Path(p))==h for p,h in protected_hashes.items()),
        'protocol_unchanged':arena.digest(out/'protocol.md')==protosha==arena.digest(ROOT/'docs/v09-rl-ablation-protocol.md')}
    if not all(v for k,v in checks.items() if k!='reference_parity'):raise ValueError('Frozen source/input/protection check failed')
    summary={'display_version':'V0.9动作拆分研究','current_adopted':'V0.8','complete':len(results)==156 and not failures,
        'results':results,'failures':failures,'verdicts':verdicts,'effects':effects,'checks':checks,
        'source_run':str(SOURCE),'source_model_hashes':model_hashes,'periods':PERIODS,'no_training':True,
        'event_audit':event_audit,'completed_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-started}
    arena.write_json(out/'summary.json',summary)
    fields=['period','case','policy','family','seed','mode','mask','stress','total_return','max_drawdown','final_equity','closed_trades','win_rate','execution_costs','average_exposure','quality_warning_count']
    flat=pd.DataFrame([{k:r[k] for k in fields} for r in results]);flat.to_csv(out/'comparison.csv',index=False)
    flat[flat.family.eq('primary')].to_csv(out/'primary-comparison.csv',index=False);pd.DataFrame(effects).to_csv(out/'effects.csv',index=False)
    print('COMPLETE',len(results),'valid',len(failures),'failed;',len(parities),'reference parity cases;',round(time.monotonic()-started,1),'seconds',flush=True)


if __name__=='__main__':main()
