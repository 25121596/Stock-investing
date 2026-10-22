"""Fixed 2022 logistic batch gate versus V0.8 and frozen 2022 RL filtering."""
import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import sklearn
import arena
import run_v09_rl_2022 as prior
from run_v09_rl_ablation import reference_parity
from v09_rl_2022 import RLReplay,QPolicy,VotePolicy,SEEDS
from v09_supervised_filter import ConstantGate,BreadthGate,FrozenLogit,LegacyRLGate,GateAdapter,matured_batch_labels

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'results/v09-rl-ablation-20261021-v2'
Q_SOURCE=ROOT/'results/v09-rl-2022-prototype-20261021-v2'
PERIODS=prior.PERIODS


def run_gate(frames,bars,days,signals,events,cfg,inputs,gate):
    _,vol,breadth,limits,context=inputs;adapter=GateAdapter(gate)
    rep=RLReplay({},days,signals,events,cfg,vol_map=vol,breadth_lookup=breadth,daily_limits=limits,
        enable_cash_overlay=False,policy=adapter,context=context)
    adapter.replay=rep;rep.frames,rep.bars=frames,bars
    try:return rep.run(),adapter
    except (ValueError,AssertionError) as exc:exc.replay=rep;raise


def coverage(rep,adapter,support):
    ops=[o for o in adapter.observations if o['pending_before']>0]
    accepted=[o for o in ops if o['action']==0];orders=sum(o['pending_before'] for o in ops)
    novel=sum(tuple(o['state']) not in support for o in ops) if support is not None else None
    return {'opportunity_batches':len(ops),'accepted_batches':len(accepted),'skipped_batches':len(ops)-len(accepted),
        'acceptance_rate':len(accepted)/len(ops) if ops else None,'proposed_orders':orders,
        'accepted_orders':sum(o['pending_before'] for o in accepted),
        'filled_buy_orders':sum(f['side']=='buy' for f in rep.fills),
        'novel_state_batches':novel,'novel_state_rate':novel/len(ops) if novel is not None and ops else None}


def export_case(folder,rep,adapter,signals,record):
    arena.save_case(folder,rep,record,signals)
    pd.DataFrame(rep.decisions).to_csv(folder/'decisions.csv',index=False)
    pd.DataFrame(adapter.observations).to_csv(folder/'gate-decisions.csv',index=False)


def criteria(results,policy,period):
    rows={r['case']:r for r in results if r['family']=='primary' and r['period']==period}
    names=['baseline','stress_baseline',policy,'stress_'+policy]
    if not all(n in rows for n in names):return {'period':period,'passed':False,'missing_or_failed':True}
    b,sb,c,sc=[rows[n] for n in names]
    checks={'period':period,'higher_net_return':c['total_return']>b['total_return']+1e-12,
        'drawdown_not_worse':c['max_drawdown']>=b['max_drawdown']-1e-12,
        'higher_stressed_return':sc['total_return']>sb['total_return']+1e-12}
    checks['passed']=all(checks[k] for k in ['higher_net_return','drawdown_not_worse','higher_stressed_return'])
    return checks


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    starttime=time.monotonic();out=args.output.resolve();out.mkdir(exist_ok=False);(out/'sources').mkdir();(out/'models').mkdir()
    sources=set(arena.source_dependencies())|{ROOT/n for n in ['run_v09_supervised_filter.py','v09_supervised_filter.py',
        'run_v09_rl_2022.py','v09_rl_2022.py','run_v09_rl_ablation.py','v09_rl_ablation.py','test_v09_supervised_filter.py']}
    source_hashes={p.name:arena.digest(p) for p in sources}
    for path in sources:shutil.copyfile(path,out/'sources'/path.name)
    shutil.copyfile(ROOT/'docs/v09-supervised-filter-protocol.md',out/'protocol.md');protohash=arena.digest(out/'protocol.md')
    original_start=json.loads((Q_SOURCE/'started.json').read_text(encoding='utf-8'))
    for name,digest in original_start['source_hashes'].items():
        if name in source_hashes and source_hashes[name]!=digest:raise ValueError('Shared frozen inference source changed: '+name)
    qfreeze=json.loads((Q_SOURCE/'training-complete.json').read_text(encoding='utf-8'));qpolicies=[]
    for seed in SEEDS:
        name=f'q-{seed}.json';path=Q_SOURCE/'models'/name
        if arena.digest(path)!=qfreeze['model_hashes'][name]:raise ValueError('Original Q changed')
        shutil.copyfile(path,out/'models'/name);qpolicies.append(QPolicy.restore(json.loads(path.read_text(encoding='utf-8'))))
    qobjects=[prior.object_hash(q.dump()) for q in qpolicies]
    files=[prior.FEATURES,prior.PRED2,prior.PRED8,Q_SOURCE/'training-complete.json']+sorted(arena.DATASET.glob('*.csv'))+[arena.DATASET/'manifest.json']
    for kind in ['v2','v08']:
        files.extend([prior.BUNDLE/f'{kind}/2022-training.json',prior.BUNDLE/f'{kind}/2022.txt'])
    input_hashes={str(f):arena.digest(f) for f in files};protected=original_start['protected_hashes']
    if not all(arena.digest(Path(f))==h for f,h in protected.items()):raise ValueError('Protected runtime changed')
    reference_hashes={}
    for period,_,_ in PERIODS:
        for stress in [False,True]:
            for name in ['baseline','filter_only']+[f'seed_{seed}_filter_only' for seed in SEEDS]:
                folder=SOURCE/f'{period}_{"stress_" if stress else ""}{name}'
                for filename in ['equity.csv','fills.csv','trades.csv','actions.csv','holdings.csv','warnings.csv','review_warnings.csv','decisions.csv','signals.csv','summary.json']:
                    reference_hashes[str(folder/filename)]=arena.digest(folder/filename)
    arena.write_json(out/'started.json',{'utc':datetime.now(timezone.utc).isoformat(),'source_hashes':source_hashes,
        'input_hashes':input_hashes,'protected_hashes':protected,'reference_hashes':reference_hashes,'protocol_sha256':protohash,
        'q_source':str(Q_SOURCE),'reference_source':str(SOURCE),'training_year':2022,'expected_accounts':84,
        'versions':{'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__},'all_evaluation_history_previously_viewed':True})
    # Training objects contain 2022 rows only; later years are first parsed after freeze.
    frames,days,events,event_audit,panel,two,eight=prior.training_data();bars=arena.prepare_bars(frames)
    inputs=prior.prepare(panel,two,eight,days);signals=inputs[0]
    cfg=arena.Config(capital=30000,order_budget=7500,max_positions=5,max_exposure=.9,hold_days=10)
    training_base,adapter=run_gate(frames,bars,days,signals,events,replace(cfg),inputs,ConstantGate())
    training_summary=training_base.summary();training_summary['audit']=arena.audit_ledger(training_base,bars,events)
    export_case(out/'training-baseline',training_base,adapter,signals,training_summary)
    training_parity=reference_parity(out/'training-baseline',SOURCE/'2022_train_baseline')
    labels=matured_batch_labels(training_base,adapter.observations)
    arena.write_json(out/'training-labels.json',labels);pd.DataFrame(labels).to_csv(out/'training-labels.csv',index=False)
    eligible=[r for r in labels if r['status']=='eligible']
    material=out/'training-inputs';material.mkdir()
    pd.concat([f for f in frames.values() if len(f)],ignore_index=True).to_csv(material/'prices-2022.csv',index=False)
    panel.to_csv(material/'context-2022.csv',index=False);signals.to_csv(material/'signals-2022.csv',index=False)
    arena.write_json(material/'events-2022.json',events);arena.write_json(material/'calendar-2022.json',days)
    upstream={}
    for kind in ['v2','v08']:
        meta=json.loads((prior.BUNDLE/f'{kind}/2022-training.json').read_text(encoding='utf-8'))
        if meta['training_last_label_end']>='2022-01-01':raise ValueError('Upstream training crosses label boundary')
        upstream[kind]={'last_label_end':meta['training_last_label_end'],'weights_sha256':arena.digest(prior.BUNDLE/f'{kind}/2022.txt')}
    logit=None;fit_error=None
    try:
        logit=FrozenLogit.fit(eligible)
        logit.record['training_labels_sha256']=arena.digest(out/'training-labels.json')
        if logit.record['last_label_end']>='2023-01-01':raise ValueError('Immature label used')
        arena.write_json(out/'models/logit.json',logit.record)
        logit=FrozenLogit(json.loads((out/'models/logit.json').read_text(encoding='utf-8')))
    except ValueError as exc:fit_error=str(exc)
    training={'rows':len(eligible),'total_opportunities':len(labels),'status_counts':dict(Counter(r['status'] for r in labels)),
        'class_counts':{str(k):sum(r['y']==k for r in eligible) for k in [0,1]},'unique_states':len({tuple(r['state']) for r in eligible}),
        'last_label_end':max((r['label_end'] for r in eligible),default=None),'model_file':'models/logit.json',
        'fit_available':logit is not None,'fit_error':fit_error,'upstream':upstream,
        'first_price':min(f.date.min() for f in frames.values() if len(f)),'last_price':max(f.date.max() for f in frames.values() if len(f)),
        'last_context':max(inputs[4]),'materialized_input_hashes':{p.name:arena.digest(p) for p in material.iterdir()},
        'label_source':'Completed executable V0.8 batch outcomes; baseline-policy distribution; utility proxy',
        'training_baseline_parity':training_parity,'event_audit':event_audit}
    arena.write_json(out/'training.json',training)
    hashes={p.name:arena.digest(p) for p in (out/'models').iterdir()}
    arena.write_json(out/'model-freeze.json',{'utc':datetime.now(timezone.utc).isoformat(),'model_hashes':hashes,
        'q_source':str(Q_SOURCE),'q_source_freeze_sha256':arena.digest(Q_SOURCE/'training-complete.json'),
        'fit_cutoff':'2023-01-01','threshold':.5,'model_objects_sha256':[prior.object_hash(q.dump()) for q in qpolicies],
        'logit_object_sha256':prior.object_hash(logit.record) if logit else None,'freeze_before_later_data_load':True})
    print('TRAINING',training['rows'],'mature batches; classes',training['class_counts'],'states',training['unique_states'],'FIT',training['fit_available'],flush=True)
    print('FROZEN; now loading later historical evaluation data',flush=True)
    _,frames,calendar,raw=arena.load(arena.DATASET);events,_=arena.canonicalize_events(raw);bars=arena.prepare_bars(frames)
    panel=pd.read_csv(prior.FEATURES,dtype={'date':str,'code':str},usecols=['date','code','adj_ret5','market_ret5','ret_vol20'],float_precision='round_trip')
    two=pd.read_csv(prior.PRED2,dtype={'date':str,'code':str},usecols=['date','code','pred_net_return_v2'],float_precision='round_trip')
    eight=pd.read_csv(prior.PRED8,dtype={'date':str,'code':str},usecols=['date','code','pred_score_v08'],float_precision='round_trip')
    inputs=prior.prepare(panel,two,eight,[d for d in calendar if '2022-01-01'<=d<='2026-10-13']);all_signals=inputs[0]
    supports=[set(q.values) for q in qpolicies]
    plan=[('baseline','primary',None,ConstantGate(),None),('fixed_breadth','primary',None,BreadthGate(),None),
          ('supervised','primary',None,logit,{tuple(r['state']) for r in eligible}),
          ('rl_vote','primary',None,LegacyRLGate(VotePolicy(qpolicies)),set.intersection(*supports))]
    plan += [(f'seed_{q.seed}','seed',q.seed,LegacyRLGate(q),set(q.values)) for q in qpolicies]
    results=[];failures=[];parities=[]
    for period,start,end in PERIODS:
        days=[d for d in calendar if start<=d<=end];signals=all_signals[all_signals.date.between(start,end)]
        for stress in [False,True]:
            for name,family,seed,gate,support in plan:
                case=('stress_' if stress else '')+name;folder=out/f'{period}_{case}'
                if gate is None:
                    failures.append({'period':period,'case':case,'policy':name,'kind':'training_unavailable','error':fit_error});continue
                rep=None
                try:
                    rep,adapter=run_gate(frames,bars,days,signals,events,replace(cfg,slippage=.001 if stress else .0005),inputs,gate)
                    summary=rep.summary();summary.update(period=period,case=case,policy=name,family=family,seed=seed,stress=stress,
                        audit=arena.audit_ledger(rep,bars,events),action_counts={a:sum(d['action']==a for d in rep.decisions) for a in range(4)},
                        average_exposure=float(np.mean([(r['equity']-r['cash']-r['receivables'])/r['equity'] for r in rep.equity])),
                        coverage=coverage(rep,adapter,support))
                    export_case(folder,rep,adapter,signals,summary)
                    if name in ['baseline','rl_vote'] or family=='seed':
                        refname='baseline' if name=='baseline' else ('filter_only' if name=='rl_vote' else f'seed_{seed}_filter_only')
                        parities.append(reference_parity(folder,SOURCE/f'{period}_{"stress_" if stress else ""}{refname}'))
                    results.append(summary)
                    if family=='primary':print('RESULT',period,case,round(summary['total_return']*100,3),'DD',round(summary['max_drawdown']*100,3),
                        'accepted',summary['coverage']['accepted_batches'],'/',summary['coverage']['opportunity_batches'],flush=True)
                except (ValueError,AssertionError) as exc:
                    failure={'period':period,'case':case,'policy':name,'error':str(exc)};failures.append(failure)
                    arena.write_json(out/f'{period}_{case}-FAILED.json',failure);failed=getattr(exc,'replay',rep)
                    if folder.exists():arena.write_json(folder/'FAILURE.json',failure)
                    elif failed is not None:arena.save_case(out/f'{period}_{case}-FAILED',failed,failure,signals)
                    print('FAILED',failure,flush=True)
        arena.write_json(out/'checkpoint.json',{'results':results,'failures':failures})
    verdicts={}
    for name in ['fixed_breadth','supervised','rl_vote']:
        checks=[criteria(results,name,period) for period in ['2023-2025','2026']]
        verdicts[name]={'checks':checks,'passed':all(c['passed'] for c in checks) and not failures}
    checks={'reference_parity':parities,'q_objects_unchanged':qobjects==[prior.object_hash(q.dump()) for q in qpolicies],
        'models_unchanged':hashes=={p.name:arena.digest(p) for p in (out/'models').iterdir()},
        'original_q_unchanged':all(arena.digest(Q_SOURCE/'models'/n)==h for n,h in qfreeze['model_hashes'].items()),
        'logit_object_unchanged':logit is None or prior.object_hash(logit.record)==json.loads((out/'model-freeze.json').read_text())['logit_object_sha256'],
        'inputs_unchanged':all(arena.digest(Path(p))==h for p,h in input_hashes.items()),
        'references_unchanged':all(arena.digest(Path(p))==h for p,h in reference_hashes.items()),
        'protected_unchanged':all(arena.digest(Path(p))==h for p,h in protected.items()),
        'sources_unchanged':all(arena.digest(p)==source_hashes[p.name] for p in sources),
        'protocol_unchanged':arena.digest(out/'protocol.md')==protohash==arena.digest(ROOT/'docs/v09-supervised-filter-protocol.md')}
    if not all(v for k,v in checks.items() if k!='reference_parity'):raise ValueError('Frozen file or model changed')
    record={'display_version':'V0.9监督过滤对照研究','current_adopted':'V0.8','complete':len(results)==84 and not failures,
        'results':results,'failures':failures,'verdicts':verdicts,'checks':checks,'training':training,
        'source_run':str(SOURCE),'q_source':str(Q_SOURCE),'model_hashes':hashes,
        'completed_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-starttime}
    arena.write_json(out/'summary.json',record)
    fields=['period','case','policy','family','seed','stress','total_return','max_drawdown','final_equity','closed_trades','win_rate','execution_costs','average_exposure','quality_warning_count']
    pd.DataFrame([{**{k:r[k] for k in fields},**r['coverage']} for r in results]).to_csv(out/'comparison.csv',index=False)
    print('COMPLETE',len(results),'accounts;',len(failures),'failures;',len(parities),'reference parities',flush=True)


if __name__=='__main__':main()
