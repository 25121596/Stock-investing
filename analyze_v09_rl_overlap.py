"""Describe frozen-account trade overlap; no retraining or counterfactual fills."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve();out=args.output.resolve();out.mkdir(exist_ok=False)
    record=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    keys=['code','signal_date','entry_date'];results=[];inputs={str(run/'summary.json'):sha(run/'summary.json')}
    for period in ['2022_train','2023','2024','2025','2023-2025','2026']:
        paths=[run/f'{period}_{name}/trades.csv' for name in ['baseline','vote']]
        for path in paths:inputs[str(path)]=sha(path)
        b,v=[pd.read_csv(path,float_precision='round_trip') for path in paths]
        joined=b.merge(v,on=keys,how='outer',suffixes=('_baseline','_rl'),indicator=True,validate='one_to_one')
        both=joined[joined._merge.eq('both')].copy();only_b=joined[joined._merge.eq('left_only')];only_v=joined[joined._merge.eq('right_only')]
        both['net_pnl_difference']=both.net_pnl_rl-both.net_pnl_baseline
        joined.to_csv(out/f'{period}-all-trades.csv',index=False)
        both.sort_values('net_pnl_difference').to_csv(out/f'{period}-matched-trades.csv',index=False)
        baseline=next(r for r in record['results'] if r['period']==period and r['case']=='baseline')
        rl=next(r for r in record['results'] if r['period']==period and r['case']=='vote')
        delta=rl['final_equity']-baseline['final_equity']
        matched=float(both.net_pnl_difference.sum());base_only=float(-only_b.net_pnl_baseline.sum());rl_only=float(only_v.net_pnl_rl.sum())
        residual=delta-matched-base_only-rl_only
        if baseline['open_positions']==rl['open_positions']==0 and abs(residual)>1e-6:
            raise ValueError('Closed-trade decomposition differs from account result')
        results.append({'period':period,'matched_trades':len(both),'same_initial_shares':int((both.initial_shares_baseline==both.initial_shares_rl).sum()),
            'earlier_rl_exit':int((both.exit_date_rl<both.exit_date_baseline).sum()),'baseline_only_count':len(only_b),'rl_only_count':len(only_v),
            'matched_pnl_difference':matched,'absence_of_baseline_only_pnl':base_only,'rl_only_pnl':rl_only,
            'account_equity_difference':delta,'unallocated_residual':residual})
    for path,digest in inputs.items():
        if sha(Path(path))!=digest:raise ValueError('Frozen account inputs changed')
    output={'analysis':'Descriptive overlap of already-viewed frozen accounts; not independent causal validation',
        'join_keys':keys,'source_hashes':inputs,'results':results,
        'limitations':['Portfolio cash, position slots and policy states can change subsequent entries.',
                       'A baseline-only trade is not necessarily a direct skip action.',
                       'Matched entry PnL may also reflect quantity changes; report equal-quantity counts.',
                       'Annual reset and multi-year continuous results overlap and must not be summed.']}
    (out/'summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    pd.DataFrame(results).to_csv(out/'comparison.csv',index=False)
    print(pd.DataFrame(results).to_string(index=False))

if __name__=='__main__':main()
