"""Fixed relevance and chronological fit boundaries for V0.10 research."""
import numpy as np
import pandas as pd

FIRST=['top_focus','tail_labels','tail_top_focus']
SECOND=['tail_xendcg','return_regressor']

def relevance(frame,tail=False):
    if tail:
        pct=frame.groupby('date').net_return.rank(method='average',pct=True)
        return np.searchsorted([.5,.8,.9,.95,.98],pct.to_numpy(),side='left').astype(int)
    return frame.groupby('date').net_return.transform(lambda s:pd.qcut(s,q=5,labels=False,duplicates='drop')).fillna(2).to_numpy(dtype=int)

def fit_rows(panel,features,year):
    boundary=f'{year}-01-01'
    mask=panel.date.lt(boundary)&panel.label_end.notna()&panel.label_end.lt(boundary)&panel.net_return.notna()&np.isfinite(panel[features].to_numpy(dtype=float)).all(axis=1)
    return panel[mask].sort_values(['date','code']).copy()

def parameters(name,rank_params,return_params):
    p=dict(return_params if name=='return_regressor' else rank_params)
    p['n_jobs']=4
    if name in ['top_focus','tail_top_focus']:p['lambdarank_truncation_level']=5
    if name in ['tail_labels','tail_top_focus']:p['label_gain']=[0,1,3,7,15,31]
    if name=='tail_xendcg':
        p['objective']='rank_xendcg';p.pop('label_gain',None);p.pop('lambdarank_truncation_level',None)
    return p

def passes(base,candidate,stress_base,stress_candidate):
    return {'higher_net_return':candidate['total_return']>base['total_return']+1e-12,
        'drawdown_not_worse':candidate['max_drawdown']>=base['max_drawdown']-1e-12,
        'higher_stressed_return':stress_candidate['total_return']>stress_base['total_return']+1e-12}
