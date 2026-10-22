"""Batch-level supervised gate using completed, executable baseline outcomes."""
from collections import Counter
import math
import warnings
import numpy as np

STATE_NAMES=['breadth_bin','market_ret5_bin','exposure_bin','drawdown_bin','floating_gain_bin','holding_age_bin','pending_count_bin']
THRESHOLD=.5


class ConstantGate:
    def decide(self,state,allowed):return 0,None


class BreadthGate:
    def decide(self,state,allowed):return (int(state[0]<2) if 1 in allowed else 0),None


class FrozenLogit:
    def __init__(self,record):self.record=record
    @classmethod
    def fit(cls,rows):
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.exceptions import ConvergenceWarning
        if len(rows)<10 or min(Counter(r['y'] for r in rows).get(k,0) for k in [0,1])<2:
            raise ValueError('Insufficient matured batches/classes for preregistered fit')
        x=np.array([r['state'] for r in rows],dtype=float);y=np.array([r['y'] for r in rows])
        scaler=StandardScaler().fit(x)
        model=LogisticRegression(C=1.,class_weight=None,solver='lbfgs',max_iter=2000)
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning);model.fit(scaler.transform(x),y)
        record={'kind':'standardized_logistic_regression','features':STATE_NAMES,'mean':scaler.mean_.tolist(),
            'scale':scaler.scale_.tolist(),'coef':model.coef_[0].tolist(),'intercept':float(model.intercept_[0]),
            'threshold':THRESHOLD,'C':1.,'class_weight':None,'solver':'lbfgs','max_iter':2000,
            'training_rows':len(rows),'class_counts':{str(k):int(sum(y==k)) for k in [0,1]},
            'training_states':[list(s) for s in sorted({tuple(r['state']) for r in rows})],
            'last_label_end':max(r['label_end'] for r in rows),'last_signal':max(r['signal_date'] for r in rows)}
        obj=cls(record)
        error=max(abs(obj.score(row['state'])-float(p)) for row,p in zip(rows,model.predict_proba(scaler.transform(x))[:,1]))
        if error>1e-12:raise ValueError('Saved logistic inference differs from sklearn')
        obj.record['max_reload_score_error']=error
        return obj
    def score(self,state):
        r=self.record;x=(np.asarray(state,dtype=float)-np.asarray(r['mean']))/np.asarray(r['scale'])
        z=float(x@np.asarray(r['coef'])+r['intercept'])
        return 1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
    def decide(self,state,allowed):
        p=self.score(state)
        return (int(p<self.record['threshold']) if 1 in allowed else 0),p


class LegacyRLGate:
    def __init__(self,policy):self.policy=policy
    def decide(self,state,allowed):return int(self.policy.choose(state,allowed))&1,None


class GateAdapter:
    """RLReplay passes prior-close state; orders are copied before any mutation."""
    def __init__(self,gate):self.gate=gate;self.replay=None;self.observations=[]
    def choose(self,state,allowed,epsilon=0.,training=False):
        if epsilon or training:raise ValueError('This comparison is frozen gate inference only')
        rep=self.replay
        if rep is None:raise ValueError('Unbound gate')
        day=rep.equity[-1]['date'];index=rep.index_by_day[day];execution=rep.calendar[index+1]
        orders=[dict(order) for order in rep.pending_buys]
        action,score=self.gate.decide(state,allowed)
        if action not in [0,1] or action not in allowed:raise ValueError('Invalid gate action')
        self.observations.append({'signal_date':day,'execution_date':execution,'state':list(state),
            'allowed':list(allowed),'action':action,'score':score,'orders':orders,
            'pending_before':len(orders),'positions_before':len(rep.positions)})
        return action


def matured_batch_labels(rep,observations,boundary='2023-01-01'):
    """Targets are observed baseline batches, not true counterfactual utilities."""
    closed={t['trade_id']:t for t in rep.trades};records=[]
    for obs in observations:
        if not obs['orders']:continue
        day=obs['execution_date'];codes={o['code'] for o in obs['orders']}
        fills=[f for f in rep.fills if f['date']==day and f['side']=='buy']
        if any(f['code'] not in codes for f in fills):raise ValueError('Fill outside captured pending batch')
        ids=[f['trade_id'] for f in fills]
        row={**obs,'filled_trade_ids':ids,'filled_count':len(ids),'status':'eligible','label_end':None,'net_pnl':None,'net_return':None,'y':None}
        if not ids:row['status']='no_execution'
        elif any(i not in closed for i in ids):row['status']='right_censored'
        else:
            trades=[closed[i] for i in ids]
            if any(t['signal_date']!=obs['signal_date'] or t['entry_date']!=day for t in trades):raise ValueError('Batch trade key mismatch')
            end=max(t['exit_date'] for t in trades)
            row['label_end']=end
            if end>=boundary:row['status']='label_not_mature_before_fit'
            else:
                pnl=sum(t['net_pnl'] for t in trades);cost=sum(t['entry_notional']+t['entry_fees'] for t in trades)
                row.update(net_pnl=float(pnl),net_return=float(pnl/cost),y=int(pnl>0))
        records.append(row)
    return records
