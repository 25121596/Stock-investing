"""Frozen-policy action-bit ablation, with voting before any bit is disabled."""

MODES={'baseline':0,'filter_only':1,'exit_only':2,'full_rl':3}


class MaskedPolicy:
    def __init__(self,policy,mask):
        if mask not in MODES.values():raise ValueError('Invalid action mask')
        self.policy=policy;self.mask=mask;self.trace=[]

    def choose(self,state,allowed,epsilon=0.,training=False):
        if training or epsilon:raise ValueError('Ablation is frozen inference only')
        # Crucially, Q selection and majority voting see the ORIGINAL allowed set.
        raw=int(self.policy.choose(state,allowed))
        if raw not in allowed:raise ValueError('Underlying policy returned unavailable action')
        action=raw&self.mask
        if action not in allowed:raise ValueError('Masked action is unavailable')
        self.trace.append({'state':list(state),'allowed':list(allowed),'raw_action':raw,
                           'action':action,'mask':self.mask})
        return action


def compare_criteria(results,mode,period):
    rows={r['case']:r for r in results if r['family']=='primary' and r['period']==period}
    names=['baseline','stress_baseline',mode,'stress_'+mode]
    if not all(n in rows for n in names):return {'period':period,'passed':False,'missing_or_failed':True}
    b,sb,c,sc=[rows[n] for n in names]
    checks={'period':period,'higher_net_return':c['total_return']>b['total_return']+1e-12,
        'drawdown_not_worse':c['max_drawdown']>=b['max_drawdown']-1e-12,
        'higher_stressed_return':sc['total_return']>sb['total_return']+1e-12}
    checks['passed']=all(checks[k] for k in ['higher_net_return','drawdown_not_worse','higher_stressed_return'])
    return checks


def factorial_effects(rows):
    """End-wealth contrasts include all resulting portfolio path interactions."""
    if set(rows)!=set(MODES):raise ValueError('Incomplete four-arm comparison')
    b,f,e,a=[rows[n]['final_equity'] for n in MODES]
    return {'filter_without_exit_rmb':f-b,'exit_without_filter_rmb':e-b,
            'filter_with_exit_rmb':a-e,'exit_with_filter_rmb':a-f,'interaction_rmb':a-f-e+b}
