"""Finite-year tabular Q-learning overlay with prior-close decisions only."""
from collections import Counter
import math
import numpy as np
from v08_engine_verified import VerifiedV08Replay

SEEDS=(2026102101,2026102102,2026102103)
EPOCHS=200

class QPolicy:
    def __init__(self,seed,alpha=.1,gamma=.95):
        self.seed=seed;self.alpha=alpha;self.gamma=gamma
        self.values={};self.visits={};self.rng=np.random.default_rng(seed);self.update_count=0
    def choose(self,state,allowed,epsilon=0.,training=False):
        if training and self.rng.random()<epsilon:return int(self.rng.choice(allowed))
        values=self.values.get(state,np.zeros(4));counts=self.visits.get(state,np.zeros(4,dtype=int))
        eligible=allowed if training else [a for a in allowed if counts[a]>=10]
        if not eligible:return 0
        return max(eligible,key=lambda a:(float(values[a]),-a))
    def update(self,state,action,reward,next_state,allowed,terminal=False):
        q=self.values.setdefault(state,np.zeros(4));n=self.visits.setdefault(state,np.zeros(4,dtype=int))
        future=0. if terminal else max(self.values.get(next_state,np.zeros(4))[a] for a in allowed)
        q[action]+=self.alpha*(reward+self.gamma*future-q[action]);n[action]+=1;self.update_count+=1
    def dump(self):
        return {'seed':self.seed,'alpha':self.alpha,'gamma':self.gamma,'update_count':self.update_count,
            'states':[{'state':list(s),'q':list(map(float,q)),'visits':list(map(int,self.visits[s]))} for s,q in sorted(self.values.items())]}
    @classmethod
    def restore(cls,record):
        obj=cls(record['seed'],record['alpha'],record['gamma']);obj.update_count=record['update_count']
        for row in record['states']:
            s=tuple(row['state']);obj.values[s]=np.array(row['q']);obj.visits[s]=np.array(row['visits'],dtype=int)
        return obj

class VotePolicy:
    def __init__(self,policies):self.policies=policies
    def choose(self,state,allowed,**kwargs):
        votes=Counter(p.choose(state,allowed) for p in self.policies)
        return max(sorted(votes),key=lambda a:(votes[a],-a))

class FollowPolicy:
    def choose(self,*args,**kwargs):return 0

def available_actions(pending,positions):
    return [a for a in range(4) if (not a&1 or pending) and (not a&2 or positions)]

def reward(nav0,dd0,nav1,dd1):
    if min(nav0,nav1)<=0:raise ValueError('Nonpositive account NAV')
    return 10000*(math.log(nav1/nav0)-.25*max(0.,dd1-dd0))

class RLReplay(VerifiedV08Replay):
    def __init__(self,*args,policy,context,training=False,epsilon=0.,**kwargs):
        super().__init__(*args,**kwargs)
        self.policy=policy;self.context=context;self.training=training;self.epsilon=epsilon
        self.rl_peak=self.config.capital;self.previous=None;self.decisions=[];self.episode_reward=0.
        self.index_by_day={d:i for i,d in enumerate(self.calendar)}
    def observe_close(self):
        if not self.equity:raise ValueError('No completed close observation')
        day=self.equity[-1]['date'];nav=self.equity[-1]['equity']
        self.rl_peak=max(self.rl_peak,nav);dd=1-nav/self.rl_peak
        c=self.context[day];positions=list(self.positions.values())
        invested=sum(p.shares*self.marks[p.code] for p in positions)
        entry=sum(p.entry_notional for p in positions)
        gain=invested/entry-1 if entry else 0.
        age=max([self.index_by_day[day]-p.entry_index for p in positions],default=0)
        state=(int(np.searchsorted([.45,.58],c['breadth'],side='right')),
               int(np.searchsorted([-.02,.02],c['market_ret5'],side='right')),
               int(np.searchsorted([.05,.35,.65],invested/nav,side='right')),
               int(np.searchsorted([.02,.05],dd,side='right')),
               int(np.searchsorted([-.03,0.,.03],gain,side='right')),
               int(np.searchsorted([3,8],age,side='right')),min(2,len(self.pending_buys)))
        return day,state,available_actions(bool(self.pending_buys),bool(positions)),nav,dd
    def corporate_actions(self,day):
        # Snapshot and choose BEFORE today's corporate action changes holdings.
        # Every observation comes from self.equity[-1] and yesterday's marks.
        if self.equity:
            signal_day,state,allowed,nav,dd=self.observe_close()
            if signal_day>=day:raise ValueError('Decision must precede execution day')
            if self.previous is not None:
                prev_state,prev_action,prev_nav,prev_dd=self.previous
                r=reward(prev_nav,prev_dd,nav,dd);self.episode_reward+=r
                if self.training:self.policy.update(prev_state,prev_action,r,state,allowed)
            action=self.policy.choose(state,allowed,epsilon=self.epsilon,training=self.training)
            if action not in allowed:raise ValueError('Policy selected unavailable action')
            self.previous=(state,action,nav,dd)
            self.decisions.append({'signal_date':signal_day,'execution_date':day,'state':list(state),'action':action,
                'pending_before':len(self.pending_buys),'positions_before':len(self.positions)})
            if action&1:self.pending_buys=[]
            if action&2:
                for position in self.positions.values():position.stop_pending=True
        # Mandatory inherited quality diagnostics and actual corporate actions.
        return super().corporate_actions(day)
    def run(self):
        super().run()
        if self.previous is not None:
            _,state,allowed,nav,dd=self.observe_close();s,a,prev_nav,prev_dd=self.previous
            r=reward(prev_nav,prev_dd,nav,dd);self.episode_reward+=r
            if self.training:self.policy.update(s,a,r,state,allowed,terminal=True)
        return self
