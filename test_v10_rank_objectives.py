import unittest
import numpy as np
import pandas as pd
from v10_rank_objectives import relevance,fit_rows,parameters,passes

class RankObjectiveTests(unittest.TestCase):
    def test_tail_resolution_and_same_return_ties(self):
        f=pd.DataFrame({'date':['2020-01-02']*100,'net_return':np.arange(100.)})
        old=relevance(f);new=relevance(f,True)
        self.assertEqual(len(set(old[-20:])),1)
        self.assertGreater(len(set(new[-20:])),1)
        f['net_return']=1.
        self.assertEqual(len(set(relevance(f,True))),1)

    def test_labels_invariant_to_code_order(self):
        f=pd.DataFrame({'date':['a']*50+['b']*50,'net_return':list(range(50))*2})
        f['net_return']=f.net_return//5
        expected=pd.Series(relevance(f,True),index=f.index)
        shuffled=f.sample(frac=1,random_state=3)
        actual=pd.Series(relevance(shuffled,True),index=shuffled.index)
        pd.testing.assert_series_equal(expected,actual.sort_index())

    def test_labels_crossing_year_boundary_are_purged(self):
        f=pd.DataFrame({'date':['2025-12-20']*3+['2026-01-02'],'code':['a','b','c','d'],
            'label_end':['2025-12-31','2026-01-01','2025-12-31','2026-01-03'],'net_return':[1.]*4,'x':[1.,1.,np.nan,1.]})
        self.assertEqual(fit_rows(f,['x'],2026).code.tolist(),['a'])

    def test_parameters_do_not_mutate_frozen_config(self):
        rank={'objective':'lambdarank','label_gain':[0,1,3,7,15]};ret={'objective':'huber'}
        p=parameters('tail_top_focus',rank,ret)
        self.assertEqual(p['lambdarank_truncation_level'],5)
        self.assertEqual(len(p['label_gain']),6)
        self.assertEqual(len(rank['label_gain']),5)
        x=parameters('tail_xendcg',rank,ret)
        self.assertNotIn('label_gain',x)
        self.assertEqual(x['objective'],'rank_xendcg')

    def test_return_improvement_cannot_hide_worse_drawdown(self):
        a={'total_return':.1,'max_drawdown':-.05};b={'total_return':.2,'max_drawdown':-.08}
        checks=passes(a,b,a,b)
        self.assertTrue(checks['higher_net_return']);self.assertFalse(checks['drawdown_not_worse'])

if __name__=='__main__':unittest.main()
