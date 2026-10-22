import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import arena
from engine import Config


class ArenaTests(unittest.TestCase):
    def fixture(self,missing=False,adjustment=False,cash_rate=0.):
        days=['2025-01-02','2025-01-03','2025-01-06']
        code='sh.600006'
        prices=[10.,10.,5. if adjustment else 10.]
        frame=pd.DataFrame([dict(date=d,code=code,open=p,high=p,low=p,close=p,preclose=p,
            volume=100000.,tradestatus=1,isST=0) for d,p in zip(days,prices)])
        if missing:frame=frame.iloc[:-1]
        frames={code:frame};bars={code:frame.set_index('date').to_dict('index')}
        signals=pd.DataFrame([dict(date=days[0],code=code,score=1.)])
        rep=arena.simulate(frames,bars,days,signals,[],Config(),{}, {},{},cash_rate)
        return rep,bars

    def test_missing_quote_blocked(self):
        with self.assertRaisesRegex(ValueError,'missing_held_stock_quote') as caught:self.fixture(missing=True)
        self.assertTrue(caught.exception.replay.equity)
        self.assertTrue(caught.exception.replay.review_warnings)

    def test_adjustment_blocked(self):
        with self.assertRaisesRegex(ValueError,'unexplained_adjustment'):self.fixture(adjustment=True)

    def test_audit_detects_cash_tampering(self):
        rep,bars=self.fixture()
        self.assertTrue(arena.audit_ledger(rep,bars,[])['passed'])
        rep.equity[-1]['cash']+=1
        with self.assertRaisesRegex(ValueError,'ledger mismatch'):arena.audit_ledger(rep,bars,[])

    def test_cash_accrual_including_first_day(self):
        days=['2025-01-02','2025-01-03','2025-01-06']
        empty=pd.DataFrame(columns=['date','code','score'])
        rep=arena.simulate({}, {}, days,empty,[],Config(),{},{},{},.022)
        self.assertAlmostEqual(rep.equity[0]['equity'],30000*1.022**(1/242))
        self.assertAlmostEqual(rep.equity[-1]['equity'],30000*1.022**(3/242))
        self.assertTrue(arena.audit_ledger(rep,{},[])['passed'])
        self.assertIsNone(rep.summary()['win_rate'])

    def test_universe_mismatch_and_duplicates_blocked(self):
        base=pd.DataFrame([dict(date='2025-01-02',code='a')])
        a=base.assign(pred_net_return_v2=.01);b=base.assign(pred_net_return_v3=.01);c=base.assign(pred_score_v08=1.)
        selections,_=arena.build_signals(a,b,c,{'2025-01-02':2})
        self.assertEqual(len(selections['v08']),1)
        with self.assertRaisesRegex(ValueError,'universes differ'):arena.build_signals(a,b,c.assign(code='b'),{})
        with self.assertRaisesRegex(ValueError,'duplicate'):arena.build_signals(pd.concat([a,a]),b,c,{})

    def test_random_counts_and_determinism(self):
        pool=pd.DataFrame([dict(date=d,code=str(c)) for d in ['a','b'] for c in range(10)])
        signals=pool.groupby('date').head(3).assign(score=1.)
        a=arena.matched_candidates(signals,arena.stable_random(pool,arena.SEEDS[0]))
        b=arena.matched_candidates(signals,arena.stable_random(pool.iloc[::-1],arena.SEEDS[0]))
        pd.testing.assert_frame_equal(a,b)
        self.assertEqual(a.groupby('date').size().tolist(),[3,3])

    def test_output_refuses_overwrite(self):
        with TemporaryDirectory() as folder:
            with self.assertRaises(FileExistsError):arena.run_tournament([],Path(folder))

    def test_dependency_scope_includes_engine_not_unrelated_bridge(self):
        names={p.name for p in arena.source_dependencies()}
        self.assertIn('engine.py',names)
        self.assertIn('v08_engine_verified.py',names)
        self.assertNotIn('trader_bridge.py',names)

    def test_annualization_and_risk_free_are_shared(self):
        rep,_=self.fixture(cash_rate=.022)
        summary=rep.summary()
        self.assertAlmostEqual(summary['cagr'],(summary['final_equity']/30000)**(365.25/4)-1)
        self.assertEqual(summary['sharpe_risk_free_annual_assumption'],0.)

    def test_custom_dates_not_ignored(self):
        with patch('sys.argv',['arena.py','run','--start','2025-03-01','--end','2025-04-01']),patch.object(arena,'run_tournament') as run:
            arena.main()
        self.assertEqual(run.call_args.args[0],[('custom','2025-03-01','2025-04-01')])

    def test_conflicting_dates_rejected(self):
        with patch('sys.argv',['arena.py','--mode','all','--start','2025-01-01','--end','2025-02-01']):
            with self.assertRaises(SystemExit):arena.main()

if __name__=='__main__':unittest.main()
