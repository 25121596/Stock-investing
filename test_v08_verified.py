import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import pandas as pd
import engine
import corporate_events
from engine import Config
from v08_engine_verified import VerifiedV08Replay


class VerifiedV08Tests(unittest.TestCase):
    def run_case(self, prices, refs=None, missing=False, days=None):
        days = days or ['2025-01-02','2025-01-03','2025-01-06']
        code='sh.600006'
        frame=pd.DataFrame([{'date':d,'code':code,'open':p,'high':p,'low':p,'close':p,
            'preclose':r,'volume':100000.,'tradestatus':1,'isST':0}
            for d,p,r in zip(days,prices,refs or prices)])
        if missing: frame=frame.iloc[:-1]
        sig=pd.DataFrame([{'date':days[0],'code':code,'score':1.}])
        return VerifiedV08Replay({code:frame},days,sig,[],Config(hold_days=10),enable_cash_overlay=False).run()

    def test_unexplained_adjustment_is_blocking(self):
        with self.assertRaisesRegex(ValueError,'unexplained_adjustment'):
            self.run_case([10.,10.,5.])

    def test_missing_held_quote_is_blocking(self):
        with self.assertRaisesRegex(ValueError,'missing_held_stock_quote'):
            self.run_case([10.,10.,10.],missing=True)

    def test_annualization_uses_real_elapsed_period(self):
        rep=self.run_case([10.,10.,11.],refs=[10.,10.,10.],days=['2025-01-02','2025-01-03','2025-12-31'])
        summary=rep.summary()
        expected=(summary['final_equity']/30000)**(365.25/363)-1
        self.assertAlmostEqual(summary['cagr'],expected)
        self.assertEqual(summary['quality_warning_count'],0)

    def test_paper_adapter_seals_actual_eight_day_exit_and_profit_state(self):
        import v08_paper_forward_v2 as paper
        days=pd.bdate_range('2025-03-03',periods=10).strftime('%Y-%m-%d').tolist()
        code='sh.600006'
        frame=pd.DataFrame([dict(date=d,code=code,open=10.,high=10.,low=10.,close=10.,
            preclose=10.,volume=100000.,tradestatus=1,isST=0) for d in days])
        frames={code:frame}; bars={code:frame.set_index('date').to_dict('index')}
        paper._SIM_PANEL=pd.DataFrame([dict(date=d,code=f'synthetic_{i}',adj_ret5=.01 if i<50 else -.01,
            ret_vol20=.02) for d in days for i in range(101)])
        signal=pd.DataFrame([dict(date=days[0],code=code,score=1.)])
        rep=paper.simulate(frames,bars,days[:9],signal,[],10,[engine])
        self.assertEqual(len(rep.trades),0)
        self.assertTrue(paper.v08_sell_due(rep,rep.positions[code],days[:9]))
        state=paper.account_state(rep)
        self.assertIn('peak_prices',state)
        self.assertIn('profit_lock_flags',state)
        settled=paper.simulate(frames,bars,days,signal,[],10,[engine])
        self.assertEqual(settled.trades[0]['exit_date'],days[9])
        self.assertEqual(settled.trades[0]['exit_reason'],'fixed_horizon')

    def test_complete_empty_signal_journal_can_seal_in_temporary_fixture(self):
        import v08_paper_forward_v2 as paper
        class FixtureClock(datetime):
            @classmethod
            def now(cls,tz=None):
                value=datetime(2026,10,20,16,0,tzinfo=paper.CST)
                return value.astimezone(tz) if tz else value
        with TemporaryDirectory() as temp:
            session=Path(temp)/'synthetic-session'
            dataset=Path(temp)/'synthetic-input';dataset.mkdir()
            paper.write_json(dataset/'manifest.json',{'end':'2026-10-20'})
            frame=pd.DataFrame([dict(date='2026-10-20',code='sh.600006',open=10.,close=10.,
                high=10.,low=10.,preclose=10.,volume=100000.,tradestatus=1,isST=0)])
            panel=pd.DataFrame([dict(date='2026-10-20',code='sh.600006',adj_ret5=.01,ret_vol20=.02)])
            def fake_infer(*args):
                paper._SIM_PANEL=panel
                return panel,pd.DataFrame(),{name:pd.DataFrame(columns=['date','code','score']) for name in paper.accounts()}
            loaded=({'end':'2026-10-20'},{'sh.600006':frame},['2026-10-20'],[],['2026-10-20','2026-10-21'])
            with patch.object(paper,'datetime',FixtureClock),patch.object(paper,'load_dataset',return_value=loaded), \
                 patch.object(paper,'load_runtime',return_value=(engine,None,corporate_events,None,None)), \
                 patch.object(paper,'infer',side_effect=fake_infer):
                paper.initialize(session)
                paper.record_day(session,dataset)
            self.assertTrue((session/'days/2026-10-20/record.json').is_file())
            self.assertEqual(len(paper.verify_records(session)[1]),1)


if __name__=='__main__': unittest.main()
