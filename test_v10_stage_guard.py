import unittest
from run_v10_rank_objectives_v2 import allow_second_stage,PERIODS,FIRST

class StageGuardTests(unittest.TestCase):
    def rows(self):
        return [dict(period=period,case=case,quality_warning_count=0,audit={'passed':True})
            for period,_,_ in PERIODS for name in ['baseline',*FIRST] for case in [name,'stress_'+name]]
    def test_only_completed_valid_negative_stage_advances(self):
        verdicts={n:{'passed':False} for n in FIRST}
        self.assertTrue(allow_second_stage(verdicts,self.rows(),[]))
        verdicts[FIRST[0]]['passed']=True
        self.assertFalse(allow_second_stage(verdicts,self.rows(),[]))
    def test_missing_case_blocks_even_if_no_candidate_passes(self):
        with self.assertRaisesRegex(ValueError,'incomplete'):allow_second_stage({n:{'passed':False} for n in FIRST},self.rows()[:-1],[])
    def test_audit_and_quality_failure_block(self):
        rows=self.rows();rows[0]['audit']['passed']=False
        with self.assertRaisesRegex(ValueError,'quality/audit'):allow_second_stage({n:{'passed':False} for n in FIRST},rows,[])
        rows=self.rows();rows[0]['quality_warning_count']=1
        with self.assertRaises(ValueError):allow_second_stage({n:{'passed':False} for n in FIRST},rows,[])
    def test_incomplete_verdicts_cannot_trigger_more_trials(self):
        with self.assertRaisesRegex(ValueError,'Invalid'):allow_second_stage({},self.rows(),[])

if __name__=='__main__':unittest.main()
