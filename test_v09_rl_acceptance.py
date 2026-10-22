import unittest
from run_v09_rl_2022 import acceptance_check

class AcceptanceTests(unittest.TestCase):
    def test_normal_and_stress_policies_do_not_get_swapped(self):
        rows={'baseline':{'total_return':.41,'max_drawdown':-.078},
              'stress_baseline':{'total_return':.21,'max_drawdown':-.076},
              'vote':{'total_return':.47,'max_drawdown':-.044},
              'stress_vote':{'total_return':.40,'max_drawdown':-.047}}
        self.assertTrue(acceptance_check(rows,'test')['passed'])
        rows['vote']['total_return']=.20
        checks=acceptance_check(rows,'test')
        self.assertFalse(checks['passed']);self.assertFalse(checks['higher_net_return'])
        self.assertTrue(checks['drawdown_not_worse']);self.assertTrue(checks['higher_stressed_return'])

    def test_missing_cases_fail_closed(self):
        self.assertFalse(acceptance_check({},'test')['passed'])

if __name__=='__main__':unittest.main()
