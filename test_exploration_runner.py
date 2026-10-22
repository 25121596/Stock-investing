import unittest
import numpy as np
import pandas as pd

from run_exploration import matched_candidates, stable_random


class RandomOpportunityTests(unittest.TestCase):
    def test_hash_scores_survive_candidate_deletion_and_reordering(self):
        panel = pd.DataFrame({'date': ['2024-01-05'] * 3, 'code': ['A', 'B', 'C']})
        original = stable_random(panel, 12).set_index('code').score
        reduced = stable_random(panel.iloc[[2, 0]], 12).set_index('code').score
        self.assertEqual(original['A'], reduced['A'])
        self.assertEqual(original['C'], reduced['C'])

    def test_matching_uses_signal_counts_without_future_labels(self):
        pool = pd.DataFrame({'date': ['2024-01-05'] * 3 + ['2024-01-08'] * 3,
            'code': ['A', 'B', 'C'] * 2, 'net_return': [np.nan] * 6})
        policy = pd.DataFrame({'date': ['2024-01-05', '2024-01-05', '2024-01-08'], 'code': ['A', 'B', 'C'], 'score': [1] * 3})
        result = matched_candidates(policy, stable_random(pool, 12))
        self.assertEqual(result.groupby('date').size().tolist(), [2, 1])
        self.assertFalse(result.duplicated(['date', 'code']).any())


if __name__ == '__main__':
    unittest.main()
