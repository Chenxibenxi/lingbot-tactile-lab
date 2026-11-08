import unittest
from audit_feedback_candidate_seeds import seed_mentions

class SeedSearch(unittest.TestCase):
    def test_nested_lists_and_trial_strings(self):
        hits=seed_mentions({'train_seeds':[100129,100144],'records':[{'trial_id':'relay_s100130_trial0'}]})
        self.assertEqual({h['seed'] for h in hits},{100129,100130,100144})
    def test_boundaries_and_non_seed_numbers(self):
        self.assertEqual(seed_mentions({'seed':1001290,'loss':100129,'seeds':[100128,100145]}),[])
