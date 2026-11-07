import unittest
from controlled_feedback_schedule import decision_positions,matched_noise_seed,execution_count


class FeedbackSchedule(unittest.TestCase):
    def test_shared_action_positions_share_noise(self):
        fast={i:matched_noise_seed(100123,i) for i in decision_positions(1)}
        slow={i:matched_noise_seed(100123,i) for i in decision_positions(10)}
        self.assertEqual(len(fast),400);self.assertEqual(len(slow),40)
        self.assertTrue(all(fast[i]==noise for i,noise in slow.items()))
        self.assertNotEqual(fast[1],slow[10])

    def test_no_cross_seed_collision_and_equal_action_budget(self):
        one={matched_noise_seed(100123,i) for i in decision_positions(1)}
        two={matched_noise_seed(100124,i) for i in decision_positions(1)}
        self.assertTrue(one.isdisjoint(two))
        for cadence in (1,10):
            self.assertEqual(sum(execution_count(cadence,i) for i in decision_positions(cadence)),400)

    def test_non_action_indices_rejected(self):
        for value in (True,-1,400,1.5):
            with self.assertRaises(ValueError):matched_noise_seed(100123,value)
        with self.assertRaises(ValueError):execution_count(10,1)
        with self.assertRaises(ValueError):decision_positions(5)
