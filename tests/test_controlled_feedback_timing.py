import unittest
from controlled_feedback_timing import request_timing, execution_timing


class TimingTest(unittest.TestCase):
    def request(self,cadence=10,**changes):
        args=dict(seed=100200,executed_actions=10,cadence=cadence,capture_step=100,
                  history_anchor=100,window_steps=25,dt_s=.004)
        args.update(changes);return request_timing(**args)

    def test_shared_action_positions_share_noise(self):
        self.assertEqual(self.request(1)['noise_seed'],self.request(10)['noise_seed'])
        self.assertNotEqual(self.request(1)['planned_action_count'],self.request(10)['planned_action_count'])

    def test_wall_latency_not_added_to_simulation_age(self):
        result=execution_timing(self.request(),physics_before=100,physics_after=200,inference_seconds=1.5)
        self.assertEqual(result['history_age_at_action_start_s'],0)
        self.assertAlmostEqual(result['history_age_at_action_end_s'],.4)
        self.assertEqual(result['inference_physics_advance'],0)

    def test_future_history_and_invalid_clock_rejected(self):
        with self.assertRaises(ValueError):self.request(history_anchor=101)
        with self.assertRaises(ValueError):self.request(dt_s=float('nan'))
        with self.assertRaises(ValueError):execution_timing(self.request(),physics_before=99,physics_after=100,inference_seconds=1)
