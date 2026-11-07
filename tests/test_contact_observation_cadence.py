import unittest
from audit_contact_observation_cadence import coverage
class CoverageClock(unittest.TestCase):
 def test_causal_boundary_and_tail_separation(self):
  c=[True]*12;r=coverage(c,[0,5,10],[False,True,True],window_steps=2)
  self.assertEqual(r['contact_steps_before_last_request'],10)
  self.assertEqual(r['contact_steps_in_query_windows'],4)
  self.assertEqual(r['tail_contact_steps_after_last_request'],2)
  self.assertEqual(r['contact_runs_seen_by_any_window'],1)
 def test_invalid_history_never_counts_and_gaps_split_runs(self):
  r=coverage([True,False,True,False,True,False],[1,5],[False,True],window_steps=2)
  self.assertEqual(r['contact_runs_before_last_request'],3)
  self.assertEqual(r['contact_runs_seen_by_any_window'],1)
  self.assertEqual(r['contact_steps_in_query_windows'],1)
 def test_future_anchor_rejected(self):
  with self.assertRaises(ValueError):coverage([True],[2],[True],window_steps=1)
