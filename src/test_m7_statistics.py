import unittest
import numpy as np
from m7_paired_seed_statistics import summarize_paired_success
class StatsTest(unittest.TestCase):
 def test_contact_bit_ties_true_cannot_pass(self):
  x=np.zeros((16,3,4),int);x[:,:,1]=1;x[:,:,3]=1
  r=summarize_paired_success(x,list(range(100087,100103)))
  self.assertFalse(r['predeclared_stable_benefit_criterion_met'])
  self.assertEqual([c['comparison'] for c in r['contrasts']],['contact_state','zero'])
  self.assertEqual(r['contrasts'][0]['holm_adjusted_p'],1)
 def test_uniform_true_benefit_and_minimum_seed_count(self):
  x=np.zeros((16,3,4),int);x[:,:,1]=1
  self.assertTrue(summarize_paired_success(x,list(range(100087,100103)))['predeclared_stable_benefit_criterion_met'])
  self.assertFalse(summarize_paired_success(x[:11],list(range(100087,100098)))['predeclared_stable_benefit_criterion_met'])
 def test_m6_seeds_rejected(self):
  with self.assertRaises(ValueError):summarize_paired_success(np.zeros((16,3,4),int),list(range(100071,100087)))
if __name__=='__main__':unittest.main()
