import unittest,numpy as np
from paired_seed_statistics import exact_sign_flip_p,holm,summarize_paired_success
class PairedStatistics(unittest.TestCase):
 def test_exact_sign_symmetry_ties_and_holm(self):
  self.assertEqual(exact_sign_flip_p(np.ones(4,dtype=int)),.125)
  self.assertEqual(exact_sign_flip_p(np.array([1,-1,0,0])),1.)
  self.assertEqual(exact_sign_flip_p(np.zeros(16,dtype=int)),1.)
  np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])
 def test_no_gain_does_not_pass_and_replications_not_pseudoreplicated(self):
  s=list(range(100071,100087));x=np.zeros((16,3,4),dtype=int);x[0,:,1]=1
  r=summarize_paired_success(x,s);self.assertFalse(r['predeclared_stable_benefit_criterion_met']);self.assertEqual(r['eligible_seed_count'],16);self.assertEqual(r['contrasts'][0]['exact_two_sided_p'],1.)
 def test_large_consistent_gain_passes_but_insufficient_seeds_refused(self):
  s=list(range(100071,100087));x=np.zeros((16,3,4),dtype=int);x[:,:,1]=1
  self.assertTrue(summarize_paired_success(x,s)['predeclared_stable_benefit_criterion_met'])
  self.assertFalse(summarize_paired_success(x[:4],s[:4])['predeclared_stable_benefit_criterion_met'])
 def test_historical_duplicate_and_fractional_inputs_refused(self):
  for x,s in [(np.zeros((1,3,4),int),[100051]),(np.zeros((2,3,4),int),[100071,100071]),(np.zeros((1,3,4),float),[100071])]:
   with self.assertRaises(ValueError):summarize_paired_success(x,s)
