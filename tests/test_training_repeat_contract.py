import unittest,torch
from tactile_adapter import TactileStateResidual
from training_repeat_contract import reinitialize_adapter,balanced_order
class RepeatContract(unittest.TestCase):
 def test_repeated_seed_identity_distinct_seed_variation_and_preserved_buffers(self):
  a=TactileStateResidual(16);a.train_median.fill_(3);a.train_scale.fill_(2);before={k:v.clone() for k,v in a.named_buffers()};rng=torch.random.get_rng_state().clone()
  x=reinitialize_adapter(a,101);state={k:v.clone() for k,v in a.named_parameters()};y=reinitialize_adapter(a,101)
  self.assertEqual(x['parameter_sha256'],y['parameter_sha256']);self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
  self.assertTrue(all(torch.equal(v,before[k]) for k,v in a.named_buffers()))
  z=reinitialize_adapter(a,102);self.assertNotEqual(x['parameter_sha256'],z['parameter_sha256']);self.assertEqual(torch.count_nonzero(a.to_state.weight),0)
  self.assertTrue(any(not torch.equal(v,state[k]) for k,v in a.named_parameters()))
 def test_shared_order_deterministic_and_rejects_heldout(self):
  train=[{'meta':{'fold':'train','seed':s,'contact':c}} for s in [1,2] for c in [False,True]]
  self.assertEqual(balanced_order(train,50,11),balanced_order(train,50,11));self.assertNotEqual(balanced_order(train,50,11),balanced_order(train,50,12))
  train[0]['meta']['fold']='validation'
  with self.assertRaises(ValueError):balanced_order(train,50,11)
 def test_missing_contact_stratum_rejected(self):
  with self.assertRaises(ValueError):balanced_order([{'meta':{'fold':'train','seed':1,'contact':True}}],5,1)
