import unittest,torch
from confirmation_tactile_controls import runtime_payload,WEIGHT_ARM
class RuntimeControl(unittest.TestCase):
 def payload(self,value,valid=True):return {'tactile_history':torch.full((1,5,4,3),float(value)),'tactile_mask':torch.ones(1,5,4),'tactile_age_ms':torch.tensor([0.]),'tactile_valid':torch.tensor([valid])}
 def pools(self):return {seed:{c:[{'meta':{'fold':'train','seed':seed,'frame_index':1,'contact':c},'tactile':self.payload(1 if c else 0)}] for c in [False,True]} for seed in [100045,100058]}
 def test_invalid_first_not_replaced_with_valid_donor(self):
  cur=self.payload(0,False);out,donor=runtime_payload(cur,'mismatch',0,self.pools());self.assertIsNone(donor);self.assertFalse(out['tactile_valid'][0]);self.assertTrue(torch.equal(out['tactile_history'],cur['tactile_history']))
 def test_same_weight_zero_preserves_validity_and_mask(self):
  cur=self.payload(4);out,_=runtime_payload(cur,'true_input_zero',0,{});self.assertEqual(WEIGHT_ARM['true_input_zero'],'true');self.assertEqual(out['tactile_history'].count_nonzero(),0);self.assertTrue(torch.equal(out['tactile_mask'],cur['tactile_mask']));self.assertEqual(cur['tactile_history'].max(),4)
 def test_mapping_deterministic_and_heldout_donor_rejected(self):
  pools=self.pools();cur=self.payload(1);_,meta=runtime_payload(cur,'mismatch',3,pools);self.assertEqual(meta['seed'],100058)
  pools[100058][True][0]['meta']['fold']='validation'
  with self.assertRaises(ValueError):runtime_payload(cur,'mismatch',3,pools)
