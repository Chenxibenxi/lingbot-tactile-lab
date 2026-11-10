import unittest
import torch
from fixed_weight_negative_controls import transform
class NegativeTest(unittest.TestCase):
 def fixture(self):
  x=dict(tactile_history=torch.ones(1,5,4,3),tactile_mask=torch.ones(1,5,4),tactile_age_ms=torch.tensor([7.]),tactile_valid=torch.tensor([True]))
  d={k:v.clone() for k,v in x.items()};d['tactile_history'][...,:2]*=2;d['tactile_age_ms'][0]=30
  return x,{1:{True:[dict(meta=dict(seed=1,fold='train',contact=True,frame_index=0),tactile=d)],False:[]}}
 def test_mismatch_changes_only_history(self):
  x,p=self.fixture();y,m=transform(x,'history_mismatch',0,p)
  for k in ['tactile_mask','tactile_age_ms','tactile_valid']:self.assertTrue(torch.equal(x[k],y[k]))
  self.assertEqual(y['tactile_history'][0,0,0,0],2);self.assertEqual(x['tactile_history'][0,0,0,0],1)
  p[1][True][0]['meta']['fold']='validation'
  with self.assertRaises(ValueError):transform(x,'history_mismatch',0,p)
 def test_coverage_and_invalid_not_bypassed(self):
  x,p=self.fixture();p[1][True][0]['tactile']['tactile_mask'].zero_()
  with self.assertRaises(ValueError):transform(x,'history_mismatch',0,p)
  x['tactile_valid'][0]=False;x['tactile_mask'].zero_();x['tactile_history'].zero_();x['tactile_age_ms'][0]=-1
  y,m=transform(x,'history_mismatch',0,{})
  self.assertIsNone(m);self.assertFalse(y['tactile_valid'][0])
 def test_contact_bit_removes_magnitude(self):
  x,p=self.fixture();y,_=transform(x,'contact_bit',0,p)
  self.assertEqual(torch.count_nonzero(y['tactile_history'][...,:2]),0)
  self.assertTrue(torch.all(y['tactile_history'][...,2]==1))
if __name__=='__main__':unittest.main()
