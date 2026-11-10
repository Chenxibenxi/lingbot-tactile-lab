import unittest
import torch
from fixed_weight_tactile_control import payload,weight_arm
class ControlTest(unittest.TestCase):
 def test_same_weights_and_metadata_unchanged(self):
  x=dict(tactile_history=torch.ones(1,5,4,3),tactile_mask=torch.full((1,5,4),.5),tactile_age_ms=torch.tensor([17.]),tactile_valid=torch.tensor([True]))
  z=payload(x,'history_zero');self.assertEqual(weight_arm('live'),weight_arm('history_zero'))
  self.assertEqual(torch.count_nonzero(z['tactile_history']).item(),0)
  self.assertTrue(torch.equal(x['tactile_history'],torch.ones_like(x['tactile_history'])))
  for k in x:
   if k!='tactile_history':self.assertTrue(torch.equal(x[k],z[k]))
  x['tactile_valid'][0]=False
  self.assertFalse(payload(x,'history_zero')['tactile_valid'][0])
 def test_zero_contact_exactly_same(self):
  x=dict(tactile_history=torch.zeros(1,5,4,3),tactile_mask=torch.ones(1,5,4),tactile_age_ms=torch.tensor([0.]),tactile_valid=torch.tensor([True]))
  a,b=payload(x,'live'),payload(x,'history_zero')
  for k in x:self.assertTrue(torch.equal(a[k],b[k]))
  with self.assertRaises(ValueError):weight_arm('bypass')
if __name__=='__main__':unittest.main()
