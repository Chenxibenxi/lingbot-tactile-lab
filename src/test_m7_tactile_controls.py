import unittest
import torch
from contact_state_control import contact_state_payload
from m7_tactile_controls import runtime_payload
class RuntimeTest(unittest.TestCase):
 def sample(self):return dict(tactile_history=torch.zeros(1,5,4,3),tactile_mask=torch.ones(1,5,4),tactile_age_ms=torch.zeros(1),tactile_valid=torch.tensor([True]))
 def test_training_runtime_exact(self):
  p=self.sample();p['tactile_history'][0,2,1,0]=2
  q,donor=runtime_payload(p,'contact_state',12,{})
  self.assertIsNone(donor)
  for k,v in contact_state_payload(p).items():self.assertTrue(torch.equal(q[k],v))
 def test_zero_and_true_do_not_change(self):
  p=self.sample();p['tactile_history'][0,2,1,0]=2
  q,_=runtime_payload(p,'true',0,{})
  for k in p:self.assertTrue(torch.equal(p[k],q[k]))
  z,_=runtime_payload(p,'zero',0,{})
  self.assertEqual(z['tactile_history'].count_nonzero(),0)
  for k in p:
   if k!='tactile_history':self.assertTrue(torch.equal(p[k],z[k]))
 def test_initial_invalid(self):
  p=self.sample();p['tactile_valid'].zero_();p['tactile_mask'].zero_();p['tactile_age_ms'].fill_(-1)
  for a in ('zero','true','mismatch','contact_state'):
   q,_=runtime_payload(p,a,0,{})
   for k in p:self.assertTrue(torch.equal(p[k],q[k]))
if __name__=='__main__':unittest.main()
