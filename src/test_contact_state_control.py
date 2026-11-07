import unittest
import torch
from contact_state_control import contact_state_payload
class ContactStateTest(unittest.TestCase):
 def sample(self):return dict(tactile_history=torch.zeros(2,5,4,3),tactile_mask=torch.ones(2,5,4),tactile_age_ms=torch.zeros(2),tactile_valid=torch.ones(2,dtype=torch.bool))
 def test_magnitude_side_time_erased_without_mutation(self):
  p=self.sample();p['tactile_history'][0,0,0,0]=2;p['tactile_history'][1,4,3,1]=100
  q=contact_state_payload(p)
  self.assertTrue(torch.equal(q['tactile_history'][0],q['tactile_history'][1]))
  self.assertEqual(p['tactile_history'][0,0,0,0],2)
  self.assertEqual(q['tactile_history'][...,:2].count_nonzero(),0)
  self.assertTrue((q['tactile_history'][...,2]==1).all())
 def test_masked_force_is_not_contact(self):
  p=self.sample();p['tactile_history'][0,0,0,0]=100;p['tactile_mask'][0,0,0]=0
  self.assertEqual(contact_state_payload(p)['tactile_history'].count_nonzero(),0)
 def test_invalid_and_stale(self):
  p=self.sample();p['tactile_valid'][0]=False;p['tactile_mask'][0]=0;p['tactile_age_ms'][0]=-1
  self.assertEqual(contact_state_payload(p)['tactile_history'].count_nonzero(),0)
  p['tactile_age_ms'][1]=201
  with self.assertRaises(ValueError):contact_state_payload(p)
 def test_reject_labels(self):
  p=self.sample();p['success']=True
  with self.assertRaises(ValueError):contact_state_payload(p)
if __name__=='__main__':unittest.main()
