import unittest
import numpy as np
from h50_contact_readiness import assess

def sample(force=0.,mask=1.):
    h=np.zeros((5,4,3));h[...,:2]=force
    return {'input':{'tactile_history':h,'tactile_mask':np.full((5,4),mask),'capture_physics_step':100},
      'target':{'future_expert_take_action_14':np.zeros((50,14)),
                'applied_physics_spans':np.array([[100+i,101+i] for i in range(50)])},
      'metadata':{'seed':100023,'trial_id':'same_episode','frame_index':2}}

class ContactGate(unittest.TestCase):
    def test_zero_force_cannot_support_tactile_training(self):
        r=assess([sample() for _ in range(8)])
        self.assertEqual(r['independent_seeds'],1)
        self.assertEqual(r['contact_windows'],0)
        self.assertFalse(r['sufficient_for_training'])
    def test_missing_coverage_not_positive(self):
        self.assertFalse(assess([sample(10.,0.)])['contact_signal_present'])
    def test_contact_is_not_training_approval(self):
        r=assess([sample(2.)]);self.assertTrue(r['contact_signal_present'])
        self.assertFalse(r['sufficient_for_training'])
    def test_wrong_clock_rejected(self):
        s=sample();s['target']['applied_physics_spans'][1,0]+=1
        with self.assertRaises(ValueError):assess([s])
    def test_short_horizon_rejected(self):
        s=sample();s['target']['future_expert_take_action_14']=np.zeros((49,14))
        with self.assertRaises(ValueError):assess([s])
if __name__=='__main__':unittest.main()
