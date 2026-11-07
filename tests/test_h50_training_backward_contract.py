import unittest
import torch
from probe_h50_training_backward import validate_batch

def batch():
    mask=torch.zeros(1,50,55,dtype=torch.bool);mask[:,:,:14]=True
    return {'actions':torch.zeros(1,50,55),'joint_mask':mask,'action_is_pad':torch.zeros(1,50)}
class TrainingContract(unittest.TestCase):
    def test_real_horizon_accepted(self):validate_batch(batch())
    def test_padding_cannot_enter_loss(self):
        b=batch();b['joint_mask'][:,:,14]=True
        with self.assertRaises(ValueError):validate_batch(b)
    def test_short_episode_not_synthesized(self):
        b=batch();b['action_is_pad'][0,-1]=1
        with self.assertRaises(ValueError):validate_batch(b)
    def test_nan_refused(self):
        b=batch();b['actions'][0,0,0]=float('nan')
        with self.assertRaises(ValueError):validate_batch(b)
if __name__=='__main__':unittest.main()
