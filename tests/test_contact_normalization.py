"""Positive-bin train-only normalization keeps zero force exactly zero."""
import unittest
import numpy as np
import torch
from contact_normalization import fit_positive_contact_stats
from tactile_adapter import TactileStateResidual


def sample(trial,scale=1.):
    history=np.zeros((5,4,3),dtype=np.float32)
    for t in range(5):
        history[t,:,:2]=t*scale
        history[t,:,2]=float(t>0)
    return {'metadata':{'trial_id':trial},'input':{
        'tactile_window_valid':True,'tactile_history':history,
        'tactile_mask':np.ones((5,4),dtype=np.float32)}}


class ContactNormTests(unittest.TestCase):
    def test_only_train_positive_bins_change_stats(self):
        train=[sample('a',1.),sample('b',2.)]
        baseline=fit_positive_contact_stats(train,{'a','b'})
        plus_val=fit_positive_contact_stats([sample('heldout',1e7),*train],{'a','b'})
        self.assertEqual(baseline,plus_val)
        self.assertEqual(baseline['schema'],'lingbot_tactile_contact_norm_v2')
        self.assertEqual(baseline['positive_force_counts_by_finger_channel'],[[8,8]]*4)
        self.assertGreater(baseline['median'][0][0],0)
        self.assertEqual(baseline['median'][0][2],0)
        self.assertEqual(baseline['scale_iqr_floor_0p1'][0][2],1)
        self.assertFalse(baseline['training_approval'])

    def test_adapter_refuses_candidate_and_zero_preserving_approved_test_contract(self):
        stats=fit_positive_contact_stats([sample('a'),sample('b',2.)],{'a','b'})
        module=TactileStateResidual(12,8)
        with self.assertRaisesRegex(ValueError,'not approved'):
            module.set_normalization(stats)
        # Synthetic unit-only approval tests mathematics, not real pilot approval.
        module.set_normalization(dict(stats,training_approval=True))
        h=torch.zeros((1,5,4,3));h[0,0,0,:2]=1e7;h[0,0,0,2]=1
        out=module._normalized_features(h)
        self.assertTrue(torch.equal(out[0,1,0],torch.zeros(3)))
        self.assertTrue(torch.equal(out[0,0,0,2:],torch.ones(1)))
        self.assertLessEqual(abs(out[0,0,0,0].item()),4.)
        self.assertLessEqual(abs(out[0,0,0,1].item()),4.)
        self.assertEqual(module.norm_mode.item(),1)
        self.assertTrue(torch.equal(module(torch.ones(1,12),h,torch.ones(1,5,4),
                                               torch.zeros(1),torch.ones(1,dtype=torch.bool)),torch.ones(1,12)))
        cloned=TactileStateResidual(12,8);cloned.load_state_dict(module.state_dict())
        self.assertEqual(cloned.norm_mode.item(),1)

    def test_no_positive_train_bins_and_bad_clip_fail(self):
        with self.assertRaisesRegex(ValueError,'too few positive'):
            fit_positive_contact_stats([sample('a',0.)],{'a'})
        with self.assertRaisesRegex(ValueError,'positive finite'):
            fit_positive_contact_stats([sample('a')],{'a'},clip_abs=float('nan'))


if __name__=='__main__':unittest.main()
