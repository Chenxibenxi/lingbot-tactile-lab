"""No validation/test use, no model forward or GPU in visual-input preflight."""
import unittest
from unittest.mock import patch
import torch
from prepare_real_vla_visual_smoke import TRIAL,check_train_anchor,prepare


class VisualSmokeInputTests(unittest.TestCase):
    def test_heldout_or_different_seed_rejected_before_any_hdf5_read(self):
        with patch('prepare_real_vla_visual_smoke.PairedEpisode',side_effect=AssertionError('must not read')):
            with self.assertRaisesRegex(ValueError,'only original relay TRAIN'):
                prepare('/mnt/relaydata/lingbot-tactile-lab',trial_id='relay_liftpot_s100048_expert1')
            with self.assertRaisesRegex(ValueError,'only fixed first causal'):
                prepare('/mnt/relaydata/lingbot-tactile-lab',trial_id=TRIAL,index=1)

    def test_original_qwen_transform_on_first_source_verified_train_frame(self):
        cfg,obs,tactile,card=prepare('/mnt/relaydata/lingbot-tactile-lab')
        self.assertEqual(cfg.num_steps,2)
        self.assertEqual(tuple(obs['state'].shape),(55,))
        self.assertEqual(tuple(obs['image_grid_thw'].shape),(3,3))
        self.assertEqual(tuple(obs['img_masks'].shape),(3,))
        self.assertTrue(bool(tactile['tactile_valid'].item()))
        self.assertEqual(tuple(tactile['tactile_history'].shape),(1,5,4,3))
        self.assertEqual(card['trial_role'],'train')
        self.assertFalse(card['training_approval'])
        self.assertFalse(card['future_target_exposed_to_model'])
        self.assertNotIn('target',obs)
        self.assertTrue(torch.isfinite(obs['state']).all())


if __name__=='__main__':unittest.main()
