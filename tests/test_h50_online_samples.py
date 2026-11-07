"""Real H50 input/target decoding only; no model training or GPU use."""
import unittest
from unittest.mock import patch
import numpy as np
from h50_online_samples import OnlineH50Candidate,REPO,TRIAL
from paired_samples import CAMERAS


class OnlineH50CandidateSampleTests(unittest.TestCase):
    def setUp(self):
        if not (REPO/'data/paired'/TRIAL/'completion.json').exists():
            self.skipTest('private independent-seed HDF5 absent in this checkout')

    def test_real_three_view_current_frame_past_touch_and_future_fifty_call_labels(self):
        reader=OnlineH50Candidate();self.assertEqual(len(reader),8)
        sample=reader.sample(0,include_images=True);last=reader.sample(7,include_images=False)
        inp=sample['input'];target=sample['target'];meta=sample['metadata']
        self.assertEqual(set(inp['images_rgb']),set(CAMERAS))
        self.assertTrue(all(im.shape==(240,320,3) and im.dtype==np.uint8
                            for im in inp['images_rgb'].values()))
        self.assertEqual(inp['tactile_history'].shape,(5,4,3))
        self.assertEqual(inp['tactile_mask'].shape,(5,4))
        self.assertEqual(inp['command_state_14'].shape,(14,))
        self.assertEqual(target['future_expert_take_action_14'].shape,(50,14))
        self.assertEqual(target['applied_physics_spans'].shape,(50,2))
        self.assertNotIn('future_expert_take_action_14',inp)
        self.assertEqual(inp['capture_physics_step'],target['applied_physics_spans'][0,0])
        self.assertFalse(meta['approved_for_training'])
        self.assertEqual(last['input']['images_rgb'],{})
        self.assertFalse(last['metadata']['approved_for_training'])

    def test_outside_verified_anchors_or_hdf5_drift_refuses(self):
        reader=OnlineH50Candidate()
        for index in (-1,8,True):
            with self.subTest(index=index),self.assertRaises(IndexError):
                reader.sample(index,include_images=False)
        with patch('h50_online_samples.sha256',return_value='0'*64):
            with self.assertRaisesRegex(ValueError,'source raw changed after verification'):
                reader.sample(0,include_images=False)


if __name__=='__main__':unittest.main()
