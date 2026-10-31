"""No GPU: exact policy RGB and actually executed action storage contract."""
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from policy_rollout_capture import PolicyRolloutWriter
from tests.test_tactile_rpc import window


def observation():
    rgb=np.ones((3,4,3),dtype=np.uint8)*17
    return {'observation.images.cam_high':rgb,
            'observation.images.cam_left_wrist':rgb+1,
            'observation.images.cam_right_wrist':rgb+2,
            'observation.state':np.arange(14,dtype=np.float32),
            'task':'Lift the pot'}


class PolicyRolloutCaptureTests(unittest.TestCase):
    def test_same_episode_frames_causal_tactile_and_applied_commands(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'rollout.hdf5'
            with PolicyRolloutWriter(path,task='lift_pot',seed=100048,episode=0) as writer:
                writer.begin_request({'request_index':0,'obs_index':0,'physics_step':0},observation())
                writer.applied_action(np.arange(14,dtype=np.float32),physics_step_before=0,physics_step_after=25)
                writer.begin_request({'request_index':1,'obs_index':1,'physics_step':25},
                                     observation(),window())
                writer.applied_action(np.ones(14,dtype=np.float32),physics_step_before=25,physics_step_after=50)
                writer.close(observed_success=True)
            with h5py.File(path,'r') as f:
                self.assertEqual(f.attrs['schema'],'phizju_policy_rollout_capture_v1')
                self.assertIn('not_expert_or_training_approved',f.attrs['target_semantics'])
                self.assertTrue(f.attrs['complete'])
                self.assertEqual(f.attrs['policy_reported_success'],1)
                self.assertEqual(f['images/head'].shape,(2,3,4,3))
                np.testing.assert_array_equal(f['images/right'][1],observation()['observation.images.cam_right_wrist'])
                np.testing.assert_array_equal(f['request/physics_step'][:],[0,25])
                np.testing.assert_array_equal(f['applied/physics_step_after'][:],[25,50])
                np.testing.assert_array_equal(f['applied/command_14'][0],np.arange(14))
                np.testing.assert_array_equal(f['tactile/window_valid'][:,0],[False,True])

    def test_reject_future_window_wrong_rgb_and_nonmonotonic_applied_action(self):
        with tempfile.TemporaryDirectory() as td:
            with PolicyRolloutWriter(Path(td)/'partial.hdf5',task='lift_pot',seed=1,episode=0) as writer:
                invalid=observation();invalid['observation.images.cam_high']=np.ones((3,4,3),dtype=np.float32)
                with self.assertRaisesRegex(ValueError,'RGB'):
                    writer.begin_request({'request_index':0,'obs_index':0,'physics_step':0},invalid)
                with self.assertRaisesRegex(ValueError,'inference step'):
                    writer.begin_request({'request_index':0,'obs_index':0,'physics_step':0},observation(),window())
                writer.begin_request({'request_index':0,'obs_index':0,'physics_step':0},observation())
                with self.assertRaisesRegex(ValueError,'move forward'):
                    writer.applied_action(np.zeros(14),physics_step_before=0,physics_step_after=0)
                with self.assertRaisesRegex(ValueError,'no applied action'):
                    writer.begin_request({'request_index':1,'obs_index':1,'physics_step':25},observation())
            with h5py.File(Path(td)/'partial.hdf5') as f:
                self.assertFalse(f.attrs['complete'])

    def test_partial_second_request_cannot_mark_capture_complete(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'partial.hdf5'
            with PolicyRolloutWriter(path,task='lift_pot',seed=1,episode=0) as writer:
                writer.begin_request({'request_index':0,'obs_index':0,'physics_step':0},observation())
                writer.applied_action(np.zeros(14),physics_step_before=0,physics_step_after=25)
                changed=observation()
                changed['observation.images.cam_left_wrist']=np.zeros((5,4,3),dtype=np.uint8)
                with self.assertRaisesRegex(ValueError,'changing capture geometry'):
                    writer.begin_request({'request_index':1,'obs_index':1,'physics_step':25},changed)
                writer.close(observed_success=True)
            with h5py.File(path) as f:
                self.assertFalse(f.attrs['complete'])
                self.assertEqual(len(f['images/head']),2)
                self.assertEqual(len(f['images/left']),1)

    def test_never_overwrite_an_episode_capture(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'rollout.hdf5'
            with PolicyRolloutWriter(path,task='lift_pot',seed=1,episode=0):pass
            with self.assertRaises((FileExistsError,OSError)):
                PolicyRolloutWriter(path,task='lift_pot',seed=1,episode=0)


if __name__=='__main__':unittest.main()
