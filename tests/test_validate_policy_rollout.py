"""Same-run policy HDF5/raw/obs/request join, diagnostic not expert training."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS

import h5py
import numpy as np
from force_probe import ForceProbe
from policy_rollout_capture import PolicyRolloutWriter
from tests.test_force_probe import fake_robot
from validate_policy_rollout import validate


class PolicyRolloutValidatorTests(unittest.TestCase):
    def make_data(self,root):
        robot=fake_robot();scene=NS(get_contacts=lambda:[])
        with patch.dict(os.environ,{'TACTILE_LIVE_WINDOW':'1'}):
            probe=ForceProbe(Path(root)/'force',task='lift_pot',seed=100048,episode=0,dt=.004,robot=robot)
        img=np.zeros((3,4,3),dtype=np.uint8)
        frame={'observation':{cam:{'rgb':img} for cam in ('head_camera','left_camera','right_camera')}}
        observation={key:img for key in ('observation.images.cam_high',
            'observation.images.cam_left_wrist','observation.images.cam_right_wrist')}
        observation.update({'observation.state':np.ones(14,dtype=np.float32),'task':'lift'})
        hdf5=Path(root)/'self_rollout.hdf5'
        with PolicyRolloutWriter(hdf5,task='lift_pot',seed=100048,episode=0) as writer:
            probe.record_observation(frame,0)
            writer.begin_request(probe.record_policy_request(),observation,probe.policy_window())
            for _ in range(25):probe.sample(scene,robot,1)
            writer.applied_action(np.ones(14),physics_step_before=0,physics_step_after=probe.step)
            probe.record_observation(frame,1)
            writer.begin_request(probe.record_policy_request(),observation,probe.policy_window())
            for _ in range(10):probe.sample(scene,robot,2)
            writer.applied_action(np.ones(14)*2,physics_step_before=25,physics_step_after=probe.step)
            writer.close(observed_success=True)
        raw=probe.raw_path;probe.close()
        return raw,raw.with_suffix('.obs.jsonl'),raw.with_suffix('.policy.jsonl'),hdf5

    def test_rgb_actions_and_force_join_without_training_approval(self):
        with tempfile.TemporaryDirectory() as td:
            result=validate(*self.make_data(td))
            self.assertEqual(result['status'],'candidate_self_rollout_not_expert_or_training_approved')
            self.assertEqual(result['actual_infer_attempts'],2)
            self.assertEqual(result['executed_action_count'],2)
            self.assertEqual(result['causal_valid_requests'],1)
            self.assertEqual(result['policy_reported_success'],1)
            self.assertEqual(len(result['sha256']['hdf5']),64)

    def test_reject_future_applied_action_and_wrong_tactile(self):
        with tempfile.TemporaryDirectory() as td:
            paths=self.make_data(td)
            with h5py.File(paths[-1],'r+') as f:f['applied/physics_step_after'][1]=36
            with self.assertRaisesRegex(ValueError,'not causal'):
                validate(*paths)
        with tempfile.TemporaryDirectory() as td:
            paths=self.make_data(td)
            with h5py.File(paths[-1],'r+') as f:f['tactile/history'][1,0,0,0]=99
            with self.assertRaisesRegex(ValueError,'tactile differs'):
                validate(*paths)


if __name__=='__main__':unittest.main()
