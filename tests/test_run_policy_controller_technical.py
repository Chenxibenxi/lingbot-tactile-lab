"""Only existing SHA-bound technical source is read; never start RoboTwin."""
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
from run_policy_controller_technical import REPO,TRIAL,TRIAL_DIAG,TRIAL_DIAG_V3,TRIAL_CAP_V4,bottle_goal_geometry,validate_source_and_paths,validate_source_proposal,validate_failed_xyz_v2,validate_fixed_cap_xyz_v3


class ControllerPilotPreflightTests(unittest.TestCase):
    def test_one_known_seed_source_stays_verifiable_but_trial_id_cannot_be_reused(self):
        out=REPO/'data/technical_replay'/TRIAL
        proposal,audit=validate_source_proposal()
        self.assertEqual((audit['physics_steps'],proposal['proposed_waypoint_count']),(1582,64))
        self.assertEqual(proposal['actual_take_action_calls'],0)
        self.assertFalse(proposal['training_approval'])
        if out.exists():
            with self.assertRaisesRegex(ValueError,'only one fresh'):
                validate_source_and_paths(out)
        else:
            path,_,_=validate_source_and_paths(out)
            self.assertEqual(path,out)
            self.assertFalse(out.exists(),'dry-run must not create any pilot output')

    def test_user_approved_diagnostic_id_is_unique_and_not_first_trial(self):
        diagnostic=REPO/'data/technical_replay'/TRIAL_DIAG
        with self.assertRaisesRegex(ValueError,'only one fresh'):
            validate_source_and_paths(diagnostic,diagnostic=False)
        if diagnostic.exists():
            with self.assertRaisesRegex(ValueError,'only one fresh'):
                validate_source_and_paths(diagnostic,diagnostic=True)
        else:
            path,proposal,_=validate_source_and_paths(diagnostic,diagnostic=True)
            self.assertEqual(path,diagnostic)
            self.assertEqual(proposal['proposed_waypoint_count'],64)
            self.assertFalse(path.exists())

    def test_one_approved_v3_repair_requires_immutable_previous_failure(self):
        repair=REPO/'data/technical_replay'/TRIAL_DIAG_V3
        self.assertTrue(validate_failed_xyz_v2())
        with self.assertRaisesRegex(ValueError,'requires user-approved'):
            validate_source_and_paths(repair,repair_v3=True)
        if repair.exists():
            with self.assertRaisesRegex(ValueError,'only one fresh'):
                validate_source_and_paths(repair,diagnostic=True,repair_v3=True)
        else:
            path,proposal,_=validate_source_and_paths(repair,diagnostic=True,repair_v3=True)
            self.assertEqual(path,repair)
            self.assertEqual(proposal['proposed_waypoint_count'],64)
            self.assertFalse(path.exists())
        with patch('run_policy_controller_technical.sha256',return_value='0'*64):
            with self.assertRaisesRegex(ValueError,'SHA changed'):
                validate_failed_xyz_v2()

    def test_one_approved_cap8000_v4_requires_immutable_prior_six_source_shas(self):
        trial=REPO/'data/technical_replay'/TRIAL_CAP_V4
        self.assertTrue(validate_fixed_cap_xyz_v3())
        with self.assertRaisesRegex(ValueError,'requires unique'):
            validate_source_and_paths(trial,cap_v4=True)
        with self.assertRaisesRegex(ValueError,'requires unique'):
            validate_source_and_paths(trial,diagnostic=True,repair_v3=True,cap_v4=True)
        if trial.exists():
            with self.assertRaisesRegex(ValueError,'only one fresh'):
                validate_source_and_paths(trial,diagnostic=True,cap_v4=True)
        else:
            path,proposal,_=validate_source_and_paths(trial,diagnostic=True,cap_v4=True)
            self.assertEqual(path,trial)
            self.assertEqual(proposal['proposed_waypoint_count'],64)
            self.assertFalse(path.exists())
        with patch('run_policy_controller_technical.sha256',return_value='0'*64):
            with self.assertRaisesRegex(ValueError,'SHA changed'):
                validate_fixed_cap_xyz_v3()

    def test_bottle_positions_are_diagnostic_only_not_gripper_force_or_labels(self):
        class Actor:
            def __init__(self,xyz):self.xyz=xyz
            def get_functional_point(self,index):
                assert index==0
                return self.xyz
        robot=SimpleNamespace(bottle1=Actor([-0.06,-0.105,0.90,1,0,0,0]),
                              bottle2=Actor([0.2,-0.105,0.80,1,0,0,0]),
                              left_target_pose=[-0.06,-0.105,1],
                              right_target_pose=[0.06,-0.105,1])
        result=bottle_goal_geometry(robot)
        self.assertTrue(result['left']['xy_within_task_gate'])
        self.assertTrue(result['left']['z_above_task_gate'])
        self.assertFalse(result['right']['xy_within_task_gate'])
        self.assertFalse(result['right']['z_above_task_gate'])

    def test_other_seed_or_output_root_rejected_before_source_read(self):
        for p in (REPO/'data/paired'/TRIAL,
                  REPO/'data/technical_replay/relay_some_other_seed',
                  Path('/tmp/h50-technical-trial')):
            with self.subTest(path=p),self.assertRaisesRegex(ValueError,'only one fresh'):
                validate_source_and_paths(p)


if __name__=='__main__':unittest.main()
