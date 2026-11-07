"""Fake physical clock verifies proposed online actions are actually executed."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from online_expert_pilot_executor import OnlineExpertPlanError,execute_online_expert
from tests.test_online_bottle_expert_actions import FakeOnlineBottle


class ExecutableFakeBottle(FakeOnlineBottle):
    def __init__(self):
        super().__init__();self._force_probe=SimpleNamespace(step=0,failed=False)
        self.no_advance=False;self.stop_at=100
    def take_action(self,command,action_type='qpos'):
        super().take_action(command,action_type=action_type)
        if not self.no_advance:self._force_probe.step+=83
        self.eval_success=len(self.commands)>=self.stop_at


def capture(task):return {'capture_physics_step':task._force_probe.step,
                          'frame_index':len(task.commands)}


class OnlineExpertPilotExecutorTests(unittest.TestCase):
    def test_real_fake_clock_intervals_and_early_success_are_diagnostic_only(self):
        task=ExecutableFakeBottle();task.stop_at=5
        result=execute_online_expert(task,capture,physics_cap=1000)
        self.assertEqual((result['actual_take_action_calls'],result['physics_steps']),(5,415))
        self.assertEqual(result['stopped_reason'],'early_goal_success')
        self.assertFalse(result['candidate_h50_call_count_sufficient'])
        self.assertFalse(result['training_approval'])
        self.assertFalse(result['source_and_completion_HDF5_verified'])
        rows=result['private_action_rows_in_memory_only']
        self.assertEqual([(r['capture_physics_step'],r['before_physics_step'],r['after_physics_step']) for r in rows],
                         [(0,0,83),(83,83,166),(166,166,249),(249,249,332),(332,332,415)])
        self.assertTrue(all(r['joint_tracking_max_abs_difference']==0 for r in rows))

    def test_cap_stops_before_next_action_and_no_stale_camera_or_failed_clock(self):
        task=ExecutableFakeBottle()
        result=execute_online_expert(task,capture,physics_cap=170)
        self.assertEqual(result['actual_take_action_calls'],3)
        self.assertEqual(result['physics_steps'],249)
        self.assertEqual(result['stopped_reason'],'bounded_physics_cap_exceeded_inside_action')
        self.assertFalse(result['goal_success_reported'])
        with self.assertRaisesRegex(OnlineExpertPlanError,'current RGB'):
            execute_online_expert(ExecutableFakeBottle(),
                                  lambda t:{'capture_physics_step':t._force_probe.step+1,'frame_index':0})
        repeated=ExecutableFakeBottle()
        with self.assertRaisesRegex(OnlineExpertPlanError,'current RGB'):
            execute_online_expert(repeated,
                                  lambda t:{'capture_physics_step':t._force_probe.step,'frame_index':0})
        self.assertEqual(len(repeated.commands),1)
        bad=ExecutableFakeBottle();bad.no_advance=True
        with self.assertRaisesRegex(OnlineExpertPlanError,'did not advance'):
            execute_online_expert(bad,capture)

    def test_boolean_markers_forged_proposals_and_measured_state_refuse(self):
        with self.assertRaisesRegex(OnlineExpertPlanError,'current RGB'):
            execute_online_expert(ExecutableFakeBottle(),
                                  lambda t:{'capture_physics_step':False,'frame_index':0})
        row={'schema':'lingbot_online_rule_expert_proposal_v1',
             'status':'proposed_not_executed_not_training_approved',
             'proposal_index':0,'stage':'grasp',
             'planning_source':'current_scene_stage_planner_not_old_episode',
             'training_approval':False,'command_state_14':[0.0]*14}
        for field,value in (('proposal_index',False),('stage','old_replay'),
                            ('command_state_14',[True]+[0.0]*13)):
            with self.subTest(field=field):
                bad=dict(row);bad[field]=value;task=ExecutableFakeBottle()
                with patch('online_expert_pilot_executor.online_expert_proposals',return_value=iter([bad])):
                    with self.assertRaisesRegex(OnlineExpertPlanError,'invalid'):
                        execute_online_expert(task,capture)
                self.assertEqual(task.commands,[])
        task=ExecutableFakeBottle()
        task.robot.get_left_arm_real_jointState=lambda:[True]+[0.0]*5+[1.0]
        with self.assertRaisesRegex(OnlineExpertPlanError,'measured current state'):
            execute_online_expert(task,capture)

    def test_strict_opt_in_stride20_only_on_new_explicit_executor_call(self):
        baseline=execute_online_expert(ExecutableFakeBottle(),capture,physics_cap=1000)
        candidate=execute_online_expert(ExecutableFakeBottle(),capture,physics_cap=1000,stride=20)
        self.assertEqual((baseline['actual_take_action_calls'],candidate['actual_take_action_calls']),(6,11))
        self.assertEqual((baseline['planning_path_stride_indices'],candidate['planning_path_stride_indices']),(25,20))
        self.assertFalse(candidate['training_approval'])
        with self.assertRaisesRegex(OnlineExpertPlanError,'pristine clock'):
            execute_online_expert(ExecutableFakeBottle(),capture,stride=True)

    def test_wrong_cap_or_nonpristine_clock_refuses_before_generator(self):
        task=ExecutableFakeBottle()
        for cap in (True,0,10001):
            with self.subTest(cap=cap),self.assertRaisesRegex(OnlineExpertPlanError,'pristine clock'):
                execute_online_expert(task,capture,physics_cap=cap)
        task._force_probe.step=1
        with self.assertRaisesRegex(OnlineExpertPlanError,'pristine clock'):
            execute_online_expert(task,capture)
        self.assertEqual(task.commands,[])


if __name__=='__main__':unittest.main()
