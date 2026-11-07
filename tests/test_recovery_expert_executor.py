from types import SimpleNamespace
import unittest
from unittest.mock import patch
from recovery_expert_executor import execute_recovery


class Task:
    def __init__(self):
        self._force_probe=SimpleNamespace(step=0,failed=False)
        self.eval_success=False;self.commands=[]
        self.robot=SimpleNamespace(get_left_arm_real_jointState=lambda:[0.]*7,
                                   get_right_arm_real_jointState=lambda:[0.]*7)
    def take_action(self,command,action_type):
        assert action_type=='qpos'
        self.commands.append(list(command));self._force_probe.step+=10
        if command[0]==3.:self.eval_success=True
    def check_success(self):return self.eval_success


def proposals(task,**kwargs):
    for i,phase in enumerate(('grasp','lift','place')):
        command=[0.]*14;command[0]=float(i+1)
        yield dict(schema='lingbot_online_rule_expert_proposal_v1',training_approval=False,
                   status='proposed_not_executed_not_training_approved',proposal_index=i,stage=phase,
                   planning_source='current_scene_stage_planner_not_old_episode',command_state_14=command)


def capture(task):return dict(capture_physics_step=task._force_probe.step,frame_index=len(task.commands))


class RecoveryExecutor(unittest.TestCase):
    def test_single_intervention_and_replan(self):
        task=Task()
        with patch('recovery_expert_executor.online_expert_proposals',side_effect=proposals) as planner:
            result=execute_recovery(task,capture,perturb=True)
        self.assertEqual(planner.call_count,2)
        self.assertTrue(result['official_success'])
        self.assertEqual([r['source_role'] for r in result['rows']],['expert','expert','external_intervention','expert','expert','expert'])
        self.assertEqual(task.commands[2][6],1.)
        self.assertEqual([x[0] for x in task.commands],[1.,2.,0.,1.,2.,3.])
        self.assertEqual(sum(result['target_contract']['target_valid_mask'][0]),2)
        self.assertFalse(any(result['target_contract']['target_valid_mask'][2]))
        self.assertFalse(result['training_approval'])

    def test_control_does_not_replan_or_perturb(self):
        with patch('recovery_expert_executor.online_expert_proposals',side_effect=proposals) as planner:
            result=execute_recovery(Task(),capture,perturb=False)
        self.assertEqual(planner.call_count,1)
        self.assertFalse(result['intervention_applied'])
        self.assertEqual(result['action_count'],3)

    def test_cap_stops_without_fake_success_or_labels(self):
        with patch('recovery_expert_executor.online_expert_proposals',side_effect=proposals):
            result=execute_recovery(Task(),capture,perturb=True,action_cap=3)
        self.assertEqual(result['termination'],'bounded_action_cap')
        self.assertFalse(result['official_success'])
        self.assertFalse(any(any(x) for x in result['target_contract']['target_valid_mask']))
