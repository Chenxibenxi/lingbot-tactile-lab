import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import test_recovery_prefix_generation_v2 as generators
from recovery_prefix_generation_v2 import seal_prefix
from recovery_paired_executor_v2 import execute_pair


class PairTest(unittest.TestCase):
    def run_pair(self, condition, drift=False):
        doc = generators.GenerationTest().generate()
        task = SimpleNamespace(eval_success=False, check_success=lambda:True,
            robot=SimpleNamespace(get_left_arm_real_jointState=lambda:[0.]*7,
                                  get_right_arm_real_jointState=lambda:[0.]*7))
        clock = [0]; receipts = []; stages = []; events = []
        final_count = 3 if condition == 'expert_control' else 6
        def execute(command):
            before = clock[0]; clock[0] += 10
            receipts.append(dict(camera_frame_index=len(receipts), capture_physics_step=before,
                before_physics_step=before, after_physics_step=clock[0],
                command_state_14=list(command), controller_method='Base_Task.take_action', training_approval=False))
            task.eval_success = len(receipts) == final_count
        def state():
            result = copy.deepcopy(doc['initial_state' if clock[0] == 0 else 'boundary_state'])
            if drift and clock[0] > 0: result['values']['bottle1.position'][0] += 1
            return result
        def commands(phase):
            stages.append(phase); yield [0.]*14
        bridge = SimpleNamespace(receipts=receipts, read_clock=lambda:clock[0], read_state=state,
            read_rgb_hashes=lambda:doc['initial_rgb_sha256'], execute=execute)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'prefix.json'; seal = seal_prefix(path, doc, expected_identity=doc['identity'])
            args = dict(expected_sha256=seal['sha256'], expected_identity=doc['identity'],
                        condition=condition, emit=events.append, bridge=bridge, commands=commands)
            if drift:
                with self.assertRaises(ValueError): execute_pair(task,path,**args)
                self.assertEqual(stages, [])
                self.assertEqual(len(receipts), 2)
                return
            result = execute_pair(task,path,**args)
        self.assertTrue(result['official_success'])
        self.assertFalse(result['training_approval'])
        return result, stages

    def test_control_only_plans_place(self):
        result, stages = self.run_pair('expert_control')
        self.assertEqual(stages, ['place'])
        self.assertFalse(result['intervention_applied'])

    def test_recovery_intervention_is_excluded(self):
        result, stages = self.run_pair('left_open_once_recovery')
        self.assertEqual(stages, ['grasp','lift','place'])
        self.assertEqual(result['rows'][2]['source_role'], 'external_intervention')
        self.assertEqual(result['rows'][2]['command_state_14'][6], 1.)
        masks = result['target_contract']['target_valid_mask']
        self.assertFalse(any(masks[2]))
        self.assertEqual(masks[0][:3], [True,True,False])

    def test_no_branch_after_boundary_failure(self):
        self.run_pair('left_open_once_recovery', drift=True)
