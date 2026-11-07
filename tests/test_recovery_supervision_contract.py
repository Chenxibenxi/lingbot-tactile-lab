import copy
import unittest
from recovery_supervision_contract import recovery_target_mask


def rows():
    return [dict(action_index=i,before_physics_step=10*i,after_physics_step=10*(i+1),
                 capture_physics_step=10*i,source_role='external_intervention' if i==2 else 'expert')
            for i in range(6)]


class RecoveryContract(unittest.TestCase):
    def test_horizon_cannot_cross_intervention(self):
        r=recovery_target_mask(rows(),official_success=True,horizon=4)
        self.assertEqual(r['target_valid_mask'],[
            [True,True,False,False],[True,False,False,False],[False]*4,
            [True,True,True,False],[True,True,False,False],[True,False,False,False]])
        self.assertFalse(r['training_approval'])
        self.assertTrue(r['requires_same_run_source_and_completion_audit'])

    def test_failure_never_becomes_successful_expert_supervision(self):
        r=recovery_target_mask(rows(),official_success=False,horizon=50)
        self.assertFalse(any(any(mask) for mask in r['target_valid_mask']))

    def test_camera_clock_and_intervals_strict(self):
        for key,value in [('capture_physics_step',False),('capture_physics_step',1),
                          ('before_physics_step',-1),('after_physics_step',0),
                          ('action_index',False),('source_role','policy')]:
            data=copy.deepcopy(rows());data[0][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                recovery_target_mask(data,official_success=True)
        data=rows();data[1]['before_physics_step']=11;data[1]['capture_physics_step']=11
        with self.assertRaises(ValueError):recovery_target_mask(data,official_success=True)

    def test_input_not_mutated_and_no_terminal_repeat(self):
        data=rows();before=copy.deepcopy(data)
        r=recovery_target_mask(data,official_success=True)
        self.assertEqual(data,before)
        self.assertEqual(sum(r['target_valid_mask'][-1]),1)
