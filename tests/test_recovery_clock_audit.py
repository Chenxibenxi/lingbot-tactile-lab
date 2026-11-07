import copy
import unittest
from recovery_clock_audit import audit_clock


class ClockAudit(unittest.TestCase):
    def test_alignment_and_non_authorization(self):
        actions=[dict(action_index=i,before_physics_step=i*2,after_physics_step=(i+1)*2,
            capture_physics_step=i*2,camera_frame_index=i,source_role='external_intervention' if i==1 else 'expert',
            controller_method='Base_Task.take_action',training_approval=False) for i in range(3)]
        physics=[dict(physics_step=i) for i in range(1,7)]
        frames=[dict(frame_index=i,obs_index=i,physics_step=i*2) for i in range(3)]
        obs=[dict(obs_index=i,physics_step=i*2) for i in range(3)]
        result=audit_clock(actions,physics,frames,obs,official_success=True)
        self.assertTrue(result['clock_alignment_passed']);self.assertFalse(result['training_approval'])
        self.assertEqual(result['external_interventions'],1)
        bad=copy.deepcopy(obs);bad[1]['physics_step']=3
        with self.assertRaises(ValueError):audit_clock(actions,physics,frames,bad,official_success=True)
        with self.assertRaises(ValueError):audit_clock(actions,physics[:-1],frames,obs,official_success=True)
