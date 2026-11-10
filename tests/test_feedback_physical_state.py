import unittest
from types import SimpleNamespace
from feedback_physical_state import snapshot
class Joint:
    def __init__(self,name,dof,target):self.name,self.dof,self.target=name,dof,target
    def get_dof(self):return self.dof
    def get_name(self):return self.name
    def get_drive_target(self):return [self.target]
class PhysicalStateTest(unittest.TestCase):
    def test_reads_actual_qpos_not_drive_targets_and_respects_dof_offsets(self):
        extra=Joint('extra',2,0);arm=Joint('arm',1,9)
        entity=SimpleNamespace(get_active_joints=lambda:[extra,arm],get_qpos=lambda:[1,2,3])
        robot=SimpleNamespace(**{s+'_entity':entity for s in ('left','right')},
            **{s+'_arm_joints':[arm] for s in ('left','right')},
            get_left_gripper_val=lambda:0.4,get_right_gripper_val=lambda:0.5)
        result=snapshot(robot)
        self.assertEqual(result['left']['arm_actual_qpos'],[3])
        self.assertEqual(result['left']['arm_drive_targets'],[9])
        self.assertEqual(result['left']['gripper_control_value'],0.4)
        arm.dof=2
        with self.assertRaises(ValueError):snapshot(robot)
if __name__=='__main__':unittest.main()
