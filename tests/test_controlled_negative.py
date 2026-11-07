"""Open grasp touches only two simulated grasp commands, not task source."""
import unittest
from controlled_negative import install_open_grasp_control,check_intervention


class FakeRobot:
    def __init__(self):self.actions=[]
    def grasp_actor(self,actor,*,arm_tag,pre_grasp_dis=.1,gripper_pos=0.):
        self.actions.append((actor,arm_tag,pre_grasp_dis,gripper_pos))
        return arm_tag,{'object':actor,'gripper_pos':gripper_pos}


class ControlledNegativeTests(unittest.TestCase):
    def test_two_explicit_open_grasps_do_not_change_source_defaults(self):
        robot=FakeRobot();original=robot.grasp_actor.__func__
        calls=install_open_grasp_control(robot)
        for side in ('left','right'):
            action=robot.grasp_actor('bottle',arm_tag=side,pre_grasp_dis=.08)
            self.assertEqual(action[1]['gripper_pos'],1.)
        self.assertEqual(len(calls),2)
        self.assertTrue(check_intervention(calls,True,False))
        self.assertIs(FakeRobot.grasp_actor,original)
        self.assertEqual(FakeRobot().grasp_actor('bottle',arm_tag='left')[1]['gripper_pos'],0.)

    def test_changed_callsite_or_success_refused_as_negative(self):
        robot=FakeRobot();calls=install_open_grasp_control(robot)
        with self.assertRaisesRegex(ValueError,'unexpected explicit'):
            robot.grasp_actor('bottle',arm_tag='left',gripper_pos=0.)
        with self.assertRaisesRegex(ValueError,'exactly two'):
            check_intervention(calls,True,False)
        robot.grasp_actor('bottle',arm_tag='left');robot.grasp_actor('bottle',arm_tag='right')
        with self.assertRaisesRegex(ValueError,'failed task goal'):
            check_intervention(calls,True,True)
        with self.assertRaisesRegex(ValueError,'successful planner'):
            check_intervention(calls,False,False)


if __name__=='__main__':unittest.main()
