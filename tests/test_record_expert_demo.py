"""CPU-only failed expert planner/goal diagnostics, no SAPIEN import."""
import unittest
from record_expert_demo import evaluate_outcome,planner_path_statuses


class FakeRobot:
    def __init__(self,planner,goal=None,error=None):
        self.plan_success=planner;self.goal=goal;self.error=error;self.calls=0
    def check_success(self):
        self.calls+=1
        if self.error: raise self.error
        return self.goal


class ExpertOutcomeTests(unittest.TestCase):
    def test_planner_failure_does_not_skip_goal_observation(self):
        robot=FakeRobot(False,True)
        self.assertEqual(evaluate_outcome(robot),(False,True,None))
        self.assertEqual(robot.calls,1)

    def test_planner_status_summary_excludes_joint_trajectories(self):
        robot=FakeRobot(False)
        robot.left_joint_path=[{'status':'IK Failed','position':[13]*100},None]
        robot.right_joint_path=[{'status':'Success','position':[17]*100}]
        self.assertEqual(planner_path_statuses(robot),
                         {'left':['IK Failed','missing_result'],'right':['Success']})

    def test_goal_failure_is_distinguished_from_exception(self):
        self.assertEqual(evaluate_outcome(FakeRobot(True,False)),(True,False,None))
        robot=FakeRobot(True,error=ValueError('nonprivate diagnostic'))
        self.assertEqual(evaluate_outcome(robot),(True,False,'ValueError'))
        self.assertEqual(robot.calls,1)


if __name__=='__main__':unittest.main()
