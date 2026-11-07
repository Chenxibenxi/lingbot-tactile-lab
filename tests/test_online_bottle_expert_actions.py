"""CPU fake-world proof of online stage timing and fail-closed planner commands."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from online_bottle_expert_actions import OnlineExpertPlanError,online_expert_proposals


class FakeRobot:
    def __init__(self):
        self.states={'left':[0.0]*6+[1.0],'right':[0.0]*6+[1.0]}
        self.plans=[];self.force_failure=False;self.bad_number=False
    def get_left_arm_jointState(self):return [99.0]*6+[self.states['left'][6]]  # stale drive target
    def get_right_arm_jointState(self):return [99.0]*6+[self.states['right'][6]]
    def get_left_arm_real_jointState(self):return list(self.states['left'])
    def get_right_arm_real_jointState(self):return list(self.states['right'])
    def left_plan_path(self,pose,constraint_pose=None):return self._plan('left',pose,constraint_pose)
    def right_plan_path(self,pose,constraint_pose=None):return self._plan('right',pose,constraint_pose)
    def _plan(self,arm,pose,constraint_pose):
        self.plans.append((arm,self.states[arm][0],pose[0],constraint_pose))
        if self.force_failure and arm=='right':return {'status':'No Solution'}
        start=self.states[arm][:6];target=[pose[0]]*6
        path=[[start[j]+(target[j]-start[j])*i/25 for j in range(6)] for i in range(26)]
        if self.bad_number:path[12][0]=float('nan')
        return {'status':'Success','position':path}


class FakeOnlineBottle:
    def __init__(self):
        self.robot=FakeRobot();self.need_plan=True;self.plan_success=True
        self.eval_success=False;self.bottle1=object();self.bottle2=object()
        self.left_target_pose=[-0.06,-0.105,1,0,1,0,0]
        self.right_target_pose=[0.06,-0.105,1,0,1,0,0]
        self.commands=[];self.stage_queries=[]
    def _action(self,arm,verb,target=None):
        return SimpleNamespace(arm_tag=arm,action=verb,target_pose=target,
                               target_gripper_pos=0.0 if verb=='gripper' else None,
                               args={'constraint_pose':None})
    def grasp_actor(self,actor,arm_tag,pre_grasp_dis):
        assert pre_grasp_dis==0.08
        self.stage_queries.append(('grasp',len(self.commands),arm_tag))
        return arm_tag,[self._action(arm_tag,'move',[1,0,0,1,0,0,0]),
                        self._action(arm_tag,'move',[2,0,0,1,0,0,0]),
                        self._action(arm_tag,'gripper')]
    def move_by_displacement(self,arm_tag,z):
        assert z==0.1
        self.stage_queries.append(('lift',len(self.commands),arm_tag))
        return arm_tag,[self._action(arm_tag,'move',[3,0,0,1,0,0,0])]
    def place_actor(self,actor,arm_tag,target_pose,functional_point_id,pre_dis,dis,is_open):
        assert functional_point_id==0 and pre_dis==dis==0 and is_open is False
        self.stage_queries.append(('place',len(self.commands),arm_tag))
        return arm_tag,[self._action(arm_tag,'move',[4,0,0,1,0,0,0]),
                        self._action(arm_tag,'move',[5,0,0,1,0,0,0])]
    def take_action(self,command,action_type='qpos'):
        assert action_type=='qpos' and len(command)==14
        self.commands.append(list(command))
        self.robot.states['left']=list(command[:7])
        self.robot.states['right']=list(command[7:])


class OnlineExpertGeneratorTests(unittest.TestCase):
    def test_stage_targets_are_queried_only_after_prior_actions_execute(self):
        task=FakeOnlineBottle()
        it=online_expert_proposals(task)
        self.assertEqual(task.stage_queries,[])
        first=next(it)
        self.assertEqual(first['stage'],'grasp')
        self.assertEqual({q[0] for q in task.stage_queries},{'grasp'})
        self.assertFalse(first['training_approval'])
        assert first['command_state_14'][6]==first['command_state_14'][13]==1.0
        task.take_action(first['command_state_14'])
        for candidate in it:
            task.take_action(candidate['command_state_14'])
        self.assertEqual(len(task.commands),6)
        self.assertEqual([q[1] for q in task.stage_queries if q[2]=='left'],[0,3,4])
        self.assertEqual(task.commands[2][6],0.0)
        self.assertEqual(task.commands[2][13],0.0)
        self.assertEqual(task.commands[2][0],2.0)  # measured, NOT fake stale drive target 99
        self.assertEqual(task.commands[2][7],2.0)
        # Lift is planned from executed grasp, place from executed lift,
        # never from the old successful same-seed drive trajectory.
        self.assertIn(('left',2.0,3.0,None),task.robot.plans)
        self.assertIn(('right',3.0,4.0,None),task.robot.plans)

    def test_invalid_plan_and_future_nan_never_yield_training_label(self):
        for reason in ('failed_right','nan_path','wrong_side','bad_gripper','boolean_joint'):
            with self.subTest(reason=reason):
                task=FakeOnlineBottle()
                if reason=='failed_right':task.robot.force_failure=True
                if reason=='nan_path':task.robot.bad_number=True
                if reason=='wrong_side':
                    original=task.grasp_actor
                    def swap(actor,arm_tag,pre_grasp_dis):
                        tag,actions=original(actor,arm_tag,pre_grasp_dis)
                        if arm_tag=='right':return 'left',actions
                        return tag,actions
                    task.grasp_actor=swap
                if reason=='bad_gripper':task.robot.states['left'][6]=1.5
                if reason=='boolean_joint':task.robot.states['left'][0]=True
                with self.assertRaises(OnlineExpertPlanError):
                    list(online_expert_proposals(task))

    def test_opt_in_density_candidate_changes_only_new_phase_waypoint_count(self):
        def exercise(stride=None):
            task=FakeOnlineBottle()
            iterator=online_expert_proposals(task) if stride is None else online_expert_proposals(task,stride=stride)
            stages=[]
            for proposal in iterator:
                stages.append(proposal['stage'])
                task.take_action(proposal['command_state_14'])
                self.assertFalse(proposal['training_approval'])
            return task,stages
        baseline,old=exercise()
        candidate,new=exercise(20)
        self.assertEqual((len(old),len(new)),(6,11))
        self.assertEqual((old.count('grasp'),new.count('grasp')),(3,5))
        self.assertEqual((old.count('lift'),new.count('lift')),(1,2))
        self.assertEqual((old.count('place'),new.count('place')),(2,4))
        self.assertEqual(candidate.commands[4][6],0.0)  # one close, never pad grips
        with self.assertRaisesRegex(OnlineExpertPlanError,'pinned'):
            list(online_expert_proposals(FakeOnlineBottle(),stride=19))
        with self.assertRaisesRegex(OnlineExpertPlanError,'pinned'):
            list(online_expert_proposals(FakeOnlineBottle(),stride=True))
        self.assertEqual(len(baseline.commands),6)

    def test_early_goal_does_not_pad_to_fifty_actions(self):
        task=FakeOnlineBottle();it=online_expert_proposals(task)
        first=next(it);task.take_action(first['command_state_14'])
        task.eval_success=True
        self.assertEqual(list(it),[])
        self.assertEqual(len(task.commands),1)


if __name__=='__main__':unittest.main()
