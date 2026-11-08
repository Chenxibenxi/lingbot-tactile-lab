from types import SimpleNamespace
import unittest
import numpy as np
from feedback_topp_diagnostics import observe_topp

class Planner:
    def TOPP(self,*args,**kwargs):
        self.received=(args,kwargs)
        if getattr(self,'failure',False):raise ValueError('original planner failure')
        self.result=(np.zeros(10),np.zeros((10,6)),np.zeros((10,6)),np.zeros((10,6)),.04)
        return self.result

class ToppTrace(unittest.TestCase):
    def robot(self):return SimpleNamespace(left_mplib_planner=Planner(),right_mplib_planner=Planner())
    def test_preserves_argument_and_result_identity_and_restores(self):
        robot=self.robot();events=[];path=np.zeros((2,6));before=path.copy()
        with observe_topp(robot,events.append,lambda:{'action_index':0,'physics_step':0}) as trace:
            result=robot.left_mplib_planner.TOPP(path,.004,verbose=True)
            self.assertIs(result,robot.left_mplib_planner.result)
            self.assertIs(robot.left_mplib_planner.received[0][0],path)
            trace.require_valid()
        self.assertTrue(np.array_equal(path,before));self.assertNotIn('TOPP',vars(robot.left_mplib_planner))
        self.assertEqual(events[-1]['samples'],10);self.assertEqual(events[0]['step'],.004)
    def test_original_exception_is_preserved(self):
        robot=self.robot();robot.right_mplib_planner.failure=True;events=[]
        with observe_topp(robot,events.append,lambda:{}) as trace:
            with self.assertRaisesRegex(ValueError,'original planner failure'):robot.right_mplib_planner.TOPP(np.zeros((2,6)),.004)
            trace.require_valid()
        self.assertEqual(events[-1]['event'],'topp_exception')
    def test_logging_failure_does_not_become_planner_fallback(self):
        robot=self.robot()
        def broken(event):raise OSError('disk unavailable')
        with observe_topp(robot,broken,lambda:{}) as trace:
            result=robot.left_mplib_planner.TOPP(np.zeros((2,6)),.004)
            self.assertIs(result,robot.left_mplib_planner.result)
            with self.assertRaisesRegex(RuntimeError,'capture failed'):trace.require_valid()

class ObservedController(unittest.TestCase):
    def task(self, failure=False):
        robot=SimpleNamespace(left_mplib_planner=Planner(),right_mplib_planner=Planner())
        robot.left_mplib_planner.failure=failure
        class Task:
            def __init__(self):self.robot=robot;self.steps=0;self.fallback=False
            def take_action(self, command, action_type):
                self.command=command;self.action_type=action_type
                try:n=self.robot.left_mplib_planner.TOPP(command,.004)[1].shape[0]
                except Exception:n=50;self.fallback=True
                self.steps+=n
                return n
        return Task()
    def test_controller_fallback_and_step_count_unchanged(self):
        from feedback_topp_diagnostics import observed_action
        for failure in (False,True):
            baseline=self.task(failure);observed=self.task(failure);command=np.zeros((2,6));events=[]
            expected=baseline.take_action(command,'qpos')
            actual=observed_action(observed,command,emit=events.append,context=lambda:{'physics_step':observed.steps})
            self.assertEqual(actual,expected);self.assertEqual(observed.steps,baseline.steps)
            self.assertEqual(observed.fallback,baseline.fallback);self.assertIs(observed.command,command)
            self.assertNotIn('TOPP',vars(observed.robot.left_mplib_planner))
    def test_recorder_failure_surfaces_after_action_without_fallback(self):
        from feedback_topp_diagnostics import observed_action
        task=self.task()
        def broken(event):raise OSError('disk failure')
        with self.assertRaisesRegex(RuntimeError,'capture failed'):
            observed_action(task,np.zeros((2,6)),emit=broken,context=lambda:{})
        self.assertEqual(task.steps,10);self.assertFalse(task.fallback)
        self.assertNotIn('TOPP',vars(task.robot.left_mplib_planner))
