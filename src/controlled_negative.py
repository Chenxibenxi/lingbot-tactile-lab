"""Opt-in SIMULATION-only open-grasp perturbation for one bottle task.

Never patches RoboTwin source or real hardware commands. The original task's
approach/lift/place plan stays intact; only the two grasp-action targets are
changed from closed (0) to fully open (1). This is a negative intervention,
not an expert demonstration or a physical gripper safety specification.
"""
from __future__ import annotations


def install_open_grasp_control(robot):
    original=robot.grasp_actor
    if not callable(original):raise ValueError('missing rule-expert grasp method')
    calls=[]
    def open_grasp(*args,**kwargs):
        # Source's gripper_pos is its fifth positional parameter (actor,
        # arm_tag, pre_grasp_dis, grasp_dis, gripper_pos). Refuse ambiguity.
        if len(args)>4 or 'gripper_pos' in kwargs:
            raise ValueError('grasp source changed: unexpected explicit gripper position')
        altered=dict(kwargs,gripper_pos=1.0)
        action=original(*args,**altered)
        calls.append({'call_index':len(calls),'source_default_closed':True,
                      'controlled_open_target':1.0})
        return action
    robot.grasp_actor=open_grasp
    return calls


def check_intervention(calls,planner_ok,goal_ok):
    if len(calls)!=2 or not all(x['controlled_open_target']==1.0 for x in calls):
        raise ValueError('controlled negative did not modify exactly two grasps')
    if planner_ok is not True or goal_ok is not False:
        raise ValueError('negative interpretation requires successful planner and failed task goal')
    return True
