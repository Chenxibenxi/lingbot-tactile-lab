"""CPU-testable executor for live online-rule proposals; NOT an expert writer.

A future isolated one-seed RoboTwin runner must create same-run camera/physics,
raw force/HDF5 and validated completion separately. This module returns private
in-memory action diagnostics only and NEVER authorizes training.
"""
from __future__ import annotations
import math
from online_bottle_expert_actions import (OnlineExpertPlanError,STRIDE,EXPERIMENTAL_CPU_STRIDE,
                                           online_expert_proposals)

MAX_CALLS=128
MAX_PHYSICS_STEPS=10000  # candidate ceiling; must be explicitly preregistered before GPU


def _measured_state(task):
    try:
        left=list(task.robot.get_left_arm_real_jointState())
        right=list(task.robot.get_right_arm_real_jointState())
        if any(isinstance(x,(bool,str,bytes)) for x in left+right):
            raise OnlineExpertPlanError('measured joint state contains bool/text')
        state=[float(x) for x in left+right]
    except (TypeError,ValueError,OverflowError,AttributeError) as exc:
        raise OnlineExpertPlanError('missing current measured joint state') from exc
    if len(state)!=14 or any(not math.isfinite(x) for x in state):
        raise OnlineExpertPlanError('nonfinite measured 14D joint state')
    return state


def execute_online_expert(task, capture, *, physics_cap=MAX_PHYSICS_STEPS, stride=STRIDE):
    """Synchronously send EACH proposed qpos through the original take_action.

    No network, old run, dataset read, output file, or learner is used here.
    Caller must separately authenticate original controller, trial seed, camera,
    complete raw/obs/frame/HDF5/completion and all source SHA before labeling.
    """
    probe=getattr(task,'_force_probe',None)
    if (type(physics_cap) is not int or not 1<=physics_cap<=MAX_PHYSICS_STEPS or
            type(stride) is not int or stride not in (STRIDE,EXPERIMENTAL_CPU_STRIDE) or
            probe is None or probe.step!=0 or getattr(probe,'failed',False) or
            getattr(task,'eval_success',False) or not callable(capture) or
            not callable(getattr(task,'take_action',None))):
        raise OnlineExpertPlanError('new unapproved online-expert pilot requires pristine clock and bounded cap')
    rows=[];reason='proposals_exhausted_without_goal'
    for proposed in online_expert_proposals(task,stride=stride):
        if probe.step>=physics_cap:
            reason='bounded_physics_cap';break
        if (not isinstance(proposed,dict) or proposed.get('schema')!='lingbot_online_rule_expert_proposal_v1' or
                proposed.get('status')!='proposed_not_executed_not_training_approved' or
                proposed.get('training_approval') is not False or
                type(proposed.get('proposal_index')) is not int or proposed['proposal_index']!=len(rows) or
                proposed.get('stage') not in ('grasp','lift','place') or
                proposed.get('planning_source')!='current_scene_stage_planner_not_old_episode'):
            raise OnlineExpertPlanError('online expert proposal provenance invalid')
        command=proposed.get('command_state_14')
        if (not isinstance(command,list) or len(command)!=14 or
                any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)
                    for v in command) or
                any(not 0<=command[i]<=1 for i in (6,13))):
            raise OnlineExpertPlanError('online expert proposed 14D command invalid')
        before=probe.step
        marker=capture(task)
        if (not isinstance(marker,dict) or type(marker.get('capture_physics_step')) is not int or
                marker['capture_physics_step']!=before or
                type(marker.get('frame_index')) is not int or marker['frame_index']<0 or
                (rows and marker['frame_index']!=rows[-1]['camera_frame_index']+1)):
            raise OnlineExpertPlanError('current RGB must be captured before same-run action')
        task.take_action(command,action_type='qpos')
        after=probe.step
        if after<=before or getattr(probe,'failed',False):
            raise OnlineExpertPlanError('real controller did not advance verified physics clock')
        measured=_measured_state(task)
        rows.append({'action_index':len(rows),'phase':proposed['stage'],
                     'controller_method':'Base_Task.take_action',
                     'camera_frame_index':marker['frame_index'],
                     'capture_physics_step':before,
                     'before_physics_step':before,'after_physics_step':after,
                     'command_state_14':list(command),
                     'joint_tracking_max_abs_difference':max(abs(a-b) for a,b in zip(measured,command)),
                     'observed_success':bool(task.eval_success),
                     'source_model_role':'online_rule_expert_candidate_needs_completion_audit',
                     'training_approval':False})
        if task.eval_success:
            reason='early_goal_success';break
        if after>physics_cap:
            reason='bounded_physics_cap_exceeded_inside_action';break
        if len(rows)>=MAX_CALLS:
            reason='bounded_action_call_cap';break
    return {'schema':'lingbot_online_rule_expert_pilot_result_v1',
            'status':'diagnostic_not_h50_expert_dataset',
            'actual_take_action_calls':len(rows),'physics_steps':probe.step,
            'planning_path_stride_indices':stride,
            'stopped_reason':reason,'goal_success_reported':bool(task.eval_success),
            'candidate_h50_call_count_sufficient':len(rows)>=50,
            'source_and_completion_HDF5_verified':False,
            'training_approval':False,'private_action_rows_in_memory_only':rows}
