"""Online dual-bottle rule-expert HIGH-LEVEL qpos proposal generator (CPU contract).

The caller owns a fresh RoboTwin task and must execute each yielded 14D command
with its original Base_Task.take_action before asking for the next proposal.
This module NEVER reads old demonstration traces, emits expert labels/HDF5,
claims completion, creates a trial, or allocates a GPU. A proposed command is
not an expert sample until a separate same-run success/provenance audit.
"""
from __future__ import annotations
import math

STRIDE=25  # executed s100022 baseline: default remains frozen
EXPERIMENTAL_CPU_STRIDE=20  # only an unexecuted future-seed design candidate
MAX_PHASE_PATH_STEPS=2000
MAX_PROPOSALS=128


class OnlineExpertPlanError(ValueError):
    """A current-scene rule plan is unavailable; do not fabricate action labels."""


def _numbers(values, count, what):
    try:
        values=list(values)
        if any(isinstance(x,(bool,str,bytes)) for x in values):
            raise OnlineExpertPlanError(f'{what} cannot contain bool/text')
        result=[float(x) for x in values]
    except OnlineExpertPlanError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise OnlineExpertPlanError(f'{what} is not a finite vector') from exc
    if len(result)!=count or any(not math.isfinite(x) for x in result):
        raise OnlineExpertPlanError(f'{what} requires {count} finite values')
    return result


def _state(robot, arm):
    # Hold each gripper-stage arm at MEASURED qpos, not the previous drive
    # target (which may be ahead when TOPP tracking lags or falls back).
    getter=getattr(robot,f'get_{arm}_arm_real_jointState',None)
    if not callable(getter):
        raise OnlineExpertPlanError(f'{arm} measured joint state API missing')
    values=_numbers(getter(),7,f'{arm} measured current state')
    if not 0<=values[6]<=1:
        raise OnlineExpertPlanError(f'{arm} gripper outside normalized 0..1 range')
    return values


def _plan(robot, action, arm):
    if action is None or action.action!='move' or action.arm_tag!=arm:
        raise OnlineExpertPlanError(f'{arm} online move action missing or mismatched')
    pose=_numbers(action.target_pose,7,f'{arm} current-stage target pose')
    constraints=action.args.get('constraint_pose')
    if constraints is not None:constraints=_numbers(constraints,6,f'{arm} constraint')
    # RoboTwin's planner reads CURRENT robot qpos; do not pass last_qpos or an
    # old episode trajectory. A phase is planned only when the iterator reaches it.
    plan=getattr(robot,f'{arm}_plan_path')(pose,constraint_pose=constraints)
    if not isinstance(plan,dict) or plan.get('status')!='Success':
        raise OnlineExpertPlanError(f'{arm} current-state path planning failed')
    path=plan.get('position')
    if path is None or not 2<=len(path)<=MAX_PHASE_PATH_STEPS:
        raise OnlineExpertPlanError(f'{arm} current-state path length invalid')
    path=[_numbers(row,6,f'{arm} planned arm qpos') for row in path]
    if max(abs(path[-1][j]-path[0][j]) for j in range(6))<1e-7:
        raise OnlineExpertPlanError(f'{arm} zero-displacement move must not pad H50')
    return path


def _paired_actions(task, stage):
    left,right='left','right'
    if stage=='grasp':
        a=task.grasp_actor(task.bottle1,arm_tag=left,pre_grasp_dis=0.08)
        b=task.grasp_actor(task.bottle2,arm_tag=right,pre_grasp_dis=0.08)
    elif stage=='lift':
        a=task.move_by_displacement(arm_tag=left,z=0.1)
        b=task.move_by_displacement(arm_tag=right,z=0.1)
    elif stage=='place':
        a=task.place_actor(task.bottle1,target_pose=task.left_target_pose,arm_tag=left,
                           functional_point_id=0,pre_dis=0.0,dis=0.0,is_open=False)
        b=task.place_actor(task.bottle2,target_pose=task.right_target_pose,arm_tag=right,
                           functional_point_id=0,pre_dis=0.0,dis=0.0,is_open=False)
    else:raise OnlineExpertPlanError('unknown online expert stage')
    if (not isinstance(a,(list,tuple)) or not isinstance(b,(list,tuple)) or
            len(a)!=2 or len(b)!=2 or a[0]!=left or b[0]!=right or
            not isinstance(a[1],list) or not isinstance(b[1],list) or
            len(a[1])!=len(b[1]) or not a[1]):
        raise OnlineExpertPlanError('asymmetric or invalid dual-arm rule stages')
    return zip(a[1],b[1])


def _commands_for_pair(task, left, right, *, stride):
    robot=task.robot
    if left.action==right.action=='move':
        lpath=_plan(robot,left,'left');rpath=_plan(robot,right,'right')
        n=max(len(lpath),len(rpath))
        ticks=list(range(stride,n,stride))
        if not ticks or ticks[-1]!=n-1:ticks.append(n-1)
        lg=_state(robot,'left')[6];rg=_state(robot,'right')[6]
        for tick in ticks:
            li=round(tick*(len(lpath)-1)/(n-1))
            ri=round(tick*(len(rpath)-1)/(n-1))
            yield lpath[li]+[lg]+rpath[ri]+[rg]
    elif left.action==right.action=='gripper':
        if left.arm_tag!='left' or right.arm_tag!='right':
            raise OnlineExpertPlanError('gripper arm side mismatch')
        l=_state(robot,'left');r=_state(robot,'right')
        lg=_numbers([left.target_gripper_pos],1,'left gripper goal')[0]
        rg=_numbers([right.target_gripper_pos],1,'right gripper goal')[0]
        if not 0<=lg<=1 or not 0<=rg<=1:
            raise OnlineExpertPlanError('gripper goal outside normalized 0..1 range')
        yield l[:6]+[lg]+r[:6]+[rg]
    else:
        raise OnlineExpertPlanError('mixed move/gripper stages are unsupported; no padding')


def online_expert_proposals(task, *, stride=STRIDE):
    """Yield live-rule 14D proposals, resuming after each caller-executed action.

    For each of grasp/lift/place, calculate its rule targets only when the stage
    begins on the ACTUAL current scene. Within one stage a geometry path is
    planned once and sampled with fixed stride; no future contact/RGB is read.
    This is NOT a run loop: the consumer must synchronously call take_action.
    """
    if type(stride) is not int or stride not in (STRIDE,EXPERIMENTAL_CPU_STRIDE):
        raise OnlineExpertPlanError('only pinned baseline or unexecuted CPU candidate stride')
    if not getattr(task,'need_plan',False) or not getattr(task,'plan_success',False):
        raise OnlineExpertPlanError('fresh task must have live rule planning enabled')
    count=0
    for stage in ('grasp','lift','place'):
        if getattr(task,'eval_success',False):return
        for left,right in _paired_actions(task,stage):
            for command in _commands_for_pair(task,left,right,stride=stride):
                if getattr(task,'eval_success',False):return
                if count>=MAX_PROPOSALS:
                    raise OnlineExpertPlanError('bounded online expert call limit')
                count+=1
                yield {'schema':'lingbot_online_rule_expert_proposal_v1',
                       'status':'proposed_not_executed_not_training_approved',
                       'stage':stage,'proposal_index':count-1,
                       'command_state_14':command,
                       'planning_source':'current_scene_stage_planner_not_old_episode',
                       'training_approval':False}
