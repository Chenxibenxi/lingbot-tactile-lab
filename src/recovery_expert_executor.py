"""Bounded diagnostic recovery execution; not an approved training collector."""
import math
import time
from online_bottle_expert_actions import online_expert_proposals,OnlineExpertPlanError
from online_expert_pilot_executor import _measured_state
from recovery_supervision_contract import recovery_target_mask


def execute_recovery(task,capture,*,perturb,side='left',action_cap=400,physics_cap=60000,stride=20,on_event=None):
    if (type(perturb) is not bool or side not in ('left','right')
            or type(action_cap) is not int or not 1<=action_cap<=400
            or type(physics_cap) is not int or not 1<=physics_cap<=60000
            or type(stride) is not int or stride!=20):
        raise ValueError('explicit bounded recovery configuration required')
    probe=getattr(task,'_force_probe',None)
    if probe is None or probe.step!=0 or getattr(probe,'failed',False) or getattr(task,'eval_success',False):
        raise ValueError('pristine live physics clock required')
    if not callable(capture) or not callable(getattr(task,'take_action',None)):
        raise ValueError('capture and original controller required')
    if on_event is not None and not callable(on_event):raise ValueError('journal callback must be callable')
    def emit(event):
        if on_event is not None:on_event(event)
    started=time.monotonic()
    rows=[];intervened=False;lift_seen=False;reason='proposals_exhausted_without_goal'
    proposals=iter(online_expert_proposals(task,stride=stride));segment_index=0;local_index=0

    def execute(command,role,phase):
        if (not isinstance(command,list) or len(command)!=14
                or any(type(v) not in (float,int) or not math.isfinite(v) for v in command)
                or any(not 0<=command[i]<=1 for i in (6,13))):
            raise OnlineExpertPlanError('invalid recovery command')
        before=probe.step;marker=capture(task)
        if (type(marker.get('capture_physics_step')) is not int or marker['capture_physics_step']!=before
                or type(marker.get('frame_index')) is not int or marker['frame_index']<0
                or (rows and marker['frame_index']!=rows[-1]['camera_frame_index']+1)):
            raise OnlineExpertPlanError('camera must precede same-run action')
        emit(dict(event='before_action',action_index=len(rows),before_physics_step=before,
                  capture_physics_step=before,camera_frame_index=marker['frame_index'],
                  command_state_14=list(command),source_role=role,phase=phase,training_approval=False))
        task.take_action(command,action_type='qpos')
        if type(probe.step) is not int or probe.step<=before or probe.failed:
            raise OnlineExpertPlanError('physics did not advance reliably')
        rows.append(dict(action_index=len(rows),before_physics_step=before,after_physics_step=probe.step,
                         capture_physics_step=before,camera_frame_index=marker['frame_index'],
                         source_role=role,phase=phase,segment_index=segment_index,
                         command_state_14=list(command),controller_method='Base_Task.take_action',training_approval=False))
        emit(dict(event='after_action',**rows[-1]))

    while len(rows)<action_cap and probe.step<physics_cap and not task.eval_success:
        if time.monotonic()-started>=1200:
            reason='bounded_wall_time';break
        try:proposal=next(proposals)
        except StopIteration:break
        if time.monotonic()-started>=1200:
            reason='bounded_wall_time';break
        if (proposal.get('schema')!='lingbot_online_rule_expert_proposal_v1'
                or proposal.get('training_approval') is not False
                or proposal.get('status')!='proposed_not_executed_not_training_approved'
                or type(proposal.get('proposal_index')) is not int or proposal['proposal_index']!=local_index
                or proposal.get('planning_source')!='current_scene_stage_planner_not_old_episode'
                or proposal.get('stage') not in ('grasp','lift','place')):
            raise OnlineExpertPlanError('invalid online expert provenance')
        local_index+=1
        phase=proposal['stage']
        if phase=='lift':lift_seen=True
        if perturb and not intervened and phase=='place':
            if not lift_seen:raise OnlineExpertPlanError('intervention requires completed lift proposals')
            command=_measured_state(task);command[6 if side=='left' else 13]=1.0
            execute(command,'external_intervention','fixed_open_after_lift')
            intervened=True;segment_index=1;local_index=0
            # Discard the old placement plan; recover from current scene geometry.
            proposals=iter(online_expert_proposals(task,stride=stride))
            continue
        execute(proposal['command_state_14'],'expert',phase)
    success=bool(task.eval_success and task.check_success())
    elapsed=time.monotonic()-started
    if elapsed>=1200:reason='bounded_wall_time'
    elif success:reason='task_success'
    elif len(rows)>=action_cap:reason='bounded_action_cap'
    elif probe.step>=physics_cap:reason='bounded_physics_cap'
    masks=recovery_target_mask(rows,official_success=success)
    result=dict(schema='recovery_expert_diagnostic_v1',status='diagnostic_not_training_approved',
                perturb_requested=perturb,intervention_applied=intervened,
                intervention_side=side if intervened else None,termination=reason,official_success=success,
                action_count=len(rows),physics_steps=probe.step,rows=rows,target_contract=masks,
                wall_seconds=elapsed,
                training_approval=False,same_run_source_audit_required=True)
    emit(dict(event='execution_finished',termination=reason,official_success=success,
              action_count=len(rows),physics_steps=probe.step,training_approval=False))
    return result
