"""Fresh same-seed TECHNICAL replay of old qpos proposals via real take_action.

Never an expert training dataset: old dense waypoints are control proposals,
not labels. Real calls and new-run physics/camera markers are kept in ignored
data/technical_replay only; no source or held-out image is modified/read.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from technical_take_action_waypoints import SOURCE_ROLE

ROLE='known_seed_policy_controller_replay_technical_only'
MAX_PHYSICS_STEPS=6000
MAX_CALLS=64
CAP_8000_TRIAL='relay_knownseed100013_h50_controller_cap8000_v4'


def approved_physics_cap(output_path,physics_cap):
    """Only the user's single fresh v4 trial may exceed the old 6,000-step cap."""
    root=Path(__file__).resolve().parents[1]
    return (type(physics_cap) is int and (physics_cap==MAX_PHYSICS_STEPS or
            (physics_cap==8000 and Path(output_path).resolve(strict=False).parent==
             root/'data/technical_replay'/CAP_8000_TRIAL)))


def run_controller(task,proposal,*,output_path,capture,physics_cap=MAX_PHYSICS_STEPS):
    """Call actual task.take_action exactly once for each old *proposal*.

    The capture callback must save the NEW run's current RGB/observation marker;
    no old frame/contact/label is ever copied into this run. Returns only a
    negative-or-technical summary; cannot authorize training by construction.
    """
    root=Path(__file__).resolve().parents[1]
    output_path=Path(output_path).resolve(strict=False)
    if (root/'data/technical_replay' not in output_path.parents or
            output_path.suffix!='.jsonl' or output_path.is_symlink() or
            output_path.exists()):
        raise ValueError('new technical replay file only under ignored data/technical_replay')
    waypoints=proposal.get('private_waypoints') if isinstance(proposal,dict) else None
    if (not isinstance(waypoints,list) or len(waypoints)!=MAX_CALLS or
            proposal.get('source_role')!=SOURCE_ROLE or
            proposal.get('status')!='not_executed_not_action_labels_not_training_approved' or
            proposal.get('training_approval') is not False or
            proposal.get('actual_take_action_calls')!=0 or
            not approved_physics_cap(output_path,physics_cap) or
            not callable(capture) or not callable(getattr(task,'take_action',None)) or
            not getattr(task,'_force_probe',None)):
        raise ValueError('only source-validated 64-waypoint real-controller technical replay')
    if not output_path.parent.is_dir():
        raise ValueError('fresh caller-created technical trial root required')
    probe=task._force_probe
    if probe.step!=0 or getattr(task,'eval_success',False):
        raise ValueError('technical run must begin with new pristine physics clock')
    executed=0;stop='proposals_exhausted'
    with output_path.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps({'schema':'lingbot_policy_take_action_technical_calls_v1',
            'source_role':ROLE,'source_raw_sha256':proposal['source_raw_sha256'],
            'source_command_sha256':proposal['source_command_sha256'],
            'known_seed_replay_is_independent_seed':False,'training_approval':False})+'\n')
        for candidate in waypoints:
            if probe.step>=physics_cap:
                stop='bounded_physics_cap';break
            command=candidate.get('proposed_qpos_command_14')
            if (not isinstance(command,list) or len(command)!=14 or
                    any(not isinstance(x,(int,float)) or not math.isfinite(float(x)) for x in command) or
                    any(not 0<=float(command[k])<=1 for k in (6,13))):
                raise ValueError('private replay proposal has invalid 14D qpos')
            before=probe.step
            marker=capture(task)
            if not isinstance(marker,dict) or marker.get('capture_physics_step')!=before:
                raise ValueError('new-run camera not captured at exact action boundary')
            task.take_action(command,action_type='qpos')
            after=probe.step
            if after<=before or getattr(probe,'failed',False):
                raise RuntimeError('real controller did not advance verified physics clock')
            stream.write(json.dumps({'call_index':executed,'source_waypoint_step':candidate['source_physics_step'],
                'before_physics_step':before,'after_physics_step':after,
                'controller_method':'Base_Task.take_action','camera_frame_index':marker.get('frame_index'),
                'command_state_14':command,'observed_success':bool(getattr(task,'eval_success',False))},
                allow_nan=False)+'\n')
            executed+=1
            if task.eval_success:
                stop='early_goal_success';break
            if after>physics_cap:
                stop='bounded_physics_cap_exceeded_inside_action';break
    return {'schema':'lingbot_policy_take_action_technical_replay_result_v1',
            'status':'mechanical_pilot_not_expert_dataset',
            'source_role':ROLE,'proposal_count':len(waypoints),
            'actual_take_action_calls':executed,'physics_steps':probe.step,
            'goal_success_reported':bool(task.eval_success),'stopped_reason':stop,
            'eligible_h50_windows_verified':0,
            'training_approval':False,'independent_expert_seed':False,
            'hdf5_expert_completion_written':False}
