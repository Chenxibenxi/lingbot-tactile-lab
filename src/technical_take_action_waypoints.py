"""Private TECHNICAL-only replay candidates from one previously verified rule trace.

These post-step drive targets are NOT expert take_action labels. Replaying them
through real Base_Task.take_action in a FRESH isolated run might fail: only
actually executed calls, new same-run images/contact and task success could
later enter the separate H50 candidate gate. Never register these as training.
"""
from __future__ import annotations
import math
import re

SOURCE_ROLE='same_seed_technical_replay_only_not_independent_training'
STRIDE=25


def propose_waypoints(command_rows, *, raw_sha256, command_sha256, stride=STRIDE):
    if (type(stride) is not int or stride!=STRIDE or
            not isinstance(command_rows,(list,tuple)) or len(command_rows)<STRIDE or
            any(not isinstance(x,str) or re.fullmatch(r'[a-f0-9]{64}',x) is None
                for x in (raw_sha256,command_sha256))):
        raise ValueError('verified technical command sidecar and fixed stride required')
    previous_action=None
    for step,row in enumerate(command_rows,1):
        if not isinstance(row,dict):raise ValueError('malformed expert drive-target row')
        cmd=row.get('drive_target_state_14');action=row.get('action_index')
        if (row.get('physics_step')!=step or row.get('motion_phase') not in
                ('expert_dense','expert_together') or type(action) is not int or
                action<1 or (previous_action is not None and action<previous_action) or
                not isinstance(cmd,list) or len(cmd)!=14 or
                any(not isinstance(v,(int,float)) or not math.isfinite(float(v)) for v in cmd) or
                any(not 0<=float(cmd[i])<=1 for i in (6,13))):
            raise ValueError('noncontiguous or invalid rule expert drive-target source')
        previous_action=action
    selected=list(range(STRIDE,len(command_rows)+1,STRIDE))
    if selected[-1]!=len(command_rows):selected.append(len(command_rows))
    return {'schema':'lingbot_policy_controller_technical_waypoint_proposal_v1',
            'status':'not_executed_not_action_labels_not_training_approved',
            'source_role':SOURCE_ROLE,'source_raw_sha256':raw_sha256,
            'source_command_sha256':command_sha256,
            'recorded_physics_steps':len(command_rows),'fixed_proposal_stride_ticks':STRIDE,
            'proposed_waypoint_count':len(selected),'actual_take_action_calls':0,
            'private_waypoints':[{'source_physics_step':step,
                                  'proposed_qpos_command_14':list(command_rows[step-1]['drive_target_state_14'])}
                                 for step in selected],
            'training_approval':False,'known_seed_replay_is_independent_seed':False}


def sanitized_card(proposal):
    if (proposal.get('status')!='not_executed_not_action_labels_not_training_approved' or
            proposal.get('actual_take_action_calls')!=0 or
            proposal.get('training_approval') is not False):
        raise ValueError('proposal cannot be published as successful expert actions')
    return {key:value for key,value in proposal.items() if key!='private_waypoints'}
