"""CPU fail-closed H50 label gate for FUTURE actual expert take_action calls.

The existing rule expert calls move/take_dense_action, not take_action: its
per-physics-step drive targets must NEVER be accepted as these 50 labels.
This module is only a candidate validator, not instrumentation, successful
expert data, a data approval, or permission to train LingBot-VLA.
"""
from __future__ import annotations
import math
import re

HORIZON=50
METHOD='Base_Task.take_action'
ROLE='relay_expert_policy_control_actual_calls'
SOURCE_SHA='448f7152b65cb9102217cf5463aa821d72810ca56f63d5a797ec7bd43e23e101'


def _sha(value):
    return type(value) is str and re.fullmatch(r'[a-f0-9]{64}',value) is not None


def select_policy_h50(actions, anchor, *, expected_trial, expected_completion_sha,
                       source_physics_steps, fold):
    """Return 50 real high-level commands and variable applied physics spans.

    The caller MUST first independently validate SHA-bound expert completion,
    same-trial HDF5/current RGB, original method provenance and tactile causality.
    This second gate demands that the observed actions were actually delivered
    through the original policy controller, not re-sampled expert drive ticks.
    """
    if (fold not in ('train','validation') or not isinstance(expected_trial,str) or
            not expected_trial or not _sha(expected_completion_sha) or
            type(source_physics_steps) is not int or source_physics_steps<=0 or
            not isinstance(actions,(list,tuple)) or not isinstance(anchor,dict)):
        raise ValueError('unapproved split, source identity or malformed H50 inputs')
    if (anchor.get('trial_id')!=expected_trial or
            anchor.get('completion_sha256')!=expected_completion_sha or
            anchor.get('controller_method')!=METHOD or
            anchor.get('source_model_role')!=ROLE or
            anchor.get('original_base_task_sha256')!=SOURCE_SHA or
            not _sha(anchor.get('current_rgb_hdf5_sha256')) or
            anchor.get('tactile_window_valid') is not True or
            type(anchor.get('capture_physics_step')) is not int or
            anchor['capture_physics_step']<25 or
            type(anchor.get('past_tactile_max_physics_step')) is not int or
            anchor['past_tactile_max_physics_step']>anchor['capture_physics_step'] or
            type(anchor.get('first_action_index')) is not int):
        raise ValueError('camera/contact/action clock or controller source not bound')
    first=anchor['first_action_index']
    if first<0 or first+HORIZON>len(actions):
        raise ValueError('fewer than 50 future real policy controller actions')
    # An actual policy request at an image capture must immediately precede its
    # first applied action. Verify entire source stream, not just the chosen 50.
    previous=None
    for i,row in enumerate(actions):
        if not isinstance(row,dict):raise ValueError('invalid high-level action row')
        before,after=row.get('before_physics_step'),row.get('after_physics_step')
        command=row.get('command_state_14')
        if (row.get('action_index')!=i or row.get('trial_id')!=expected_trial or
                row.get('completion_sha256')!=expected_completion_sha or
                row.get('controller_method')!=METHOD or
                row.get('source_model_role')!=ROLE or
                row.get('original_base_task_sha256')!=SOURCE_SHA or
                type(before) is not int or type(after) is not int or
                before<0 or after<=before or after>source_physics_steps or
                (previous is not None and before!=previous) or
                not isinstance(command,(list,tuple)) or len(command)!=14 or
                any(not isinstance(x,(int,float)) or not math.isfinite(float(x)) for x in command) or
                any(not 0<=float(command[j])<=1 for j in (6,13))):
            raise ValueError(f'expert take_action applied interval/14D source invalid at {i}')
        previous=after
    if actions[first]['before_physics_step']!=anchor['capture_physics_step']:
        raise ValueError('expert image not taken at first executed action boundary')
    selected=actions[first:first+HORIZON]
    return {'schema':'lingbot_expert_policy_take_action_h50_candidate_v1',
            'status':'source_checked_not_training_approved',
            'trial_id':expected_trial,'fold':fold,
            'completion_sha256':expected_completion_sha,
            'current_rgb_hdf5_sha256':anchor['current_rgb_hdf5_sha256'],
            'anchor_physics_step':anchor['capture_physics_step'],
            'past_tactile_max_physics_step':anchor['past_tactile_max_physics_step'],
            'controller_method':METHOD,'original_base_task_sha256':SOURCE_SHA,
            'target_action_count':HORIZON,'target_semantics':'actual_take_action_calls_not_resampled_drive_ticks',
            'targets':[{'action_index':row['action_index'],
                        'before_physics_step':row['before_physics_step'],
                        'after_physics_step':row['after_physics_step'],
                        'command_state_14':list(row['command_state_14'])} for row in selected],
            'training_approval':False,'test_fold_read':False}
