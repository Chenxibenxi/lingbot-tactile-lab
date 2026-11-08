"""Shared-prefix replay plus bounded control/recovery; no dataset admission."""
import time

from online_expert_pilot_executor import _measured_state
from recovery_capture_bridge_v2 import CaptureBridge
from recovery_shared_prefix_v2 import load_prefix, replay_verified_prefix
from recovery_supervision_contract import recovery_target_mask


def live_stage_commands(task, stage):
    from online_bottle_expert_actions import _paired_actions, _commands_for_pair
    if stage not in ('grasp','lift','place') or not task.need_plan or not task.plan_success:
        raise ValueError('invalid live planning state')
    for left, right in _paired_actions(task, stage):
        yield from _commands_for_pair(task, left, right, stride=20)


def execute_pair(task, prefix_path, *, expected_sha256, expected_identity,
                 condition, emit, bridge=None, commands=None, monotonic=time.monotonic):
    if condition not in ('expert_control','left_open_once_recovery'):
        raise ValueError('unknown paired condition')
    bridge = bridge if bridge is not None else CaptureBridge(task, emit=emit)
    if bridge.receipts or bridge.read_clock() != 0 or task.eval_success:
        raise ValueError('fresh replay task required')
    commands = commands if commands is not None else lambda stage:live_stage_commands(task, stage)
    started = monotonic()
    def bounded_execute(command):
        if len(bridge.receipts) >= 400 or bridge.read_clock() >= 60000 or monotonic()-started >= 1200:
            raise ValueError('paired execution bound reached')
        bridge.execute(command)
        if bridge.read_clock() > 60000 or monotonic()-started >= 1200:
            raise ValueError('paired execution exceeded bound')
    prefix = load_prefix(prefix_path, expected_sha256=expected_sha256, expected_identity=expected_identity)
    audit = replay_verified_prefix(prefix_path, expected_sha256=expected_sha256,
        expected_identity=expected_identity, read_clock=bridge.read_clock,
        read_state=bridge.read_state, read_rgb_hashes=bridge.read_rgb_hashes,
        execute=bounded_execute, emit=emit)
    rows = [dict(receipt, action_index=i, phase=prefix['rows'][i]['phase'],
                 source_role='expert', segment_index=0) for i,receipt in enumerate(bridge.receipts)]
    intervened = False
    def perform(command, role, phase, segment):
        emit(dict(event='branch_before_action', action_index=len(rows), source_role=role,
                  phase=phase, segment_index=segment, training_approval=False))
        bounded_execute(command)
        row = dict(bridge.receipts[-1], action_index=len(rows), source_role=role,
                   phase=phase, segment_index=segment)
        rows.append(row)
        emit(dict(event='branch_after_action', **row))
    if condition == 'left_open_once_recovery':
        if task.eval_success:
            raise ValueError('task ended before registered intervention')
        command = _measured_state(task); command[6] = 1.0
        perform(command, 'external_intervention', 'fixed_open_after_lift', 1)
        intervened = True
        phases = ('grasp','lift','place')
    else:
        phases = ('place',)
    reason = 'proposals_exhausted_without_goal'
    for phase in phases:
        if task.eval_success: break
        for command in commands(phase):
            if task.eval_success: break
            perform(command, 'expert', phase, 2 if intervened else 0)
    success = bool(task.eval_success and task.check_success())
    if success: reason = 'task_success'
    result = dict(schema='recovery_paired_execution_v2', condition=condition,
                  official_success=success, termination=reason, intervention_applied=intervened,
                  prefix_audit=audit, rows=rows, action_count=len(rows), physics_steps=bridge.read_clock(),
                  wall_seconds=monotonic()-started, training_approval=False,
                  target_contract=recovery_target_mask(rows, official_success=success))
    emit(dict(event='execution_finished', official_success=success, termination=reason,
              action_count=len(rows), physics_steps=bridge.read_clock(), training_approval=False))
    return result
