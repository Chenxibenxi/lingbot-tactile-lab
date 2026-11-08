"""Fail-closed command replay contract; not a training or GPU admission gate."""
import hashlib
import json
import math
from pathlib import Path


def validate_prefix(document, *, expected_identity):
    if document.get('schema') != 'recovery_shared_prefix_v2':
        raise ValueError('unsupported prefix schema')
    if document.get('training_approval') is not False:
        raise ValueError('prefix cannot authorize training')
    identity = document.get('identity')
    fields = {'seed', 'asset_sha256', 'controller_sha256', 'protocol_sha256'}
    if not isinstance(identity, dict) or set(identity) != fields or identity != expected_identity:
        raise ValueError('prefix identity mismatch')
    if type(identity['seed']) is not int or identity['seed'] < 0:
        raise ValueError('invalid seed')
    for key in fields - {'seed'}:
        value = identity[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('invalid source digest')
    rows = document.get('rows')
    if not isinstance(rows, list) or not 2 <= len(rows) < 400:
        raise ValueError('bounded nonempty grasp/lift prefix required')
    clock = 0
    phase = 'grasp'
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError('invalid row')
        if type(row.get('action_index')) is not int or row['action_index'] != index:
            raise ValueError('missing or reordered action')
        if row.get('source_role') != 'expert' or row.get('training_approval') is not False:
            raise ValueError('intervention or approved label in prefix')
        current = row.get('phase')
        if current not in ('grasp', 'lift') or (phase == 'lift' and current != 'lift') or (index == 0 and current != 'grasp'):
            raise ValueError('invalid phase order')
        phase = current
        before, after = row.get('before_physics_step'), row.get('after_physics_step')
        if type(before) is not int or type(after) is not int or before != clock or not before < after <= 60000:
            raise ValueError('noncontiguous physics clock')
        clock = after
        command = row.get('command_state_14')
        if not isinstance(command, list) or len(command) != 14 or any(type(v) not in (int, float) or not math.isfinite(v) for v in command):
            raise ValueError('invalid command')
        if any(not 0 <= command[i] <= 1 for i in (6, 13)):
            raise ValueError('invalid gripper command')
    if phase != 'lift' or document.get('boundary') != 'before_first_place':
        raise ValueError('prefix must finish lift before placement')
    return document


def load_prefix(path, *, expected_sha256, expected_identity):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('sealed prefix changed')
    return validate_prefix(json.loads(raw), expected_identity=expected_identity)


def validate_rgb_hashes(value):
    if not isinstance(value, dict) or set(value) != {'head_camera','left_camera','right_camera'}:
        raise ValueError('all three camera digests required')
    for digest in value.values():
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('invalid camera digest')


def validate_references(document):
    from recovery_pair_state_v2 import compare_states
    for name, clock in (('initial',0), ('boundary',document['rows'][-1]['after_physics_step'])):
        reference = document.get(name+'_state', {})
        compare_states(reference, reference)
        if reference['physics_step'] != clock:
            raise ValueError('reference state clock mismatch')
        validate_rgb_hashes(document.get(name+'_rgb_sha256'))


def replay_prefix(document, *, expected_identity, read_clock, execute, emit):
    """Caller supplies durable journal and original controller/capture callback.

    This only verifies commands/clocks. Caller must separately compare initial
    and boundary states and validate same-run captures before branching.
    """
    # Copy after validation so caller mutation cannot alter upcoming commands.
    frozen = json.loads(json.dumps(validate_prefix(document, expected_identity=expected_identity), allow_nan=False))
    for row in frozen['rows']:
        before = read_clock()
        if type(before) is not int or before != row['before_physics_step']:
            raise ValueError('live prefix clock diverged before action')
        command = list(row['command_state_14'])
        emit(json.loads(json.dumps(dict(event='before_prefix_action', **row))))
        execute(command)
        after = read_clock()
        if type(after) is not int or after != row['after_physics_step']:
            raise ValueError('live prefix clock diverged after action')
        emit(json.loads(json.dumps(dict(event='after_prefix_action', **row))))
    return dict(status='commands_and_clocks_verified', action_count=len(frozen['rows']),
                boundary_state_audit_required=True, training_approval=False)


def replay_verified_prefix(path, *, expected_sha256, expected_identity,
                           read_clock, read_state, read_rgb_hashes, execute, emit):
    """Verify a sealed reference before allowing a caller to enter either branch.

    execute must capture same-run observations then call the original controller.
    This does not replace source/completion validation or approve training.
    """
    from recovery_pair_state_v2 import compare_states
    document = load_prefix(path, expected_sha256=expected_sha256, expected_identity=expected_identity)
    # Validate both references before any physical execution.
    validate_references(document)
    def audit(name):
        state = read_state()
        result = compare_states(document[name+'_state'], state)
        rgb = read_rgb_hashes()
        validate_rgb_hashes(rgb)
        result['rgb_equal'] = rgb == document[name+'_rgb_sha256']
        # Reading state/cameras must not advance simulation.
        live_clock = read_clock()
        result['live_clock_equal'] = type(live_clock) is int and live_clock == state['physics_step']
        emit(dict(event=name+'_state_audit', observed_state=state, observed_rgb_sha256=rgb, **result))
        if not all(result[key] for key in ('observable_state_equal','rgb_equal','live_clock_equal')):
            raise ValueError(name+' state or camera mismatch')
        return result
    initial = audit('initial')
    execution = replay_prefix(document, expected_identity=expected_identity,
                              read_clock=read_clock, execute=execute, emit=emit)
    boundary = audit('boundary')
    return dict(status='shared_prefix_verified', initial_audit=initial, boundary_audit=boundary,
                execution=execution, training_approval=False,
                same_run_source_audit_required=True)
