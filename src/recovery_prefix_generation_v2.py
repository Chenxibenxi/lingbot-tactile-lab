"""Generate grasp/lift once; never invoke placement to find its boundary."""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

from recovery_shared_prefix_v2 import validate_prefix, validate_references, validate_rgb_hashes
from recovery_pair_state_v2 import compare_states


def stage_commands(task, stage):
    from online_bottle_expert_actions import _paired_actions, _commands_for_pair
    if stage not in ('grasp', 'lift'):
        raise ValueError('prefix cannot plan placement')
    if not task.need_plan or not task.plan_success:
        raise ValueError('live planning unavailable')
    for left, right in _paired_actions(task, stage):
        yield from _commands_for_pair(task, left, right, stride=20)


def generate_prefix(*, identity, commands, read_clock, read_state, read_rgb_hashes,
                    execute, emit, monotonic=time.monotonic):
    started = monotonic()
    initial = read_state(); initial_rgb = read_rgb_hashes()
    compare_states(initial, initial)
    validate_rgb_hashes(initial_rgb)
    if read_clock() != 0 or initial['physics_step'] != 0:
        raise ValueError('pristine initial clock required')
    rows = []
    for phase in ('grasp', 'lift'):
        count = 0
        for command in commands(phase):
            if len(rows) >= 128 or monotonic()-started >= 1200 or read_clock() >= 60000:
                raise ValueError('prefix generation bound reached')
            if (not isinstance(command, list) or len(command) != 14
                    or any(type(v) not in (int,float) or not math.isfinite(v) for v in command)
                    or any(not 0 <= command[i] <= 1 for i in (6,13))):
                raise ValueError('invalid planned command')
            before = read_clock()
            if type(before) is not int or before != (rows[-1]['after_physics_step'] if rows else 0):
                raise ValueError('generation clock changed outside controller')
            row = dict(action_index=len(rows), phase=phase, source_role='expert',
                       command_state_14=list(command), before_physics_step=before,
                       training_approval=False)
            emit(json.loads(json.dumps(dict(event='before_prefix_action', **row))))
            execute(list(command))
            after = read_clock()
            if type(after) is not int or not before < after <= 60000:
                raise ValueError('invalid executed clock')
            row['after_physics_step'] = after
            rows.append(row); count += 1
            emit(json.loads(json.dumps(dict(event='after_prefix_action', **row))))
        if count == 0:
            raise ValueError('empty grasp/lift stage')
    boundary = read_state(); boundary_rgb = read_rgb_hashes()
    compare_states(boundary, boundary)
    if boundary['physics_step'] != rows[-1]['after_physics_step'] or read_clock() != boundary['physics_step']:
        raise ValueError('boundary capture advanced physics')
    if monotonic()-started >= 1200:
        raise ValueError('prefix generation wall bound reached')
    document = dict(schema='recovery_shared_prefix_v2', identity=identity, rows=rows,
                    initial_state=initial, boundary_state=boundary,
                    initial_rgb_sha256=initial_rgb, boundary_rgb_sha256=boundary_rgb,
                    boundary='before_first_place', training_approval=False)
    validate_prefix(document, expected_identity=identity)
    validate_references(document)
    return document


def seal_prefix(path, document, *, expected_identity):
    """Publish complete bytes exclusively; an interrupted temp file is not usable."""
    validate_prefix(document, expected_identity=expected_identity)
    validate_references(document)
    raw = json.dumps(document, ensure_ascii=False, allow_nan=False, sort_keys=True).encode()
    target = Path(path)
    fd, scratch = tempfile.mkstemp(prefix='.unsealed-prefix-', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.link(scratch, target)  # Fails rather than replacing an existing seal.
        directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        os.unlink(scratch)
    return dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), training_approval=False)
