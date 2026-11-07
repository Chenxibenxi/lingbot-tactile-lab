"""Opt-in private per-physics-step drive-target sidecar for FUTURE relay experts.

This is the robot's drive target *after* the scene step, not measured qpos,
force, a policy infer marker, or a retrospectively reconstructed old episode.
The old force/raw/frames/HDF5 streams and policy input remain unchanged.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from validate_expert_data import read_jsonl, sha256

SCHEMA = 'lingbot_expert_command_timeline_v1'
ORIGIN = 'first_instrumented_expert_step_after_setup_stability'


def _paths(raw, commands_dir):
    raw, directory = Path(raw), Path(commands_dir)
    if (not raw.is_absolute() or raw.is_symlink() or raw.suffix != '.jsonl' or
            not re.fullmatch(r'[A-Za-z0-9_]+', raw.stem) or raw.parent.name != 'force' or
            not directory.is_absolute() or directory.name != 'commands' or directory.is_symlink() or
            raw.parent.resolve(strict=False).parent != directory.resolve(strict=False).parent):
        raise ValueError('command sidecar must be an absolute trial/commands sibling of trial/force')
    return raw, directory / f'{raw.stem}.commands.jsonl'


class ExpertCommandTimeline:
    """Write exactly one post-scene-step 14D drive target per physical tick."""

    def __init__(self, raw, commands_dir, *, task, seed, episode, dt_s, clock_origin):
        raw, path = _paths(raw, commands_dir)
        if (clock_origin != ORIGIN or not math.isfinite(dt_s) or dt_s <= 0 or
                not isinstance(seed, int) or not isinstance(episode, int) or episode < 0 or
                not isinstance(task, str) or not task):
            raise ValueError('unverified expert command clock/identity')
        path.parent.mkdir(parents=False, exist_ok=True)
        self.path = path
        self._file = path.open('x', encoding='utf-8')
        self.step = 0
        self._file.write(json.dumps({'schema': SCHEMA, 'probe_file': raw.name,
            'task': task, 'seed': seed, 'episode': episode, 'dt_s': float(dt_s),
            'clock_origin': clock_origin,
            'meaning': 'post-scene-step 14D robot drive target, not qpos, applied force, or a policy request'}) + '\n')

    def write_after_step(self, physics_step, robot, *, action_index, motion_index, motion_phase):
        if self._file is None or type(physics_step) is not int or physics_step != self.step + 1:
            raise ValueError('closed/noncontiguous expert command timeline')
        if (type(action_index) is not int or type(motion_index) is not int or motion_index < 1 or
                motion_phase not in ('expert_dense', 'expert_together')):
            raise ValueError('command row must be from an expert physics motion')
        state = [*robot.get_left_arm_jointState(), *robot.get_right_arm_jointState()]
        if (len(state) != 14 or any(not math.isfinite(float(x)) for x in state) or
                not 0 <= float(state[6]) <= 1 or not 0 <= float(state[13]) <= 1):
            raise ValueError('invalid expert 14D drive target/gripper range')
        row = {'physics_step': physics_step, 'action_index': action_index,
               'motion_index': motion_index, 'motion_phase': motion_phase,
               'drive_target_state_14': [float(x) for x in state]}
        self._file.write(json.dumps(row, allow_nan=False) + '\n')
        self.step = physics_step
        if self.step % 250 == 0:
            self._file.flush()

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


def validate_commands(raw, frames, commands):
    """Validate *new* sidecar against same-trial raw and saved-frame markers.

    To accept for training, caller must additionally validate the original
    completion/HDF5/success gate and bind this returned commands SHA to it.
    """
    raw, frames, commands = map(Path, (raw, frames, commands))
    expected_raw, expected_path = _paths(raw, commands.parent)
    if (commands != expected_path or commands.is_symlink() or
            frames != raw.with_suffix('.frames.jsonl') or frames.is_symlink()):
        raise ValueError('command/raw/frame sidecars do not share one trial')
    rr, ff, cc = (read_jsonl(path) for path in (raw, frames, commands))
    if min(len(rr), len(ff), len(cc)) < 2:
        raise ValueError('empty expert command/physics/frame stream')
    rh, fh, ch = rr[0], ff[0], cc[0]
    if (rh.get('schema') != 'phizju_force_probe_v1' or
            fh.get('schema') != 'phizju_saved_frames_v1' or ch.get('schema') != SCHEMA or
            fh.get('probe_file') != raw.name or ch.get('probe_file') != raw.name or
            any(rh.get(key) != h.get(key) for h in (fh, ch) for key in ('task','seed','episode')) or
            rh.get('clock_origin') != ORIGIN or fh.get('clock_origin') != ORIGIN or
            ch.get('clock_origin') != ORIGIN or not math.isfinite(float(rh['dt_s'])) or
            any(not math.isclose(float(h['dt_s']), float(rh['dt_s']), abs_tol=1e-7)
                for h in (fh, ch))):
        raise ValueError('command header not bound to original expert physics clock')
    if len(cc) != len(rr):
        raise ValueError('per-step command count differs from physics stream')
    for step, (force, target) in enumerate(zip(rr[1:], cc[1:]), start=1):
        command = target.get('drive_target_state_14')
        if (force.get('physics_step') != step or target.get('physics_step') != step or
                target.get('action_index') != force.get('action_index') or
                target.get('motion_index') != force.get('motion_index') or
                target.get('motion_phase') != force.get('motion_phase') or
                not isinstance(command, list) or len(command) != 14 or
                any(not math.isfinite(float(x)) for x in command) or
                not 0 <= float(command[6]) <= 1 or not 0 <= float(command[13]) <= 1):
            raise ValueError(f'command/physics source mismatch at step {step}')
    previous = -1
    for i, frame in enumerate(ff[1:]):
        step = frame.get('physics_step')
        if (frame.get('frame_index') != i or type(step) is not int or step < previous or
                step > len(rr)-1):
            raise ValueError('saved expert frame outside verified physical clock')
        previous = step
        if step == 0:
            continue  # initial camera frame before first instrumented expert command
        saved = frame.get('command_state_14')
        target = cc[step].get('drive_target_state_14')
        if (not isinstance(saved, list) or len(saved) != 14 or
                any(not math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-5)
                        for a,b in zip(saved, target))):
            raise ValueError(f'saved frame does not match same-step command at {step}')
    return {'schema':'lingbot_expert_command_timeline_audit_v1',
            'status':'source_checked_not_training_approved',
            'physics_steps':len(rr)-1,'saved_frames':len(ff)-1,
            'initial_frames_without_instrumented_command':sum(f['physics_step']==0 for f in ff[1:]),
            'source_sha256':{name:sha256(path) for name,path in
                             (('raw',raw),('frames',frames),('commands',commands))},
            'meaning':'post-scene-step drive targets, not actual joint qpos or permission to train'}
