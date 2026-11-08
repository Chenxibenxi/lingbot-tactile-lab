"""Adapter to original simulator APIs; no task setup, GPU claim, or launch."""
import hashlib
import json
import math
from pathlib import Path

from recovery_pair_state_v2 import capture_state

CAMERAS = ('head_camera', 'left_camera', 'right_camera')


def rgb_digest(array):
    import numpy as np
    value = np.asarray(array)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) <= 0:
        raise ValueError('expected nonempty uint8 RGB image')
    header = json.dumps(dict(shape=list(value.shape), dtype=str(value.dtype)), sort_keys=True).encode()
    return hashlib.sha256(header+b'\n'+value.tobytes(order='C')).hexdigest()


class CaptureBridge:
    def __init__(self, task, *, emit, image_dir=None):
        self.task = task
        self.emit = emit
        if not callable(emit) or getattr(task, '_force_probe', None) is None:
            raise ValueError('live force probe and durable journal required')
        self.receipts = []
        self.image_dir = Path(image_dir) if image_dir is not None else None
        self.snapshot_count = 0

    def read_clock(self):
        probe = self.task._force_probe
        if probe.failed or type(probe.step) is not int or probe.step < 0:
            raise ValueError('invalid live force clock')
        return probe.step

    def read_state(self):
        before = self.read_clock()
        state = capture_state(self.task)
        if self.read_clock() != before or state['physics_step'] != before:
            raise ValueError('state capture advanced physics')
        return state

    def read_rgb_hashes(self):
        # Do not call get_obs: it would append an extra force observation.
        before = self.read_clock(); frame = self.task.FRAME_IDX
        self.task._update_render()
        self.task.cameras.update_picture()
        rgb = self.task.cameras.get_rgb()
        result = {name:rgb_digest(rgb[name]['rgb']) for name in CAMERAS}
        if self.read_clock() != before or self.task.FRAME_IDX != frame:
            raise ValueError('boundary camera read advanced physics or saved frames')
        if self.image_dir is not None:
            from PIL import Image
            target = self.image_dir/str(self.snapshot_count)
            target.mkdir(parents=True, exist_ok=False)
            for name in CAMERAS:
                Image.fromarray(rgb[name]['rgb']).save(target/(name+'.png'))
            metadata = dict(physics_step=before, saved_frame_count=frame, rgb_sha256=result,
                            training_approval=False)
            with (target/'snapshot.json').open('x') as stream:
                json.dump(metadata, stream, indent=2)
            self.snapshot_count += 1
        return result

    def execute(self, command):
        if (not isinstance(command, list) or len(command) != 14
                or any(type(v) not in (int,float) or not math.isfinite(v) for v in command)
                or any(not 0 <= command[i] <= 1 for i in (6,13))):
            raise ValueError('invalid command')
        task = self.task
        before = self.read_clock(); frame = task.FRAME_IDX
        if type(frame) is not int or frame < 0 or (self.receipts and frame != self.receipts[-1]['camera_frame_index']+1):
            raise ValueError('noncontiguous saved camera index')
        task._take_picture()
        if self.read_clock() != before or task.FRAME_IDX != frame+1:
            raise ValueError('same-run camera capture failed')
        receipt = dict(camera_frame_index=frame, capture_physics_step=before,
                       before_physics_step=before, command_state_14=list(command),
                       controller_method='Base_Task.take_action', training_approval=False)
        self.emit(dict(event='captured_before_controller', **receipt))
        task.take_action(list(command), action_type='qpos')
        after = self.read_clock()
        if after <= before:
            raise ValueError('controller did not advance physics')
        receipt['after_physics_step'] = after
        self.receipts.append(receipt)
        self.emit(dict(event='captured_controller_completed', **receipt))
