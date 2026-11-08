from types import SimpleNamespace
import unittest
import numpy as np
from recovery_capture_bridge_v2 import CaptureBridge, CAMERAS, rgb_digest


class Task:
    def __init__(self):
        self._force_probe = SimpleNamespace(step=0, failed=False)
        self.FRAME_IDX = 0
        self.cameras = SimpleNamespace(update_picture=lambda:None,
            get_rgb=lambda:{name:{'rgb':np.zeros((2,3,3),dtype=np.uint8)} for name in CAMERAS})
        self.calls = []
    def _update_render(self): pass
    def _take_picture(self): self.calls.append('capture'); self.FRAME_IDX += 1
    def take_action(self, command, action_type):
        self.calls.append('controller'); self._force_probe.step += 10


class BridgeTest(unittest.TestCase):
    def test_camera_read_has_no_saved_frame_or_controller(self):
        task = Task(); bridge = CaptureBridge(task, emit=lambda event:None)
        self.assertEqual(set(bridge.read_rgb_hashes()), set(CAMERAS))
        self.assertEqual(task.calls, [])
        self.assertEqual(task.FRAME_IDX, 0)

    def test_capture_precedes_controller(self):
        task = Task(); events = []; bridge = CaptureBridge(task, emit=events.append)
        bridge.execute([0.]*14)
        self.assertEqual(task.calls, ['capture','controller'])
        self.assertEqual(bridge.receipts[0]['capture_physics_step'], 0)
        self.assertEqual(bridge.receipts[0]['after_physics_step'], 10)

    def test_failed_capture_prevents_controller(self):
        task = Task(); task._take_picture = lambda:None
        bridge = CaptureBridge(task, emit=lambda event:None)
        with self.assertRaises(ValueError): bridge.execute([0.]*14)
        self.assertEqual(task.calls, [])
        self.assertEqual(bridge.receipts, [])

    def test_render_advancing_clock_rejected(self):
        task = Task()
        task._update_render = lambda:setattr(task._force_probe, 'step', 1)
        with self.assertRaises(ValueError): CaptureBridge(task, emit=lambda event:None).read_rgb_hashes()

    def test_image_schema_and_pixel_identity(self):
        array = np.zeros((2,3,3),dtype=np.uint8)
        first = rgb_digest(array); array[0,0,0] = 1
        self.assertNotEqual(first, rgb_digest(array))
        with self.assertRaises(ValueError): rgb_digest(array.astype(float))
