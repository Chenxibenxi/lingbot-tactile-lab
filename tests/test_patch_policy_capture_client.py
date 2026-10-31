"""Strict anchors protect original RoboTwin and both applied-action paths."""
import hashlib
import unittest
from patch_policy_capture_client import SOURCE,patched


class PolicyCaptureClientPatchTest(unittest.TestCase):
    def test_inference_and_both_action_paths_are_wrapped_only_when_opted_in(self):
        original=SOURCE.read_bytes()
        result=patched(original.decode())
        self.assertEqual(hashlib.sha256(original).digest(),hashlib.sha256(SOURCE.read_bytes()).digest())
        self.assertEqual(result.count('TASK_ENV.take_action(act)'),original.decode().count('TASK_ENV.take_action(act)'))
        self.assertEqual(result.count('TASK_ENV.take_action(action)'),original.decode().count('TASK_ENV.take_action(action)'))
        self.assertEqual(result.count('model.infer(formatted_observation)'),1)
        self.assertIn('if os.environ.get("POLICY_SELF_ROLLOUT_CAPTURE_DIR"):',result)
        self.assertIn('TACTILE_LIVE_WINDOW=1',result)
        self.assertIn('_rollout_writer.close(observed_success=succ)',result)

    def test_changed_upstream_source_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'unsupported'):
            patched('def main(): pass')


if __name__=='__main__':unittest.main()
