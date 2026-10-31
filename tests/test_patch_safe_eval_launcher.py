"""CPU-only regression: a private eval launcher cannot rewrite source assets."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from patch_safe_eval_launcher import SOURCE,patched


class IsolatedEvalTest(unittest.TestCase):
    def test_refuse_stale_anchors_and_remove_mutating_block(self):
        source=SOURCE.read_bytes(); result=patched(source.decode())
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).digest(),hashlib.sha256(source).digest())
        self.assertNotIn('update_embodiment_config_path.py',result)
        self.assertNotIn('> "$_curobo_pth"',result)
        self.assertIn('UNSAFE curobo .pth differs; abort without writes',result)
        self.assertEqual(source.decode().count('RELAY_TASK_ONLY'),result.count('RELAY_TASK_ONLY'))
        with self.assertRaisesRegex(ValueError,'unexpected launcher revision'):
            patched('echo not our eval launcher')


if __name__=='__main__':unittest.main()
