import hashlib
from pathlib import Path
import tempfile
import unittest

from patch_policy_client import patch_client

SOURCE=Path('/mnt/relaydata/lingbot/RoboTwin/script/eval_policy_client_lingbotvla.py.orig')


class PolicyClientPatchTest(unittest.TestCase):
    def test_both_references_untouched_and_requires_server_capability(self):
        before=SOURCE.read_bytes()
        patched=patch_client(before.decode())
        self.assertEqual(hashlib.sha256(before).digest(),hashlib.sha256(SOURCE.read_bytes()).digest())
        self.assertEqual(patched.count('model.infer('),before.decode().count('model.infer('))
        self.assertIn('require_tactile_capability(model.get_server_metadata())',patched)
        self.assertLess(patched.index('require_tactile_capability(model.get_server_metadata())'),
                        patched.index('ret = model.infer(dict(reset'))
        self.assertIn('probe.record_policy_request()',patched)
        self.assertLess(patched.index('probe.record_policy_request()'),
                        patched.index('ret = model.infer(formatted_observation)'))
        self.assertEqual(patched.count('if TASK_ENV.eval_success:'),before.decode().count('if TASK_ENV.eval_success:'))

    def test_unknown_upstream_client_refused(self):
        with self.assertRaisesRegex(ValueError,'unsupported'):
            patch_client('def eval_policy():\n    pass\n')


if __name__=='__main__':unittest.main()
