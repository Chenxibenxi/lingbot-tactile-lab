import tempfile
from pathlib import Path
import unittest
from verify_feedback_confirmation_v3 import identity,verify,SEEDS,ARMS

class Admission(unittest.TestCase):
    def test_matrix_identity_includes_repeat(self):
        ids={identity(s,a,r,c) for s in SEEDS for a in ARMS for r in (0,1,2) for c in (10,1)}
        self.assertEqual(len(ids),384)
    def test_reject_development_seed_and_bad_repeat(self):
        for args in ((100127,'true',0,1),(100147,'true',True,1),(100147,'true',3,1),(100147,'true',0,5)):
            with self.assertRaises(ValueError):identity(*args)
    def test_missing_formal_manifest_denies_execution(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):verify(100147,'true',0,1,Path(d))
