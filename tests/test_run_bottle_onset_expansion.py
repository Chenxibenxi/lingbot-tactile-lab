"""A fixed seed batch must not cherry-pick or acquire GPU on dry-run."""
from pathlib import Path
import subprocess
import unittest

SCRIPT=Path(__file__).resolve().parents[1]/'src/run_bottle_onset_expansion.sh'


class BottleOnsetRunnerTests(unittest.TestCase):
    def test_predeclared_two_attempts_and_no_gpu_dry_run(self):
        checked=subprocess.run(['bash','-n',str(SCRIPT)],capture_output=True,text=True)
        self.assertEqual(checked.returncode,0,checked.stderr)
        dry=subprocess.run(['bash',str(SCRIPT),'--dry-run'],capture_output=True,text=True)
        self.assertEqual(dry.returncode,0,dry.stderr)
        self.assertIn('seed=100013',dry.stdout)
        self.assertIn('seed=100015',dry.stdout)
        self.assertNotIn('seed=100001',dry.stdout)
        self.assertIn('No GPU lease',dry.stdout)
        invalid=subprocess.run(['bash',str(SCRIPT),'--retry'],capture_output=True,text=True)
        self.assertEqual(invalid.returncode,2)


if __name__=='__main__':unittest.main()
