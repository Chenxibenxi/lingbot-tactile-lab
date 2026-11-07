"""Two preregistered unique command-sidecar expert seeds; no GPU on dry-run."""
from pathlib import Path
import subprocess
import unittest

SCRIPT=Path(__file__).resolve().parents[1]/'src/run_command_timeline_pilot.sh'


class CommandTimelinePilotRunnerTests(unittest.TestCase):
    def test_one_shot_two_seed_dry_run_does_not_acquire_gpu(self):
        syntax=subprocess.run(['bash','-n',str(SCRIPT)],capture_output=True,text=True)
        self.assertEqual(syntax.returncode,0,syntax.stderr)
        dry=subprocess.run(['bash',str(SCRIPT),'--dry-run'],capture_output=True,text=True)
        self.assertEqual(dry.returncode,0,dry.stderr)
        self.assertIn('seed=100017',dry.stdout)
        self.assertIn('seed=100021',dry.stdout)
        self.assertEqual(dry.stdout.count('DRY-RUN host='),2)
        self.assertIn('DRY_RUN_NO_GPU_LEASE_NO_OUTPUT',dry.stdout)
        self.assertIn('relay_pickdiverse_s100021_commandpilot_expert0/commands',dry.stdout)
        refused=subprocess.run(['bash',str(SCRIPT),'--retry'],capture_output=True,text=True)
        self.assertEqual(refused.returncode,2)


if __name__=='__main__':unittest.main()
