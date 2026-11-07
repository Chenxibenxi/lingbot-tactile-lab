"""Latest owner rule: no future tactile GPU wrapper auto-starts Strata on exit."""
from pathlib import Path
import re
import subprocess
import unittest

REPO=Path(__file__).resolve().parents[1]
RETIRED={'run_h50_policy_controller_technical_once.sh','real_v2_154800_stream_once.sh'}
AUTO_START=re.compile(r'api/strata/start|/v1/load|systemctl\s+--user\s+start\s+strata')


class NoAutomaticStrataRestartTests(unittest.TestCase):
    def test_repo_instructions_forbid_autostart_after_trial(self):
        text=(REPO/'AGENTS.md').read_text()
        self.assertIn('Never automatically start/load Strata',text)
        self.assertIn('release **only our own** GPU lease',text)

    def test_private_trial_wrappers_have_no_autostart_and_return_own_lease(self):
        paths=[REPO/'work'/name for name in (
            'run_online_expert_s100022_once.sh',
            'run_online_expert_s100023_stride20_once.sh',
            'run_online_expert_s100024_validation_once.sh',
            'run_online_expert_s100025_validation_once.sh',
            'run_h50_policy_controller_technical_once.sh',
            'run_one_relay_pilot.sh','run_two_verified_expert_seeds.sh',
            'run_eval_154800.sh','run_eval_93k.sh',
            'real_v2_154800_stream_once.sh')]
        if any(not p.is_file() for p in paths):
            self.skipTest('private ignored GPU wrappers unavailable on this checkout')
        for path in paths:
            with self.subTest(name=path.name):
                text=path.read_text()
                self.assertIsNone(AUTO_START.search(text))
                subprocess.run(['bash','-n',str(path)],check=True,capture_output=True,timeout=3)
                if path.name not in RETIRED:
                    self.assertIn('release --owner "$OWNER"',text)


if __name__=='__main__':unittest.main()
