"""Registration must reject external/LXC or out-of-repository trial bytes."""
import json
from pathlib import Path
import tempfile
import unittest

from register_dataset import build_card


class ProvenanceTest(unittest.TestCase):
    def test_reject_lxc_and_unknown_source_before_data_validation(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            trial = repo / "data/paired/trial0"
            trial.mkdir(parents=True)
            for role, hostname in (("historic_lxc_two_rtx3090", "zjui"),
                                   ("unverified", "admin001-MS-7D90")):
                (trial / "completion.json").write_text(json.dumps({
                    "schema":"phizju_expert_episode_complete_v1", "success":True,
                    "source_machine_role":role, "source_hostname":hostname}))
                with self.assertRaisesRegex(ValueError, "non-relay provenance"):
                    build_card(trial, repo)

    def test_reject_trial_outside_relay_data_directory(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"; (repo / "data/paired").mkdir(parents=True)
            external = Path(td) / "external"; external.mkdir()
            with self.assertRaisesRegex(ValueError, "data/paired"):
                build_card(external, repo)


if __name__ == "__main__": unittest.main()
