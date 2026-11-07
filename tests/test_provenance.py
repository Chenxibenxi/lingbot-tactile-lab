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

    def test_technical_success_cannot_enter_expert_registry_even_through_alias(self):
        with tempfile.TemporaryDirectory() as td:
            repo=Path(td)/'repo';paired=repo/'data/paired';paired.mkdir(parents=True)
            pilot=repo/'data/technical_replay/same_seed_success';pilot.mkdir(parents=True)
            (pilot/'technical_result.json').write_text(json.dumps({
                'goal_check_after_run':True,'training_approval':False,
                'status':'mechanical_pilot_not_expert_dataset'}))
            with self.assertRaisesRegex(ValueError,'data/paired'):
                build_card(pilot,repo)
            alias=paired/'forged_expert';alias.symlink_to(pilot,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'data/paired'):
                build_card(alias,repo)

    def test_real_online_task_success_below_h50_cannot_register_as_expert(self):
        repo=Path(__file__).resolve().parents[1]
        trial=repo/'data/paired/relay_pickdiverse_s100022_online_h50_expert0'
        if not trial.exists():self.skipTest('single private new-seed pilot not present in this checkout')
        result=json.loads((trial/'pilot_result.json').read_text())
        self.assertTrue(result['goal_ok'])
        self.assertEqual(result['actual_take_action_calls'],46)
        self.assertFalse(result['success_candidate_not_training_approved'])
        self.assertFalse((trial/'completion.json').exists())
        self.assertFalse((trial/'demonstration/data/episode0.hdf5').exists())
        with self.assertRaises(FileNotFoundError):build_card(trial,repo)

    def test_two_failed_independent_validation_seeds_cannot_register_as_experts(self):
        repo=Path(__file__).resolve().parents[1]
        for name in ('relay_pickdiverse_s100024_online_h50_stride20_validation_expert0',
                     'relay_pickdiverse_s100025_online_h50_stride20_validation_expert0'):
            with self.subTest(trial=name):
                trial=repo/'data/paired'/name
                if not trial.is_dir():self.skipTest('private negative validation trial not present')
                self.assertTrue((trial/'failure.json').is_file())
                self.assertFalse((trial/'completion.json').exists())
                self.assertFalse((trial/'demonstration/data/episode0.hdf5').exists())
                with self.assertRaises(FileNotFoundError):build_card(trial,repo)

    def test_reject_trial_outside_relay_data_directory(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"; (repo / "data/paired").mkdir(parents=True)
            external = Path(td) / "external"; external.mkdir()
            with self.assertRaisesRegex(ValueError, "data/paired"):
                build_card(external, repo)


if __name__ == "__main__": unittest.main()
