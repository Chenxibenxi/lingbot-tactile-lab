"""Only reserve an unused seed; cannot silently promote technical success to H50."""
import json
from pathlib import Path
import tempfile
import unittest
from preregister_online_expert_trial import REPO,SEED,SOURCE,TRIAL,preregister


class IndependentOnlineExpertPreregTests(unittest.TestCase):
    def test_local_new_seed_is_read_only_and_not_gpu_training_approved(self):
        trial=REPO/'data/paired'/TRIAL
        if trial.exists():
            with self.assertRaisesRegex(ValueError,'already attempted'):
                preregister()
            result=json.loads((trial/'pilot_result.json').read_text())
            self.assertEqual(result['actual_take_action_calls'],46)
            self.assertTrue(result['goal_ok'])
            self.assertFalse(result['success_candidate_not_training_approved'])
            self.assertFalse(result['training_approval'])
        else:
            result=preregister()
            self.assertEqual((result['seed'],result['required_consecutive_calls']),(SEED,50))
            self.assertTrue(result['online_generator_cpu_contract_tested'])
            self.assertFalse(result['online_generator_implemented_and_validated'])
            self.assertFalse(result['gpu_run_allowed_now'])
            self.assertFalse(result['training_approval'])
        with self.assertRaisesRegex(ValueError,'seed'):
            preregister(seed=100013)

    def test_collision_and_source_or_approval_change_refuse_before_gpu(self):
        with tempfile.TemporaryDirectory() as td:
            repo=Path(td);(repo/'datasets').mkdir();(repo/'data/paired').mkdir(parents=True)
            for name in ('relay_pickdiverse_seed_split_candidate_v2.json','provenance_v1.json'):
                (repo/'datasets'/name).write_bytes((REPO/'datasets'/name).read_bytes())
            self.assertEqual(preregister(repo)['seed'],SEED)
            (repo/'data/paired'/f'relay_pickdiverse_s{SEED}_old_attempt.status').touch()
            with self.assertRaisesRegex(ValueError,'already attempted'):
                preregister(repo)
            (repo/'data/paired'/f'relay_pickdiverse_s{SEED}_old_attempt.status').unlink()
            split=repo/'datasets/relay_pickdiverse_seed_split_candidate_v2.json'
            changed=json.loads(split.read_text());changed['task_seed_groups']['pick_diverse_bottles']['test'].append(SEED)
            split.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'overlaps'):
                preregister(repo)
            with self.assertRaisesRegex(ValueError,'SHA changed'):
                bad=repo/'wrong_base.py';bad.write_text('not the original controller')
                preregister(repo,source=bad)
            with self.assertRaisesRegex(ValueError,'path-planner source SHA changed'):
                preregister(repo,task_source=bad)
            with self.assertRaisesRegex(ValueError,'path-planner source SHA changed'):
                preregister(repo,robot_source=bad)


if __name__=='__main__':unittest.main()
