"""GPU is the eventual real trainer; CPU only inspects source approval metadata."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from plan_h50_gpu_training import (REPO,SHORT_TRIAL,VAL_REJECTION_CARD,UNSTABLE_VAL_CARD,
                                   UNSTABLE_VAL_TRIAL,inspect,verify_rejected_validation,
                                   verify_unstable_validation_setup)


class H50GpuTrainingReadinessTests(unittest.TestCase):
    def test_real_repo_reports_zero_approved_h50_without_training_or_GPU(self):
        report=inspect(REPO)
        self.assertEqual((report['approved_h50_train_windows'],report['approved_h50_validation_windows']),(0,0))
        self.assertEqual(report['short_newseed_task_success_but_calls'],46)
        self.assertTrue(report['gpu_training_is_intended_after_data_gates'])
        self.assertFalse(report['gpu_training_permitted_now'])
        self.assertFalse(report['CPU_proxy_training_run'])
        self.assertFalse(report['training_approval'])
        self.assertEqual(report['verified_unapproved_h50_train_intent_windows'],8)
        self.assertEqual(report['verified_unapproved_h50_validation_windows'],0)
        self.assertTrue(report['independent_validation_trial_failed_goal_verified'])
        self.assertTrue(report['independent_validation_trial_unstable_setup_verified'])
        self.assertEqual(len(report['source_sha256']),8)

    def test_failed_validation_source_drift_or_silent_completion_blocks_readiness(self):
        assert verify_rejected_validation(REPO)
        with patch('plan_h50_gpu_training.sha256',return_value='0'*64):
            with self.assertRaisesRegex(ValueError,'six-source SHA changed'):
                verify_rejected_validation(REPO)
        with tempfile.TemporaryDirectory() as td:
            repo=Path(td);card=repo/'datasets'/VAL_REJECTION_CARD
            card.parent.mkdir(parents=True);card.write_bytes((REPO/'datasets'/VAL_REJECTION_CARD).read_bytes())
            trial=repo/'data/paired/relay_pickdiverse_s100024_online_h50_stride20_validation_expert0'
            trial.mkdir(parents=True)
            (trial/'completion.json').write_text('{"success":true}')
            with self.assertRaisesRegex(ValueError,'silently promoted'):
                verify_rejected_validation(repo)

    def test_unstable_new_validation_cannot_be_promoted_to_expert(self):
        assert verify_unstable_validation_setup(REPO)
        with patch('plan_h50_gpu_training.sha256',return_value='0'*64):
            with self.assertRaisesRegex(ValueError,'private source SHA changed'):
                verify_unstable_validation_setup(REPO)
        with tempfile.TemporaryDirectory() as td:
            repo=Path(td);card=repo/'datasets'/UNSTABLE_VAL_CARD
            card.parent.mkdir(parents=True);card.write_bytes((REPO/'datasets'/UNSTABLE_VAL_CARD).read_bytes())
            trial=repo/'data/paired'/UNSTABLE_VAL_TRIAL;trial.mkdir(parents=True)
            (trial/'online_actions.jsonl').write_text('forged action')
            with self.assertRaisesRegex(ValueError,'falsely promoted'):
                verify_unstable_validation_setup(repo)

    def test_forged_short_success_or_candidate_norm_approval_cannot_start_training(self):
        with tempfile.TemporaryDirectory() as td:
            repo=Path(td)
            for relative in ('datasets/provenance_v1.json',
                             'datasets/relay_pickdiverse_seed_split_candidate_v2.json',
                             'configs/relay_pickdiverse_contact_norm_candidate_v3.json',
                             'datasets/relay_v2_real_visual_zero_identity_v1.json',
                             'datasets/relay_online_expert_s100022_h50_short_v1.json'):
                dest=repo/relative;dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes((REPO/relative).read_bytes())
            self.assertFalse(inspect(repo)['gpu_training_permitted_now'])
            norm=repo/'configs/relay_pickdiverse_contact_norm_candidate_v3.json'
            changed=json.loads(norm.read_text());changed['training_approval']=True
            norm.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'owner audit'):inspect(repo)
            changed['training_approval']=False;norm.write_text(json.dumps(changed))
            trial=repo/'data/paired'/SHORT_TRIAL;trial.mkdir(parents=True)
            (trial/'online_actions.jsonl').write_text('forged completed H50')
            with self.assertRaisesRegex(ValueError,'action SHA changed'):inspect(repo)
            (trial/'completion.json').write_text('{"success":true}')
            with self.assertRaisesRegex(ValueError,'silently promoted'):inspect(repo)


if __name__=='__main__':unittest.main()
