"""CPU-only synthetic checks for isolated bottle-v2 exploratory proxy."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import train_tactile_proxy_pilot as pilot


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ProxyPilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        for folder in ('datasets/cards', 'configs', 'data/reports', 'data/paired', 'models'):
            (self.repo/folder).mkdir(parents=True)
        ds = self.repo/'datasets'
        self.trials = {f'relay_pickdiverse_s{seed}_expert0': seed for seed in
                       (*pilot.TRAIN_SEEDS[:3], pilot.VALID_SEED, pilot.TEST_SEED)}
        self.trials.update({f'relay_pickdiverse_s{seed}_onsetpilot_expert0': seed for seed in
                            pilot.TRAIN_SEEDS[3:]})
        train = [trial for trial, seed in self.trials.items() if seed in pilot.TRAIN_SEEDS]
        valid = [trial for trial, seed in self.trials.items() if seed == pilot.VALID_SEED]
        test = [trial for trial, seed in self.trials.items() if seed == pilot.TEST_SEED]
        for trial, seed in self.trials.items():
            (ds/'cards'/f'{trial}.json').write_text(json.dumps({
                'trial_id': trial, 'task': pilot.TASK, 'seed': seed}))
        (ds/'provenance_v1.json').write_text(json.dumps({
            'schema': 'lingbot_tactile_dataset_registry_v1', 'approved_paired_episodes': []}))
        old = {'status': 'candidate_not_approved', 'train': train[:3], 'validation': valid,
               'test': test, 'expert_card_sha256': {
                   trial: digest(ds/'cards'/f'{trial}.json') for trial in self.trials}}
        old_path = ds/'relay_pickdiverse_seed_split_candidate_v1.json'
        old_path.write_text(json.dumps(old))
        new = dict(old)
        new.update({'train': train, 'parent_v1_split_sha256': digest(old_path),
                    'new_unassigned_parent_experts': train[3:], 'evaluation_claims_approved': False,
                    'task_seed_groups': {pilot.TASK: {
                        'train': list(pilot.TRAIN_SEEDS), 'validation': [pilot.VALID_SEED],
                        'test': [pilot.TEST_SEED]}}})
        self.new_path = ds/'relay_pickdiverse_seed_split_candidate_v2.json'
        self.new_path.write_text(json.dumps(new))
        norm = self.repo/'configs/relay_pickdiverse_train_norm_candidate_v2.json'
        norm.write_text(json.dumps({'training_approval': False, 'source_trials': train,
                                   'split_candidate_sha256': hashlib.sha256(
                                       json.dumps(new, sort_keys=True).encode()).hexdigest()}))
        private = self.repo/'data/reports/relay_paired_training_index_candidate_v2.jsonl'
        self.rows = []
        for fold, trials, count in (('train', train, 118), ('validation', valid, 126)):
            for trial in trials:
                for frame in range(count):
                    self.rows.append({'schema': 'lingbot_private_relay_pair_index_v1',
                        'task': pilot.TASK, 'fold': fold, 'trial_id': trial,
                        'seed': self.trials[trial], 'current_frame_index': frame,
                        'next_expert_frame_index': frame+1, 'capture_physics_step': frame*5,
                        'next_expert_physics_step': frame*5+5,
                        'history_nonzero_contact': frame < (70 if fold == 'validation' else 60),
                        'target_semantics': 'next_saved_frame_expert_command_14_not_future_force_input'})
        private.write_text(''.join(json.dumps(row)+'\n' for row in self.rows))
        old_private = self.repo/'data/reports/relay_paired_training_index_candidate_v1.jsonl'
        old_private.write_text('frozen old index bytes\n')
        (ds/'relay_paired_training_index_candidate_v1.json').write_text(json.dumps({
            'private_index_sha256': digest(old_private)}))
        card = {'schema': 'lingbot_relay_paired_training_index_candidate_v1',
                'status': 'candidate_source_verified_not_training_approved',
                'variant': 'bottle_v2_add_train_keep_v1_heldout', 'training_approval': False,
                'source_machine_role': 'relay_single_rtx3090',
                'provenance_registry_sha256': digest(ds/'provenance_v1.json'),
                'private_index_sha256': digest(private), 'bottle_v1_split_sha256': digest(old_path),
                'tasks': {pilot.TASK: {'split_sha256': digest(self.new_path),
                                      'train_only_normalization_sha256': digest(norm),
                                      'folds': {'train': {'pairs': 590}, 'validation': {'pairs': 126}}}}}
        (ds/'relay_paired_training_index_candidate_v2.json').write_text(json.dumps(card))
        self.split = new

    def episode_class(self, observed):
        root = self.repo
        class FakeEpisode:
            def __init__(self, repo, trial, *, instruction):
                observed.append(trial)
                if 's100009_' in trial:
                    raise AssertionError('test episode MUST NEVER load')
                self.trial = trial
                self.card = json.loads((root/'datasets/cards'/f'{trial}.json').read_text())
            def __len__(self): return 118 if self.card['seed'] in pilot.TRAIN_SEEDS else 126
            def sample(self, index, *, include_images):
                assert include_images
                seed = self.card['seed']; frame = index
                hist = np.zeros((5, 4, 3), dtype=np.float32)
                if frame < (70 if seed == pilot.VALID_SEED else 60):
                    hist[0, 0, 0] = 1.
                return {'input': {'tactile_history': hist, 'tactile_mask': np.ones((5, 4)),
                    'tactile_age_ms': np.float32(2.), 'command_state_14': np.ones(14, dtype=np.float32),
                    'capture_physics_step': frame*5, 'tactile_window_valid': True,
                    'images_rgb': {cam: np.full((240, 320, 3), frame % 255, dtype=np.uint8)
                                   for cam in pilot.CAMERAS}},
                    'target': {'action_command_14': np.ones(14, dtype=np.float32)},
                    'metadata': {'trial_id': self.trial, 'task': pilot.TASK, 'seed': seed,
                                 'frame_index': frame, 'target_frame_index': frame+1,
                                 'next_frame_step_delta': 5}}
        return FakeEpisode

    def test_source_cards_holdout_and_no_test_episode(self):
        observed = []
        with patch.object(pilot, 'build_card', side_effect=lambda path, repo:
                          json.loads((self.repo/'datasets/cards'/f'{path.name}.json').read_text())), \
             patch.object(pilot, 'PairedEpisode', self.episode_class(observed)):
            v2, rows, evidence = pilot.verify_sources(self.repo)
            tr = pilot.load_fold(self.repo, v2, rows, 'train')
            va = pilot.load_fold(self.repo, v2, rows, 'validation')
            self.assertEqual((len(tr[0]), len(va[0]), int(va[4].sum())), (590, 126, 70))
            self.assertEqual(len(observed), 6)
            self.assertFalse(any('s100009_' in name for name in observed))
            self.assertEqual(evidence['old_holdout_sha256'], digest(
                self.repo/'datasets/relay_pickdiverse_seed_split_candidate_v1.json'))
            self.split['validation'] = self.split['test']
            self.new_path.write_text(json.dumps(self.split))
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                pilot.verify_sources(self.repo)

    def test_private_index_sha_and_card_sha_fail_closed(self):
        with patch.object(pilot, 'build_card', side_effect=lambda path, repo:
                          json.loads((self.repo/'datasets/cards'/f'{path.name}.json').read_text())):
            private = self.repo/'data/reports/relay_paired_training_index_candidate_v2.jsonl'
            private.write_text(private.read_text()+'tampered\n')
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                pilot.verify_sources(self.repo)
            private.write_text(''.join(json.dumps(row)+'\n' for row in self.rows))
            card = self.repo/'datasets/cards'/f'{self.split["train"][0]}.json'
            card.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'HDF5 SHA mismatch'):
                pilot.verify_sources(self.repo)

    def test_joint_and_gripper_mae_use_robot_command_indices(self):
        labels = np.zeros((2, 14), dtype=np.float32)
        visual = labels.copy()
        tactile = labels.copy()
        tactile[0, 6] = 2.
        tactile[1, 13] = 4.
        shuffled = tactile.copy()
        shuffled[0, 0] = 12.
        shuffled[1, 12] = 12.
        result = pilot._metrics({'visual': visual, 'tactile': tactile,
                                 'shuffled_past': shuffled}, labels, np.array([True, False]))
        for stratum, expected_gripper in (('past_contact', 1.), ('no_past_contact', 2.)):
            self.assertEqual(result[stratum]['arms']['tactile']['mae_12_joints'], 0.)
            self.assertEqual(result[stratum]['arms']['tactile']['mae_2_grippers'], expected_gripper)
            self.assertEqual(result[stratum]['paired_tactile_minus_visual']['mae_12_joints'], 0.)
            self.assertEqual(result[stratum]['paired_shuffled_minus_tactile']['mae_12_joints'], 1.)
            self.assertEqual(result[stratum]['paired_shuffled_minus_tactile']['mae_2_grippers'], 0.)

    def test_identical_arms_shuffle_metrics_and_explicit_status(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch unavailable in this CPU interpreter')
        observed = []
        out = self.repo/'models/cpu_proxy_fixture'
        # Synthetic fixture is intentionally outside Git; simulated ignore check.
        with patch.object(pilot, '_safe_out', return_value=out), \
             patch.object(pilot, 'build_card', side_effect=lambda path, repo:
                          json.loads((self.repo/'datasets/cards'/f'{path.name}.json').read_text())), \
             patch.object(pilot, 'PairedEpisode', self.episode_class(observed)):
            report = pilot.run(self.repo, out, 'cpu', steps=1)
        self.assertEqual(report['status'], 'exploratory_proxy_not_vla_not_approved')
        self.assertFalse(report['training_approval'])
        self.assertFalse(report['test_episode_loaded'])
        self.assertEqual(report['input_dimension_per_arm'], pilot.VIS_DIM+14+pilot.TOUCH_DIM)
        self.assertIsNotNone(report['validation']['past_contact']['paired_tactile_minus_visual']['mae_14d'])
        self.assertEqual(report['validation']['no_past_contact']['n'], 56)
        self.assertEqual(len(observed), 6)
        weights = torch.load(out/'proxy_weights.pt', map_location='cpu', weights_only=True)
        self.assertEqual(weights['visual']['0.weight'].shape, weights['tactile']['0.weight'].shape)
        touch = np.arange(126*pilot.TOUCH_DIM).reshape(126, pilot.TOUCH_DIM)
        shuffled, shift = pilot.permute_past(touch)
        self.assertTrue(0 < shift < 126)
        self.assertTrue(np.array_equal(shuffled, np.roll(touch, shift, axis=0)))
        self.assertFalse(np.any(np.all(shuffled == touch, axis=1)))


if __name__ == '__main__': unittest.main()
