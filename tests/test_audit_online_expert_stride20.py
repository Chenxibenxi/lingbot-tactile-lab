"""A real single-seed H50 candidate remains unapproved despite valid HDF5."""
from pathlib import Path
import unittest
from unittest.mock import patch
from audit_online_expert_stride20 import REPO,TRIAL,generate,audit


class RealOnlineExpertH50AuditTests(unittest.TestCase):
    def setUp(self):
        if not (REPO/'data/paired'/TRIAL/'completion.json').exists():
            self.skipTest('ignored single-seed GPU trial not available in this checkout')

    def test_real_hdf5_and_causal_eight_anchors_do_not_authorize_training(self):
        report=audit()
        self.assertEqual((report['actual_take_action_calls'],report['physics_steps']),(59,6808))
        self.assertEqual(report['causal_h50_candidate_anchor_indices'],list(range(2,10)))
        self.assertEqual(report['approved_h50_train_windows'],0)
        self.assertEqual(report['approved_h50_validation_windows'],0)
        self.assertFalse(report['gpu_training_permitted_now'])
        self.assertFalse(report['training_approval'])

    def test_candidate_registry_card_must_match_rebuilt_live_source(self):
        with patch('audit_online_expert_stride20.build_card',return_value={'status':'candidate_unassigned_split'}):
            with self.assertRaisesRegex(ValueError,'candidate card differs'):
                audit()

    def test_private_newseed_collector_drift_blocks_reinterpretation(self):
        original=generate()
        forged=dict(original)
        forged['record_online_expert_stride20.py']+='\n# unreviewed change\n'
        with patch('audit_online_expert_stride20.generate',return_value=forged):
            with self.assertRaisesRegex(ValueError,'differs from pinned'):
                audit()


if __name__=='__main__':unittest.main()
