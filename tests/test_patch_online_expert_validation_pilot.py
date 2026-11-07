"""Validation-only source clone remains separate from the one training-intent seed."""
import hashlib
import unittest
from unittest.mock import patch
from patch_online_expert_validation_pilot import REPO,PRIVATE,TRIAL,generate
from patch_online_expert_stride20_pilot import OLD_PRE,OLD_REC,PRE_SHA,REC_SHA


class ValidationOnlineSeedSourceTests(unittest.TestCase):
    def test_validation_fold_and_unique_seed_are_locked_before_gpu(self):
        self.assertEqual(hashlib.sha256(OLD_PRE.read_bytes()).hexdigest(),PRE_SHA)
        self.assertEqual(hashlib.sha256(OLD_REC.read_bytes()).hexdigest(),REC_SHA)
        files=generate()
        self.assertTrue(all('100022' not in x and '100023' not in x for x in files.values()))
        self.assertIn('stride=20',files['record_online_expert_validation.py'])
        scope={'__file__':str(PRIVATE/'preregister_online_expert_validation.py')}
        exec(compile(files['preregister_online_expert_validation.py'],scope['__file__'],'exec'),scope)
        trial=REPO/'data/paired'/TRIAL
        if trial.exists():
            with self.assertRaisesRegex(ValueError,'already attempted'):scope['preregister']()
        else:
            plan=scope['preregister']()
            self.assertEqual((plan['seed'],plan['new_trial_id']),(100024,TRIAL))
            self.assertEqual(plan['fold_intent'],'validation_candidate_only_not_assigned')
            self.assertFalse(plan['training_approval'])
            self.assertFalse(plan['gpu_run_allowed_now'])
        if PRIVATE.exists():
            self.assertTrue(all((PRIVATE/name).read_text()==text for name,text in files.items()))

    def test_source_drift_refuses_validation_collector(self):
        with patch('patch_online_expert_validation_pilot.PRE_SHA','0'*64):
            with self.assertRaisesRegex(ValueError,'SHA changed'):generate()
        with patch('patch_online_expert_validation_pilot.REC_SHA','0'*64):
            with self.assertRaisesRegex(ValueError,'SHA changed'):generate()


if __name__=='__main__':unittest.main()
