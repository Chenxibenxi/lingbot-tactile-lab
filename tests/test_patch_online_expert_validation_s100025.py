"""Only approved new validation seed100025; old source and train fold unchanged."""
import hashlib
import unittest
from unittest.mock import patch
from patch_online_expert_validation_s100025 import REPO,PRIVATE,TRIAL,generate
from patch_online_expert_stride20_pilot import OLD_PRE,OLD_REC,PRE_SHA,REC_SHA


class NextValidationSeedSourceTests(unittest.TestCase):
    def test_fresh_validation_only_private_source(self):
        self.assertEqual(hashlib.sha256(OLD_PRE.read_bytes()).hexdigest(),PRE_SHA)
        self.assertEqual(hashlib.sha256(OLD_REC.read_bytes()).hexdigest(),REC_SHA)
        files=generate()
        self.assertTrue(all(str(old) not in text for old in (100022,100023,100024)
                            for text in files.values()))
        self.assertIn('stride=20',files['record_online_expert_validation_s100025.py'])
        scope={'__file__':str(PRIVATE/'preregister_online_expert_validation_s100025.py')}
        exec(compile(files['preregister_online_expert_validation_s100025.py'],scope['__file__'],'exec'),scope)
        trial=REPO/'data/paired'/TRIAL
        if trial.exists():
            with self.assertRaisesRegex(ValueError,'already attempted'):scope['preregister']()
        else:
            plan=scope['preregister']()
            self.assertEqual((plan['seed'],plan['new_trial_id']),(100025,TRIAL))
            self.assertEqual(plan['fold_intent'],'validation_candidate_only_not_assigned')
            self.assertFalse(plan['gpu_run_allowed_now'])
            self.assertFalse(plan['training_approval'])
        if PRIVATE.exists():
            self.assertTrue(all((PRIVATE/name).read_text()==text for name,text in files.items()))

    def test_frozen_original_source_drift_refuses(self):
        with patch('patch_online_expert_validation_s100025.PRE_SHA','0'*64):
            with self.assertRaisesRegex(ValueError,'SHA changed'):generate()
        with patch('patch_online_expert_validation_s100025.REC_SHA','0'*64):
            with self.assertRaisesRegex(ValueError,'SHA changed'):generate()


if __name__=='__main__':unittest.main()
