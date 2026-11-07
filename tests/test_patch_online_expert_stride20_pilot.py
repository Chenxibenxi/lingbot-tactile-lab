"""A new seed derives from exact immutable source bytes; no GPU or private output."""
import hashlib
from unittest.mock import patch
import unittest
from patch_online_expert_stride20_pilot import REPO,PRIVATE,TRIAL,PRE_SHA,REC_SHA,OLD_PRE,OLD_REC,generate


class IsolatedNewSeedPilotPatchTests(unittest.TestCase):
    def test_sha_locked_cpu_preview_leaves_old_source_and_private_trials_untouched(self):
        self.assertEqual(hashlib.sha256(OLD_PRE.read_bytes()).hexdigest(),PRE_SHA)
        self.assertEqual(hashlib.sha256(OLD_REC.read_bytes()).hexdigest(),REC_SHA)
        before=(OLD_PRE.read_bytes(),OLD_REC.read_bytes())
        generated=generate()
        self.assertEqual(set(generated),{'preregister_online_expert_stride20.py','record_online_expert_stride20.py'})
        self.assertTrue(all('100022' not in text for text in generated.values()))
        self.assertIn("SEED=100023",generated['preregister_online_expert_stride20.py'])
        self.assertIn('execute_online_expert(robot,capture,physics_cap=MAX_PHYSICS_STEPS,stride=20)',
                      generated['record_online_expert_stride20.py'])
        self.assertEqual((OLD_PRE.read_bytes(),OLD_REC.read_bytes()),before)
        if PRIVATE.exists():
            self.assertTrue(all((PRIVATE/name).read_text()==text for name,text in generated.items()))
        scope={'__file__':str(PRIVATE/'preregister_online_expert_stride20.py')}
        exec(compile(generated['preregister_online_expert_stride20.py'],scope['__file__'],'exec'),scope)
        trial=REPO/'data/paired'/TRIAL
        if trial.exists():
            with self.assertRaisesRegex(ValueError,'already attempted'):
                scope['preregister']()
        else:
            plan=scope['preregister']()
            self.assertEqual((scope['REPO'],plan['seed'],plan['new_trial_id']),(REPO,100023,TRIAL))
            self.assertFalse(plan['gpu_run_allowed_now'])
            self.assertFalse(plan['training_approval'])

    def test_changed_old_collector_fails_closed_before_generation(self):
        with patch('patch_online_expert_stride20_pilot.PRE_SHA','0'*64):
            with self.assertRaisesRegex(ValueError,'SHA changed'):generate()
        with patch('patch_online_expert_stride20_pilot.REC_SHA','0'*64):
            with self.assertRaisesRegex(ValueError,'SHA changed'):generate()


if __name__=='__main__':unittest.main()
