import unittest
from prepare_multiseed_m1 import generate,FOLDS
class Protocol(unittest.TestCase):
    def test_fixed_disjoint_folds_and_future_eval(self):
        train={s for s,f in FOLDS.items() if f=='train'};val={s for s,f in FOLDS.items() if f=='validation'}
        self.assertEqual(train,set(range(100033,100039)));self.assertEqual(val,{100039,100040})
        self.assertFalse((train|val)&set(range(100041,100045)))
    def test_generated_identity_and_instrumentation(self):
        for seed,fold in FOLDS.items():
            files=generate(seed);pre=files['preregister_multiseed_m1.py'];rec=files['record_multiseed_m1.py']
            self.assertIn(f'SEED={seed}',pre);self.assertIn(f"'fold_intent':'{fold}_candidate_only_not_assigned'",pre)
            self.assertIn(f's{seed}_online_h50_stride20_{fold}_expert0',pre)
            self.assertIn('physics_cap=MAX_PHYSICS_STEPS,stride=20',rec)
            self.assertIn(f"LEASE_OWNER='lingbot-tactile-lab/multiseed-m1-{seed}'",rec)
            self.assertNotIn('100022',pre+rec)
    def test_other_seed_rejected(self):
        for seed in [100023,100041,100032,'100033',True]:
            with self.assertRaises(ValueError):generate(seed)
