import unittest
from prepare_contact_h50_batch import generate,SEED_FOLDS
class FixedProtocol(unittest.TestCase):
    def test_fold_and_real_horizon_not_changed(self):
        self.assertEqual(SEED_FOLDS,{100028:'train',100029:'validation'})
        for seed,fold in SEED_FOLDS.items():
            f=generate(seed)
            self.assertIn(f"'{fold}_candidate_only_not_assigned'",f['preregister_contact_h50.py'])
            self.assertIn('stride=10',f['record_contact_h50.py'])
            self.assertIn('MAX_CALLS=128',f['contact_h50_executor.py'])
            self.assertIn('MAX_PROPOSALS=128',f['contact_h50_proposals.py'])
            self.assertIn("task.take_action(command,action_type='qpos')",f['contact_h50_executor.py'])
    def test_no_seed_relabel_or_repeat(self):
        for s in (100023,100026,100027,100030):
            with self.assertRaises(ValueError):generate(s)
if __name__=='__main__':unittest.main()
