import unittest
from prepare_validation_batch_20261106 import generate
class BatchContract(unittest.TestCase):
    def test_only_preregistered_seeds(self):
        for seed in (100023,100024,100025,100028,True):
            with self.assertRaises(ValueError):generate(seed)
    def test_independent_validation_no_source_mutation(self):
        for seed in (100026,100027):
            files=generate(seed)
            pre=files['preregister_validation_batch.py'];rec=files['record_validation_batch.py']
            self.assertIn(f'SEED={seed}',pre)
            self.assertIn('validation_candidate_only_not_assigned',pre)
            self.assertIn('stride=20',rec)
            self.assertNotIn('100022',pre+rec)
            self.assertIn('verify_gpu_lease()',rec)
            self.assertIn('preregister(repo=REPO',pre)
if __name__=='__main__':unittest.main()
