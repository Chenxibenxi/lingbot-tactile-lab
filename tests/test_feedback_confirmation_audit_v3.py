import unittest
from audit_feedback_confirmation_v3 import analyze_complete,SEEDS,ARMS


def data():
    rows=[dict(seed=s,repeat=r,cadence=c,arm=a,success=False,technical_pass=True) for s in SEEDS for r in (0,1,2) for c in (10,1) for a in ARMS]
    pairs={f'{s}_r{r}':True for s in SEEDS for r in (0,1,2)}
    return rows,pairs


class FinalGate(unittest.TestCase):
    def test_all_failed_tasks_are_valid_negative_evidence(self):
        r,p=data();x=analyze_complete(r,p)
        self.assertTrue(x['confirmation_gate'])
        self.assertFalse(x['statistics']['high_frequency_unique_tactile_benefit'])
    def test_unknown_or_bad_pair_disables_main_inference(self):
        r,p=data();r[0]['success']=None
        self.assertIsNone(analyze_complete(r,p)['statistics'])
        r,p=data();p[next(iter(p))]=False
        self.assertIsNone(analyze_complete(r,p)['statistics'])
    def test_duplicate_or_missing_trials_rejected(self):
        r,p=data()
        with self.assertRaises(ValueError):analyze_complete(r[:-1],p)
        r[-1]=r[0]
        with self.assertRaises(ValueError):analyze_complete(r,p)
