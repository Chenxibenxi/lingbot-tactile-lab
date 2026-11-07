import unittest
from report_recovery_development_batch import feasibility,SEEDS,CONDITIONS


class Feasibility(unittest.TestCase):
    def fixture(self):
        rows={(s,c):dict(seed=s,condition=c,status='completed',official_success=True,termination='task_success',
                         intervention_applied=c!=CONDITIONS[0],usable_recovery_candidate=True) for s in SEEDS for c in CONDITIONS}
        pairs={s:dict(technical_comparable=True) for s in SEEDS}
        return rows,pairs

    def test_thresholds_and_unknown_not_dropped(self):
        rows,pairs=self.fixture()
        rows[SEEDS[0],CONDITIONS[0]]['official_success']=False
        for s in SEEDS[:2]:rows[s,CONDITIONS[1]]['official_success']=False
        self.assertTrue(feasibility(rows,pairs)['feasibility_gate'])
        rows[SEEDS[0],CONDITIONS[0]]['status']='needs_inspection'
        result=feasibility(rows,pairs)
        self.assertFalse(result['feasibility_gate'])
        self.assertEqual(result['denominator_per_condition'],4)
        self.assertEqual(result['unknown_attempts'],[[SEEDS[0],CONDITIONS[0]]])

    def test_success_without_intervention_is_not_recovery(self):
        rows,pairs=self.fixture()
        for s in SEEDS:rows[s,CONDITIONS[1]]['intervention_applied']=False
        result=feasibility(rows,pairs)
        self.assertEqual(result['usable_recovery_successes'],0)
        self.assertFalse(result['feasibility_gate'])

    def test_incomplete_matrix_rejected(self):
        rows,pairs=self.fixture();rows.pop((SEEDS[0],CONDITIONS[0]))
        with self.assertRaises(ValueError):feasibility(rows,pairs)
