import unittest
import numpy as np
from m8_paired_seed_statistics import summarize


class DurationStatistics(unittest.TestCase):
    def test_shared_learning_is_not_unique_tactile_benefit(self):
        x=np.zeros((16,3,2,4),dtype=int)
        x[:,:,1,:]=1
        r=summarize(x,list(range(100103,100119)))
        self.assertFalse(r['duration_explanation_supported'])
        self.assertFalse(r['final_unique_tactile_benefit'])
        self.assertEqual([c['effect'] for c in r['contrasts']],[0.,0.,0.])

    def test_genuine_interaction_detected(self):
        x=np.zeros((16,3,2,4),dtype=int);x[:,:,1,1]=1
        r=summarize(x,list(range(100103,100119)))
        self.assertTrue(r['duration_explanation_supported'])
        self.assertTrue(r['final_unique_tactile_benefit'])
        self.assertEqual(r['contrasts'][0]['effect'],1.)

    def test_preexisting_advantage_not_duration_explanation(self):
        x=np.zeros((16,3,2,4),dtype=int);x[:,:,:,1]=1
        r=summarize(x,list(range(100103,100119)))
        self.assertFalse(r['duration_explanation_supported'])
        self.assertTrue(r['final_unique_tactile_benefit'])

    def test_missing_or_reused_test_rejected(self):
        with self.assertRaises(ValueError):summarize(np.zeros((16,3,2,4)),list(range(100087,100103)))
        with self.assertRaises(ValueError):summarize(np.zeros((15,3,2,4),dtype=int),list(range(100103,100119)))
        x=np.zeros((11,3,2,4),dtype=int);x[:,:,1,1]=1
        self.assertFalse(summarize(x,list(range(100103,100114)))['duration_explanation_supported'])
