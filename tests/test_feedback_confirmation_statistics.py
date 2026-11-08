import unittest
import numpy as np
from feedback_confirmation_statistics import summarize,SEEDS

class FeedbackStatistics(unittest.TestCase):
    def test_shared_frequency_gain_is_not_tactile_gain(self):
        x=np.zeros((16,3,2,4),dtype=np.int64);x[:,:,1,:]=1
        result=summarize(x,SEEDS)
        self.assertFalse(result['feedback_interaction_supported'])
        self.assertFalse(result['high_frequency_unique_tactile_benefit'])
        self.assertTrue(all(r['effect']==0 for r in result['contrasts']))

    def test_unique_high_frequency_effect_and_donor_control(self):
        x=np.zeros((16,3,2,4),dtype=np.int64);x[:,:,1,1]=1
        result=summarize(x,SEEDS)
        self.assertTrue(result['feedback_interaction_supported'])
        self.assertTrue(result['high_frequency_unique_tactile_benefit'])
        x[:,:,1,2]=1
        self.assertFalse(summarize(x,SEEDS)['high_frequency_unique_tactile_benefit'])

    def test_missing_pilot_or_unknown_rejected(self):
        x=np.zeros((16,3,2,4),dtype=np.int64)
        for values,seeds in ((x[:15],SEEDS[:15]),(x,[100127]+SEEDS[1:]),(x.astype(float),SEEDS)):
            with self.assertRaises(ValueError):summarize(values,seeds)
        x[0,0,0,0]=-1
        with self.assertRaises(ValueError):summarize(x,SEEDS)
