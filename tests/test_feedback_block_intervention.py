import unittest
import numpy as np
from feedback_block_intervention import BlockIntervention

class BlockTest(unittest.TestCase):
    def test_sources_and_refresh(self):
        fresh, cached = BlockIntervention('fresh_first'), BlockIntervention('cached_block')
        for i in range(23):
            p = np.repeat((1000*i+np.arange(50,dtype=float))[:,None],14,axis=1)
            a, fa = fresh.select(p,action_index=i)
            b, ca = cached.select(p,action_index=i)
            np.testing.assert_array_equal(a, np.full(14,1000*i))
            np.testing.assert_array_equal(b, np.full(14,1000*(i//10*10)+i%10))
            self.assertEqual(ca['source_request_index'],i//10*10)
            self.assertEqual(ca['source_prediction_offset'],i%10)
            self.assertEqual(fa['source_request_index'],i)
            p[:] = -999  # retained block must not alias caller storage
            b[:] = -888  # output must not alias cached storage
    def test_reject_and_no_state_advance(self):
        c=BlockIntervention('cached_block')
        for p,index in [(np.zeros((49,14)),0),(np.full((50,14),np.nan),0),(np.zeros((50,14)),1)]:
            with self.assertRaises(ValueError):c.select(p,action_index=index)
        c.select(np.zeros((50,14)),action_index=0)
        with self.assertRaises(ValueError):c.select(np.zeros((50,14)),action_index=0)

if __name__=='__main__':unittest.main()
