import unittest,torch
from multiseed_tactile_controls import payload_for,assign_mismatches,balanced_order

def item(seed,fold,contact,frame=0):
    return {'meta':{'seed':seed,'fold':fold,'contact':contact,'frame_index':frame},'tactile':{'tactile_history':torch.arange(60.).reshape(1,5,4,3),'tactile_mask':torch.ones(1,5,4),'tactile_age_ms':torch.zeros(1),'tactile_valid':torch.ones(1,dtype=torch.bool)}}
class Controls(unittest.TestCase):
    def test_mean_preserves_masked_statistic_without_future_data(self):
        x=item(1,'train',True);x['tactile']['tactile_mask'][:,0]=0
        before=x['tactile']['tactile_history'].clone();y=payload_for(x,'mean');w=y['tactile_mask'].unsqueeze(-1)
        self.assertTrue(torch.equal((before*w).sum(1),(y['tactile_history']*w).sum(1)))
        self.assertTrue(torch.equal(before,x['tactile']['tactile_history']))
        self.assertTrue(torch.equal(y['tactile_history'][:,0],y['tactile_history'][:,4]))
    def test_fractional_mask_mean(self):
        x=item(1,'train',True);x['tactile']['tactile_mask'].fill_(.1)
        y=payload_for(x,'mean')
        self.assertTrue(torch.allclose(y['tactile_history'][:,0],x['tactile']['tactile_history'].mean(1)))
    def test_donors_no_validation_or_same_seed_and_no_target_access(self):
        xs=[item(s,f,c) for s,f in [(1,'train'),(2,'train'),(3,'validation')] for c in [False,True]]
        assign_mismatches(xs)
        for x in xs:
            self.assertEqual(x['mismatch_meta']['fold'],'train');self.assertNotEqual(x['meta']['seed'],x['mismatch_meta']['seed']);self.assertEqual(x['meta']['contact'],x['mismatch_meta']['contact'])
    def test_balancing_is_fixed_and_complete(self):
        xs=[item(s,'train',c) for s in [1,2,3] for c in [False,True]]
        a=balanced_order(xs);self.assertEqual(a,balanced_order(xs));self.assertEqual(len(a),360);self.assertEqual(set(a),set(range(6)))
    def test_missing_stratum_and_single_seed_rejected(self):
        with self.assertRaises(ValueError):balanced_order([item(1,'train',True)])
        with self.assertRaises(ValueError):assign_mismatches([item(1,'train',True)])
