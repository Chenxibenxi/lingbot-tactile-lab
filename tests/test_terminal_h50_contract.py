import unittest
import torch
from terminal_h50_contract import prepare_censored_flow,tactile_arm
class TerminalContract(unittest.TestCase):
    def fixture(self):
        a=torch.randn(1,50,55,requires_grad=True);j=torch.zeros_like(a,dtype=torch.bool);j[:,:,:14]=True
        valid=torch.arange(50)[None,:]<19;t=torch.randn_like(a,requires_grad=True)
        return a,j,valid,t
    def test_sentinel_invariance_and_teacher_detach(self):
        a,j,v,t=self.fixture();b=a.detach().clone();b[:,19:]=float('nan')
        x,m=prepare_censored_flow(a,j,v,t);y,n=prepare_censored_flow(b,j,v,t)
        self.assertTrue(torch.equal(x,y));self.assertEqual(m.sum().item(),19*14)
        x.square().sum().backward()
        self.assertTrue(torch.all(a.grad[:,19:]==0));self.assertIsNone(t.grad)
    def test_empty_or_holey_labels_refused(self):
        a,j,v,t=self.fixture()
        for bad in (torch.zeros_like(v),v.roll(1,1)):
            with self.assertRaises(ValueError):prepare_censored_flow(a,j,bad,t)
    def test_real_nan_refused(self):
        a,j,v,t=self.fixture();a=a.detach();a[0,0,0]=float('nan')
        with self.assertRaises(ValueError):prepare_censored_flow(a,j,v,t)
    def test_full_horizon_preserves_real_joint_values(self):
        a,j,v,t=self.fixture();v[:]=True;x,m=prepare_censored_flow(a,j,v,t)
        self.assertTrue(torch.equal(x[j],a[j]));self.assertEqual(m.sum(),700)
    def test_past_reverse_preserves_coverage_pairing(self):
        p={'tactile_history':torch.arange(60.).reshape(1,5,4,3),'tactile_mask':torch.arange(20.).reshape(1,5,4)}
        q=tactile_arm(p,'reverse')
        self.assertTrue(torch.equal(q['tactile_history'][:,0],p['tactile_history'][:,-1]))
        self.assertTrue(torch.equal(q['tactile_mask'][:,0],p['tactile_mask'][:,-1]))
        self.assertTrue(torch.equal(p['tactile_history'].flatten(),torch.arange(60.)))
if __name__=='__main__':unittest.main()
