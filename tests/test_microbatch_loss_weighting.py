import unittest
import torch
from microbatch_loss_weighting import weighted_micro_loss


class LossWeighting(unittest.TestCase):
    def test_unequal_valid_lengths_loss_and_gradient(self):
        # Same fixed inputs and parameter; unequal valid tails expose naive averaging.
        x = torch.arange(20, dtype=torch.float64).reshape(4,5)/10
        mask = torch.arange(5)[None,:] < torch.tensor([1,5,2,4])[:,None]
        def terms(p, sl):
            values = (p*x[sl]-0.7).square()
            action = values[mask[sl]].mean()
            auxiliary = (p*x[sl].mean(1)+0.3).square().mean()
            return action+auxiliary, action
        p = torch.tensor(0.8,dtype=torch.float64,requires_grad=True)
        expected = terms(p,slice(None))[0]
        expected_grad = torch.autograd.grad(expected,p)[0]
        for size in (1,2,4):
            q = p.detach().clone().requires_grad_()
            weighted = 0
            for start in range(0,4,size):
                sl = slice(start,start+size)
                total,action = terms(q,sl)
                weighted += weighted_micro_loss(total,action,
                    valid_coordinates=int(mask[sl].sum()),global_valid_coordinates=int(mask.sum()),
                    samples=size,global_samples=4,alignment_disabled=True,
                    sequence_mode='per_sequence',equal_router_tokens=True)
            torch.testing.assert_close(weighted,expected)
            torch.testing.assert_close(torch.autograd.grad(weighted,q)[0],expected_grad)
        naive = sum(terms(p,slice(i,i+1))[0] for i in range(4))/4
        self.assertGreater(abs(float((naive-expected).detach())), 0.01)

    def test_reject_unverified_contract(self):
        args=dict(valid_coordinates=1,global_valid_coordinates=4,samples=1,
                  global_samples=4,alignment_disabled=True,sequence_mode='per_sequence',equal_router_tokens=True)
        for change in ({'sequence_mode':'global'},{'alignment_disabled':False},
                       {'equal_router_tokens':False},{'valid_coordinates':0},
                       {'samples':True},{'valid_coordinates':5}):
            with self.assertRaises(ValueError): weighted_micro_loss(2.,1.,**(args|change))
