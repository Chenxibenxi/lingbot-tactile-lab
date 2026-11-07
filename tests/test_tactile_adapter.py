"""CPU-only research-model contracts: identity, masks, shape and gradients."""
import unittest
import json
from pathlib import Path

import torch
from tactile_adapter import TactileStateResidual
from split_and_norm import fit_train_only_stats


class TactileAdapterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.module = TactileStateResidual(state_width=12, hidden_width=8)
        self.state = torch.randn(3, 12)
        self.force = torch.ones(3, 5, 4, 3)
        self.force[..., 0] = 3.0
        self.force[..., 1] = 4.0
        self.mask = torch.ones(3, 5, 4)
        self.age = torch.zeros(3)
        self.valid = torch.tensor([True, False, True])

    def test_exact_initial_vision_identity_and_sensor_missing_fallback(self):
        initial = self.module(self.state, self.force, self.mask, self.age, self.valid)
        self.assertTrue(torch.equal(initial, self.state))
        self.assertIs(self.module(self.state, enabled=False), self.state)
        with torch.no_grad():
            self.module.to_state.weight.fill_(0.1)
        self.assertTrue(torch.equal(self.module(self.state, enabled=False), self.state))
        self.assertTrue(torch.equal(self.module(self.state, None), self.state))
        output = self.module(self.state, self.force, self.mask, self.age, self.valid)
        self.assertTrue(torch.equal(output[1], self.state[1]))
        self.assertFalse(torch.equal(output[0], self.state[0]))

    def test_bad_geometry_and_nan_do_not_silently_train(self):
        with self.assertRaisesRegex(ValueError, 'tactile_history'):
            self.module(self.state, self.force[:, :4], self.mask, self.age, self.valid)
        bad=self.force.clone();bad[0,0,0,0]=float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            self.module(self.state, bad, self.mask, self.age, self.valid)
        with self.assertRaisesRegex(ValueError,'mask'):
            self.module(self.state, self.force, self.mask*2, self.age, self.valid)

    def test_normalization_is_checkpointed_and_train_only(self):
        import numpy as np
        samples = []
        for trial, load in [('train_a', 3.), ('train_b', 5.), ('heldout', 1e7)]:
            x = np.ones((5,4,3), dtype=np.float32)
            x[:,:,:2] = load
            samples.append({'metadata':{'trial_id':trial}, 'input':{
                'tactile_history':x, 'tactile_mask':np.ones((5,4)),
                'tactile_window_valid':True}})
        raw_stats = fit_train_only_stats(samples, {'train_a','train_b'})
        with self.assertRaisesRegex(ValueError,'explicit true approval'):
            self.module.set_normalization(raw_stats)
        stats = dict(raw_stats,training_approval=True)  # synthetic math-only fixture, not real data approval
        self.module.set_normalization(stats)
        self.assertTrue(torch.equal(self.module(self.state,self.force,self.mask,self.age,self.valid), self.state))
        copy = TactileStateResidual(state_width=12, hidden_width=8)
        copy.load_state_dict(self.module.state_dict())
        self.assertTrue(torch.equal(copy.train_median, self.module.train_median))
        bad = dict(stats, source_trials=[])
        with self.assertRaisesRegex(ValueError, 'training-only'):
            self.module.set_normalization(bad)
        with self.assertRaisesRegex(ValueError,'not approved'):
            self.module.set_normalization(dict(stats,training_approval=False))

    def test_every_real_candidate_norm_still_rejected_by_training_gate(self):
        configs=Path(__file__).resolve().parents[1]/'configs'
        paths=sorted(configs.glob('relay_*norm_candidate_v*.json'))
        self.assertGreaterEqual(len(paths),6)
        for path in paths:
            stats=json.loads(path.read_text())
            self.assertIs(stats.get('training_approval'),False,path.name)
            with self.subTest(config=path.name), self.assertRaisesRegex(ValueError,'explicit true approval'):
                self.module.set_normalization(stats)

    def test_output_projection_receives_gradient(self):
        output=self.module(self.state, self.force, self.mask, self.age, self.valid)
        output.square().mean().backward()
        self.assertGreater(self.module.to_state.weight.grad.abs().sum().item(), 0)
        self.assertIsNotNone(self.module.temporal.weight_ih_l0.grad)
        # Zero-init output projection learns first; temporal encoder receives a
        # nonzero gradient after the projection takes a single optimizer step.
        with torch.no_grad():
            self.module.to_state.weight -= .01 * self.module.to_state.weight.grad
        self.module.zero_grad(set_to_none=True)
        self.module(self.state, self.force, self.mask, self.age, self.valid).square().mean().backward()
        self.assertGreater(self.module.temporal.weight_ih_l0.grad.abs().sum().item(), 0)


if __name__ == '__main__': unittest.main()
