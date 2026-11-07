"""CPU-only strict-load-order contract; NOT real LingBotV2 model inference."""
from types import SimpleNamespace
import unittest
import torch
from model_attach_contract import attach_visual_identity_adapter


class MinimalPatchedV2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(tactile_enabled=False)
        self.state_proj=torch.nn.Linear(14,8)
        self.tactile_adapter=None
    def _tactile_delta(self,state,hist,mask,age,valid):
        base=self.state_proj(state)
        return self.tactile_adapter(base,hist,mask,age,valid)-base
    def sample_actions(self,*args,**kwargs):pass
    def predict_velocity(self,*args,**kwargs):pass


class WrappedPatchedV2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model=MinimalPatchedV2()
        self.config=self.model.config


class AttachContractTests(unittest.TestCase):
    def test_top_level_lingbot_wrapper_load_attaches_only_to_inner_flow_matching(self):
        wrapped=WrappedPatchedV2();old_keys=set(wrapped.state_dict())
        self.assertTrue(all(name.startswith('model.') for name in old_keys))
        proof=wrapped.load_state_dict(wrapped.state_dict(),strict=True)
        report=attach_visual_identity_adapter(wrapped,proof)
        self.assertFalse(report['training_approval'])
        self.assertIsNotNone(wrapped.model.tactile_adapter)
        self.assertFalse(hasattr(wrapped,'tactile_adapter'))
        self.assertTrue(wrapped.config.tactile_enabled)
        self.assertFalse(wrapped.model.state_proj.weight.requires_grad)
        self.assertTrue(all(not p.requires_grad for name,p in wrapped.named_parameters()
                            if not name.startswith('model.tactile_adapter.')))
        self.assertTrue(any(p.requires_grad for name,p in wrapped.named_parameters()
                            if name.startswith('model.tactile_adapter.')))
        with self.assertRaisesRegex(ValueError,'without adapter'):
            attach_visual_identity_adapter(wrapped,proof)

    def test_strict_old_load_precedes_trainable_zero_residual(self):
        torch.manual_seed(5)
        m=MinimalPatchedV2();states=torch.ones(2,14)
        old=m.state_proj(states).clone()
        result=m.load_state_dict(m.state_dict(),strict=True)
        summary=attach_visual_identity_adapter(m,result)
        self.assertFalse(summary['training_approval'])
        self.assertFalse(summary['server_capability_advertised'])
        self.assertGreater(summary['adapter_parameter_count'],0)
        self.assertTrue(m.config.tactile_enabled)
        self.assertFalse(m.state_proj.weight.requires_grad)
        history=torch.ones(2,5,4,3);coverage=torch.ones(2,5,4)
        delta=m._tactile_delta(states,history,coverage,torch.zeros(2),torch.ones(2,dtype=torch.bool))
        self.assertTrue(torch.equal(delta,torch.zeros_like(old)))
        self.assertTrue(torch.equal(old,m.state_proj(states)))
        (delta+old).square().mean().backward()
        self.assertIsNone(m.state_proj.weight.grad)
        self.assertGreater(m.tactile_adapter.to_state.weight.grad.abs().sum().item(),0)
        with self.assertRaisesRegex(ValueError,'without adapter'):
            attach_visual_identity_adapter(m,result)

    def test_missing_or_extra_old_checkpoint_key_refuses_attachment(self):
        for result in (SimpleNamespace(missing_keys=['vision'],unexpected_keys=[]),
                       SimpleNamespace(missing_keys=[],unexpected_keys=['tactile_adapter.to_state.weight']),
                       SimpleNamespace(missing_keys=[],unexpected_keys=[]),  # forged success
                        None):
            with self.subTest(result=result):
                m=MinimalPatchedV2()
                with self.assertRaisesRegex(ValueError,'strict'):
                    attach_visual_identity_adapter(m,result)
                self.assertIsNone(m.tactile_adapter)
                self.assertFalse(m.config.tactile_enabled)

    def test_preinitialized_adapter_or_unpatched_model_refused(self):
        m=MinimalPatchedV2();m.config.tactile_enabled=True
        with self.assertRaisesRegex(ValueError,'strictly loaded'):
            attach_visual_identity_adapter(m,m.load_state_dict(m.state_dict(),strict=True))
        m=MinimalPatchedV2();m.predict_velocity=None
        with self.assertRaisesRegex(ValueError,'strictly loaded'):
            attach_visual_identity_adapter(m,m.load_state_dict(m.state_dict(),strict=True))
        wrapped=WrappedPatchedV2();wrapped.config=SimpleNamespace(tactile_enabled=False)
        with self.assertRaisesRegex(ValueError,'strictly loaded'):
            attach_visual_identity_adapter(wrapped,wrapped.load_state_dict(wrapped.state_dict(),strict=True))


if __name__=='__main__':unittest.main()
