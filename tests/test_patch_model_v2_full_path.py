"""Execute patched V2 sampling *methods* on a tiny CPU Qwen stand-in.

This checks the real generator's denoising control flow, NOT a real LingBot
instance, original checkpoint, end-to-end training, or server deployment.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from patch_model_v2 import SOURCE, patch_model
from tactile_adapter import TactileStateResidual


class MinimalV1:
    def embed_suffix(self, state, noisy_actions, timestep):
        b=state.shape[0]
        emb=torch.cat((self.state_proj(state)[:,None],noisy_actions),dim=1)
        return timestep,emb,torch.ones(b,emb.shape[1],dtype=torch.bool),torch.zeros(b,emb.shape[1],dtype=torch.bool)


class TinyQwen:
    def __init__(self):self.prefix_calls=0;self.suffix_calls=0
    def forward(self, *, inputs_embeds, **kwargs):
        prefix,suffix=inputs_embeds
        if suffix is None:
            self.prefix_calls+=1
            return None,object(),None
        self.suffix_calls+=1
        # Force action tokens to depend on the suffix's STATE token.
        return (None,suffix+suffix[:,0:1]),None,None


def minimal_masks(pad,attention):
    return torch.ones(pad.shape[0],pad.shape[1],pad.shape[1],dtype=torch.bool)


def extracted_generated_methods():
    patched=patch_model(SOURCE.read_text())
    tree=ast.parse(patched)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FlowMatchingV2')
    wanted={'_tactile_delta','embed_suffix','sample_actions','predict_velocity'}
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in methods}==wanted
    node=ast.ClassDef(name='TinyPatched',bases=[ast.Name(id='MinimalV1',ctx=ast.Load())],keywords=[],body=methods,decorator_list=[])
    scope={'MinimalV1':MinimalV1,'torch':torch,'Tensor':torch.Tensor,
           'make_att_2d_masks':minimal_masks}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),scope)
    return scope['TinyPatched']


class GeneratedSamplingRouteTest(unittest.TestCase):
    def test_real_generated_denoise_calls_tactile_once_and_falls_back_to_vision(self):
        Model=extracted_generated_methods();m=Model()
        m.config=SimpleNamespace(action_fp32=False,use_cache=True,num_steps=2,
                                 n_action_steps=2,max_action_dim=8)
        torch.manual_seed(2)
        m.state_proj=torch.nn.Linear(4,8)
        m.action_out_proj=torch.nn.Linear(8,8,bias=False)
        m.qwenvl_with_expert=TinyQwen()
        m.block_future_depth_to_action=False
        m._block_suffix_to_future_video_if_enabled_=lambda a,**kw:a
        m._build_full_position_ids=lambda *args:torch.zeros(1,3,4,dtype=torch.long)
        m.embed_prefix=lambda *args,**kw:(torch.zeros(1,1,8),torch.ones(1,1,dtype=torch.bool),
                   torch.zeros(1,1,dtype=torch.bool),torch.zeros(1,3,1,dtype=torch.long),None,None)
        m.tactile_adapter=TactileStateResidual(8)
        args=(torch.zeros(1,1,8),torch.ones(1,1,dtype=torch.bool),torch.ones(1,1,dtype=torch.long),
              torch.ones(1,1,dtype=torch.bool),torch.ones(1,4))
        # The original sampler mutates x_t=noise in place; each comparison needs
        # a FRESH identical noise tensor or results would alias and falsely match.
        call=lambda **touch:m.sample_actions(*args,noise=torch.ones(1,2,8),
                              image_grid_thw=torch.ones(1,3,dtype=torch.long),**touch)
        baseline=call()
        h=torch.ones(1,5,4,3);mask=torch.ones(1,5,4)
        with patch.object(m.tactile_adapter.temporal,'forward',wraps=m.tactile_adapter.temporal.forward) as gru:
            zero=call(tactile_history=h,tactile_mask=mask,
                                  tactile_age_ms=torch.zeros(1),tactile_valid=torch.ones(1,dtype=torch.bool))
            self.assertEqual(gru.call_count,1)
        self.assertTrue(torch.equal(baseline,zero))
        self.assertGreater(m.qwenvl_with_expert.suffix_calls,1)
        with torch.no_grad():m.tactile_adapter.to_state.weight.fill_(.02)
        changed=call(tactile_history=h,tactile_mask=mask,
                                  tactile_age_ms=torch.zeros(1),tactile_valid=torch.ones(1,dtype=torch.bool))
        self.assertFalse(torch.equal(baseline,changed))
        invalid=call(tactile_history=h,tactile_mask=mask,
                                  tactile_age_ms=torch.zeros(1),tactile_valid=torch.zeros(1,dtype=torch.bool))
        self.assertTrue(torch.equal(baseline,invalid))
        m.tactile_adapter=None
        with self.assertRaisesRegex(RuntimeError,'without a wired adapter'):
            call(tactile_history=h,tactile_mask=mask,
                             tactile_age_ms=torch.zeros(1),tactile_valid=torch.ones(1,dtype=torch.bool))


class GeneratedTrainingRouteTest(unittest.TestCase):
    def test_actual_generated_top_forward_passes_all_four_causal_fields(self):
        tree=ast.parse(patch_model(SOURCE.read_text()))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LingbotVlaV2Policy')
        method,=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward']
        node=ast.ClassDef(name='TinyPolicy',bases=[],keywords=[],body=[method],decorator_list=[])
        scope={'torch':torch,'Tensor':torch.Tensor}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),scope)
        policy=scope['TinyPolicy']()
        policy.config=SimpleNamespace(action_fp32=False,loss_type='fm',action_dim=14)
        class CaptureInner:
            def __init__(self):self.forwarded=None
            def forward(self,*args,**kwargs):
                self.forwarded=kwargs
                loss=torch.ones(1,2,55,requires_grad=True)
                zero=loss.new_zeros(())
                return loss,zero,zero,zero,None,zero,zero,{},None,None,None
        policy.model=CaptureInner()
        history=torch.ones(1,5,4,3);mask=torch.ones(1,5,4)
        age=torch.tensor([0.]);valid=torch.tensor([True])
        out=policy.forward(images=torch.zeros(1,1,8),img_masks=None,state=torch.ones(1,14),
               lang_tokens=torch.ones(1,1,dtype=torch.long),lang_masks=None,
               actions=torch.ones(1,2,55),tactile_history=history,tactile_mask=mask,
               tactile_age_ms=age,tactile_valid=valid)
        self.assertTrue(torch.is_tensor(out[0]))
        for key,value in (('tactile_history',history),('tactile_mask',mask),
                          ('tactile_age_ms',age),('tactile_valid',valid)):
            self.assertIs(policy.model.forwarded[key],value)
        out[0].backward()
        self.assertIsNotNone(out[0].grad_fn)
        policy.forward(images=torch.zeros(1,1,8),img_masks=None,state=torch.ones(1,14),
               lang_tokens=torch.ones(1,1,dtype=torch.long),lang_masks=None,
               actions=torch.ones(1,2,55))
        self.assertTrue(all(policy.model.forwarded[key] is None for key in
            ('tactile_history','tactile_mask','tactile_age_ms','tactile_valid')))


if __name__=='__main__':unittest.main()
