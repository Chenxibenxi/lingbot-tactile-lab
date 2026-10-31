"""Uninstalled model patch: AST anchors plus isolated CPU state-token test."""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from patch_model_v2 import SOURCE,patch_model
from tactile_adapter import TactileStateResidual


class MinimalV1:
    def embed_suffix(self,state,noisy_actions,timestep):
        token=self.state_proj(state)[:,None]
        embs=torch.cat((token,noisy_actions),dim=1)
        return timestep,embs,torch.ones(embs.shape[:2],dtype=torch.bool),torch.zeros(embs.shape[:2],dtype=torch.bool)


def extract_cpu_methods(source):
    tree=ast.parse(source)
    real=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='FlowMatchingV2')
    methods=[node for node in real.body if isinstance(node,ast.FunctionDef) and node.name in ('_tactile_delta','embed_suffix')]
    assert len(methods)==2
    cls=ast.ClassDef(name='TestV2',bases=[ast.Name(id='MinimalV1',ctx=ast.Load())],keywords=[],body=methods,decorator_list=[])
    namespace={'MinimalV1':MinimalV1,'torch':torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),
                 '<isolated-tactile-v2-methods>','exec'),namespace)
    return namespace['TestV2']


class ModelPatchTests(unittest.TestCase):
    def test_base_model_source_unchanged_and_paths_explicit(self):
        original=SOURCE.read_bytes(); patched=patch_model(original.decode())
        self.assertEqual(hashlib.sha256(original).digest(),hashlib.sha256(SOURCE.read_bytes()).digest())
        self.assertIn('tactile_history=kwargs.get("tactile_history")',patched)
        self.assertIn('tactile_delta=tactile_delta',patched)
        self.assertIn('while time >= -dt / 2:',patched)
        self.assertEqual(patched.count('def sample_actions('),original.decode().count('def sample_actions('))
        with self.assertRaisesRegex(ValueError,'unsupported'):
            patch_model('class FlowMatchingV2: pass')

    def test_zero_adapter_identity_and_trained_residual_changes_state_only(self):
        Model=extract_cpu_methods(patch_model(SOURCE.read_text()))
        m=Model(); m.config=SimpleNamespace(action_fp32=False)
        m.state_proj=torch.nn.Linear(4,8)
        m.tactile_adapter=TactileStateResidual(8)
        state=torch.ones(2,4);actions=torch.rand(2,3,8);time=torch.zeros(2)
        legacy=MinimalV1.embed_suffix(m,state,actions,time)
        no_sensor=m.embed_suffix(state,actions,time)
        self.assertTrue(torch.equal(legacy[1],no_sensor[1]))
        history=torch.ones(2,5,4,3);mask=torch.ones(2,5,4)
        age=torch.ones(2,1);valid=torch.tensor([[True],[False]])
        zero_delta=m._tactile_delta(state,history,mask,age,valid)
        self.assertTrue(torch.equal(zero_delta,torch.zeros_like(zero_delta)))
        with torch.no_grad(): m.tactile_adapter.to_state.weight.fill_(.02)
        delta=m._tactile_delta(state,history,mask,age,valid)
        changed=m.embed_suffix(state,actions,time,tactile_delta=delta)
        self.assertFalse(torch.equal(changed[1][0,0],legacy[1][0,0]))
        self.assertTrue(torch.equal(changed[1][1],legacy[1][1]))
        self.assertTrue(torch.equal(changed[1][:,1:],legacy[1][:,1:]))
        m.tactile_adapter=None
        with self.assertRaisesRegex(RuntimeError,'without a wired adapter'):
            m._tactile_delta(state,history,mask,age,valid)


if __name__=='__main__':unittest.main()
