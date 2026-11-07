"""CPU fake-server route test for strict isolated deployment source generator."""
import ast
import hashlib
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from patch_server_tactile_v2 import SOURCE,patch_server
from server_tactile_bridge import prepare_input,model_tactile_kwargs
from tactile_rpc import encode_tactile_window
from tests.test_tactile_rpc import window


class FakeTransform:
    def apply(self,observation,policy_eval=False):
        assert policy_eval and not any(k.startswith('observation.tactile.') for k in observation)
        return {'state':torch.as_tensor(observation['observation.state']),
                'images':torch.zeros(3,3,32,32)}


def extracted_prepare(source):
    tree=ast.parse(source)
    original=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LingbotVLAv2Server')
    method,=[n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='_prepare_model_input']
    node=ast.ClassDef(name='TinyServer',bases=[],keywords=[],body=[method],decorator_list=[])
    namespace={'torch':torch,'np':np,'prepare_input':prepare_input}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),namespace)
    server=namespace['TinyServer']()
    server.vla=SimpleNamespace(feature_transform=FakeTransform())
    server.resize_image=lambda msg:None
    server.use_bf16=False
    return server


class ServerPreviewSourceTests(unittest.TestCase):
    def test_visual_identity_and_fail_closed_incoming_rpc(self):
        original=SOURCE.read_bytes();patched=patch_server(original.decode())
        self.assertEqual(hashlib.sha256(original).digest(),hashlib.sha256(SOURCE.read_bytes()).digest())
        old=extracted_prepare(original.decode());preview=extracted_prepare(patched)
        raw={'observation.state':np.zeros(14,dtype=np.float32)}
        before=old._prepare_model_input(dict(raw))
        after=preview._prepare_model_input(dict(raw))
        self.assertEqual(set(before),set(after))
        self.assertTrue(all(torch.equal(before[k],after[k]) for k in before))
        populated=dict(raw,**encode_tactile_window(window()))
        with self.assertRaisesRegex(RuntimeError,'disabled'):
            preview._prepare_model_input(dict(populated))
        preview._tactile_capability_ready=True  # synthetic route fixture, never enabled by preview itself
        ready=preview._prepare_model_input(dict(populated))
        self.assertTrue(ready['tactile_valid'][0])
        self.assertEqual(tuple(ready['tactile_history'].shape),(5,4,3))
        kwargs=model_tactile_kwargs(ready,device='cpu',dtype=torch.float32,batched=False)
        self.assertEqual(tuple(kwargs['tactile_history'].shape),(5,4,3))
        with self.assertRaisesRegex(ValueError,'partial'):
            preview._prepare_model_input(dict(raw,**{'observation.tactile.history':np.zeros((5,4,3))}))
        fallback=preview._prepare_model_input(dict(raw))
        self.assertFalse(fallback['tactile_valid'][0])
        self.assertEqual(fallback['tactile_mask'].sum().item(),0)

    def test_all_original_sampling_call_sites_receive_bridge_only_in_preview(self):
        base=SOURCE.read_text();patched=patch_server(base)
        self.assertEqual(patched.count('**model_tactile_kwargs('),4)
        self.assertIn('batched=False',patched)
        self.assertEqual(patched.count('batched=True'),3)
        self.assertEqual(patched.count('def sample_actions_batch('),base.count('def sample_actions_batch('))
        self.assertEqual(patched.count('def _infer_batch('),base.count('def _infer_batch('))
        self.assertNotIn('tactile_v1',patched)  # no capability claim or approved adapter in source preview
        with self.assertRaisesRegex(ValueError,'source anchor'):
            patch_server('class LingbotVLAv2Server: pass')


if __name__=='__main__':unittest.main()
