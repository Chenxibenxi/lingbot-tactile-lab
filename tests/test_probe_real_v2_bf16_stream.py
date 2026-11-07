"""CPU-only source and meta→BF16→empty→six-shard strict-copy checks."""
import hashlib
import io
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from safetensors.torch import save_file
from probe_real_v2_bf16_stream import LEASE_OWNER,_lease_parent_pids,require_gpu_window,verify_fixed_inputs,count_unfrozen_base_parameters,compare_visual_and_zero
from strict_stream_visual_weights import stream_into_allocated_model


class Bf16RealStreamGates(unittest.TestCase):
    def test_fixed_noise_is_cloned_for_visual_vs_zero_sample_actions(self):
        class TinyFlow(torch.nn.Module):
            def __init__(self):
                super().__init__();self.state_proj=torch.nn.Linear(55,2);self.nonzero=False
            def sample_actions(self,images,masks,tokens,token_masks,state,*,noise,image_grid_thw,**kwargs):
                noise.add_(1)  # original denoiser also mutates its noise argument
                return noise+(1 if self.nonzero and kwargs.get('tactile_history') is not None else 0)
        root=torch.nn.Module();root.model=TinyFlow()
        cfg=SimpleNamespace(n_action_steps=2,max_action_dim=3,num_steps=2)
        obs={'images':torch.zeros(3,2,4),'img_masks':torch.ones(3,dtype=torch.bool),
             'lang_tokens':torch.ones(4,dtype=torch.long),'lang_masks':torch.ones(4,dtype=torch.bool),
             'state':torch.zeros(55),'image_grid_thw':torch.ones(3,3,dtype=torch.long)}
        tactile={'tactile_history':torch.zeros(1,5,4,3),'tactile_mask':torch.ones(1,5,4),
                 'tactile_age_ms':torch.zeros(1),'tactile_valid':torch.ones(1,dtype=torch.bool)}
        report=compare_visual_and_zero(root,obs,tactile,cfg,{'trial_role':'train'})
        self.assertTrue(report['bitwise_equal'])
        self.assertEqual(report['action_shape'],[1,2,3])
        root.model.nonzero=True
        with self.assertRaisesRegex(RuntimeError,'max_abs_delta=1'):
            compare_visual_and_zero(root,obs,tactile,cfg,{'trial_role':'train'})

    def test_wrapper_only_inner_adapter_stays_trainable(self):
        wrapped=torch.nn.Module()
        wrapped.model=torch.nn.Module()
        wrapped.model.visual=torch.nn.Linear(2,2)
        wrapped.model.tactile_adapter=torch.nn.Linear(2,2)
        for p in wrapped.parameters():p.requires_grad_(False)
        for p in wrapped.model.tactile_adapter.parameters():p.requires_grad_(True)
        self.assertEqual(count_unfrozen_base_parameters(wrapped),0)
        wrapped.model.visual.weight.requires_grad_(True)
        self.assertEqual(count_unfrozen_base_parameters(wrapped),1)

    def _window(self,info,*,compute='',loaded=False,service='strata'):
        with (patch('probe_real_v2_bf16_stream.os.uname',return_value=SimpleNamespace(nodename='admin001-MS-7D90')),
              patch('probe_real_v2_bf16_stream._lease_parent_pids',return_value={333,777}),
              patch('probe_real_v2_bf16_stream.subprocess.run',side_effect=[
                  SimpleNamespace(stdout=json.dumps(info)),SimpleNamespace(stdout=compute)]),
              patch('probe_real_v2_bf16_stream.request.urlopen',return_value=io.BytesIO(
                  json.dumps({'loaded':loaded,'service':service}).encode()))):
            return require_gpu_window()

    def test_exclusive_parent_lease_waiters_compute_and_strata_health_are_hard_gates(self):
        info={'state':'held','holder':LEASE_OWNER,'lease':{'pid':777},
              'pid_alive':True,'remaining_s':700,'waiting':[]}
        self.assertTrue(self._window(info))
        with self.assertRaisesRegex(RuntimeError,'no other GPU waiter'):
            self._window({**info,'waiting':[{'owner':'secretary-digest'}]})
        with self.assertRaisesRegex(RuntimeError,'compute tasks'):
            self._window(info,compute='4942\n')
        with self.assertRaisesRegex(RuntimeError,'Strata already loaded'):
            self._window(info,loaded=True)
        with self.assertRaisesRegex(RuntimeError,'health ambiguous'):
            self._window(info,service='unknown')
        with self.assertRaisesRegex(RuntimeError,'ancestor-shell lease'):
            self._window({**info,'lease':{'pid':1}})
        with self.assertRaisesRegex(RuntimeError,'ancestor-shell lease'):
            self._window({**info,'pid_alive':False})
        with self.assertRaisesRegex(RuntimeError,'ancestor-shell lease'):
            self._window({**info,'remaining_s':599})

    def test_timeout_wrapper_keeps_lease_shell_in_short_ancestor_chain(self):
        child=subprocess.run(['timeout','4s',sys.executable,'-B','-c',
                'from probe_real_v2_bf16_stream import _lease_parent_pids; print(sorted(_lease_parent_pids()))'],
                check=True,capture_output=True,text=True,timeout=6)
        self.assertIn(os.getpid(),json.loads(child.stdout))
        self.assertIn(os.getppid(),_lease_parent_pids())

    def test_only_fixed_original_source_preview_and_manifest_allowed(self):
        anchors,preview,manifest=verify_fixed_inputs(Path(__file__).resolve().parents[1])
        self.assertEqual(anchors['checkpoint_index_sha256'],manifest['checkpoint_index_sha256'])
        self.assertEqual(len(manifest['shard_files']),6)
        self.assertTrue(preview.is_file())
        with patch('probe_real_v2_bf16_stream.PREVIEW_SHA','0'*64):
            with self.assertRaisesRegex(ValueError,'source missing or SHA mismatch'):
                verify_fixed_inputs(Path(__file__).resolve().parents[1])

    def test_realistic_allocation_order_with_six_synthetic_bf16_copies_on_cpu(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);mapping={};total=0
            with torch.device('meta'):
                model=torch.nn.Sequential(*(torch.nn.Linear(2,2) for _ in range(3)))
            original_keys=set(model.state_dict())
            for i,(key,tensor) in enumerate(model.state_dict().items(),1):
                name=f'model-{i:05d}-of-00006.safetensors'
                value=torch.full(tuple(tensor.shape),float(i),dtype=torch.float32)
                save_file({key:value},str(path/name));mapping[key]=name;total+=value.numel()*4
            index=path/'model.safetensors.index.json'
            index.write_text(json.dumps({'metadata':{'total_size':total},'weight_map':mapping}))
            digest=hashlib.sha256(index.read_bytes()).hexdigest()
            model.to(dtype=torch.bfloat16)
            self.assertTrue(all(t.is_meta for t in model.state_dict().values()))
            model.to_empty(device=torch.device('cpu'))
            self.assertTrue(all(not t.is_meta and t.dtype==torch.bfloat16
                                for t in model.state_dict().values()))
            proof=stream_into_allocated_model(model,path,available_bytes=1_000_000,
                                             expected_index_sha=digest,margin_bytes=100)
            self.assertEqual(set(model.state_dict()),original_keys)
            self.assertEqual(proof.copied_key_count,6)
            self.assertEqual(proof.destination_dtype,'torch.bfloat16')
            for key,tensor in model.state_dict().items():
                self.assertTrue(torch.all(tensor==list(mapping).index(key)+1))


if __name__=='__main__':unittest.main()
