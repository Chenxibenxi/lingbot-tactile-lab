"""Tiny synthetic strict shard copy; never read original FP32 23GB model."""
from pathlib import Path
import hashlib
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from safetensors.torch import save_file
from safetensors import safe_open as real_safe_open
from model_attach_contract import attach_visual_identity_adapter
from pin_shard_sha import pin
from strict_stream_visual_weights import stream_into_allocated_model


class ToyPatched(torch.nn.Module):
    def __init__(self,*,meta=False):
        super().__init__()
        self.config=SimpleNamespace(tactile_enabled=False)
        self.state_proj=torch.nn.Linear(2,2)
        self.action_in=torch.nn.Linear(2,2)
        self.action_out=torch.nn.Linear(2,2)
        self.tactile_adapter=None
    def _tactile_delta(self,*args):pass
    def sample_actions(self,*args):pass
    def predict_velocity(self,*args):pass


class StrictShardedLoadTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.model=ToyPatched()
        self.expected={};mapping={}
        size=0
        for i,(key,value) in enumerate(self.model.state_dict().items(),start=1):
            name=f'model-{i:05d}-of-00006.safetensors';tensor=torch.full(tuple(value.shape),float(i))
            save_file({key:tensor},str(self.root/name));self.expected[key]=tensor;mapping[key]=name
            size+=tensor.numel()*4
        self.index=self.root/'model.safetensors.index.json'
        self.index.write_text(json.dumps({'metadata':{'total_size':size},'weight_map':mapping}))
        self.digest=hashlib.sha256(self.index.read_bytes()).hexdigest()

    def test_all_six_shards_copy_exactly_and_are_valid_for_zero_adapter_attach(self):
        with torch.no_grad():
            for p in self.model.parameters():p.zero_()
        proof=stream_into_allocated_model(self.model,self.root,available_bytes=1_000_000,
                                         expected_index_sha=self.digest,margin_bytes=100)
        self.assertEqual(proof.copied_key_count,6)
        self.assertEqual((proof.missing_keys,proof.unexpected_keys),((),()))
        self.assertFalse(proof.training_approval)
        for key,value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value,self.expected[key]),key)
        attached=attach_visual_identity_adapter(self.model,proof)
        self.assertFalse(attached['training_approval'])
        self.assertTrue(self.model.config.tactile_enabled)

    def test_fp32_shards_cast_to_bfloat16_destination_without_overwriting_source(self):
        self.model.to(dtype=torch.bfloat16)
        manifest=pin(self.root,available_bytes=1_000_000,expected_index_sha=self.digest)
        proof=stream_into_allocated_model(self.model,self.root,available_bytes=1_000_000,
                                         expected_index_sha=self.digest,margin_bytes=100,
                                         expected_shard_manifest=manifest)
        self.assertEqual(proof.destination_dtype,'torch.bfloat16')
        self.assertEqual(proof.source_dtype,'F32')
        for key,value in self.model.state_dict().items():
            self.assertEqual(value.dtype,torch.bfloat16)
            self.assertTrue(torch.equal(value,self.expected[key].to(dtype=torch.bfloat16)))

    def test_wrong_source_shape_or_low_memory_refuses_before_mutating_model(self):
        before={k:v.clone() for k,v in self.model.state_dict().items()}
        with self.assertRaisesRegex(ValueError,'one-shard host memory gate'):
            stream_into_allocated_model(self.model,self.root,available_bytes=10,
                                        expected_index_sha=self.digest,margin_bytes=100)
        key='state_proj.weight';p=self.root/'model-00001-of-00006.safetensors'
        save_file({key:torch.ones(3,2)},str(p))
        index=json.loads(self.index.read_text());index['metadata']['total_size']+=8
        self.index.write_text(json.dumps(index))
        new_digest=hashlib.sha256(self.index.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError,'shape mismatch'):
            stream_into_allocated_model(self.model,self.root,available_bytes=1_000_000,
                                        expected_index_sha=new_digest,margin_bytes=100)
        for key,value in self.model.state_dict().items():self.assertTrue(torch.equal(value,before[key]))

    def test_partial_shard_io_error_taints_model_and_refuses_adapter(self):
        class FailingReader:
            def __init__(self,path):self.path=str(path);self.context=None
            def __enter__(self):
                self.context=real_safe_open(self.path,framework='pt',device='cpu')
                return self
            def __exit__(self,*args):return self.context.__exit__(*args)
            def keys(self):return self.context.keys()
            def get_slice(self,key):return self.context.get_slice(key)
            def get_tensor(self,key):
                if key=='state_proj.bias':raise OSError('synthetic second shard failure')
                return self.context.get_tensor(key)
        with patch('strict_stream_visual_weights.safe_open',side_effect=lambda path,**kw:FailingReader(path)):
            with self.assertRaisesRegex(OSError,'second shard'):
                stream_into_allocated_model(self.model,self.root,available_bytes=1_000_000,
                                            expected_index_sha=self.digest,margin_bytes=100)
        self.assertTrue(self.model._strict_stream_partial_failure)
        with self.assertRaisesRegex(ValueError,'strictly loaded'):
            attach_visual_identity_adapter(self.model,SimpleNamespace(missing_keys=[],unexpected_keys=[]))

    def test_pinned_shard_hash_mismatch_or_missing_manifest_taints_destination(self):
        manifest=pin(self.root,available_bytes=1_000_000,expected_index_sha=self.digest)
        manifest['shard_files']['model-00001-of-00006.safetensors']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'shard SHA'):
            stream_into_allocated_model(self.model,self.root,available_bytes=1_000_000,
                                        expected_index_sha=self.digest,margin_bytes=100,
                                        expected_shard_manifest=manifest)
        self.assertTrue(self.model._strict_stream_partial_failure)
        with patch('strict_stream_visual_weights.REFERENCE_INDEX_SHA',self.digest):
            with self.assertRaisesRegex(ValueError,'require pinned'):
                stream_into_allocated_model(ToyPatched(),self.root,available_bytes=1_000_000,
                                            expected_index_sha=self.digest,margin_bytes=100)

    def test_meta_or_preattached_model_is_not_a_load_destination(self):
        with torch.device('meta'):meta=ToyPatched()
        with self.assertRaisesRegex(ValueError,'allocate model'):
            stream_into_allocated_model(meta,self.root,available_bytes=1_000_000,
                                        expected_index_sha=self.digest,margin_bytes=100)
        self.model.tactile_adapter=torch.nn.Linear(1,1)
        with self.assertRaisesRegex(ValueError,'pre-attached'):
            stream_into_allocated_model(self.model,self.root,available_bytes=1_000_000,
                                        expected_index_sha=self.digest,margin_bytes=100)


if __name__=='__main__':unittest.main()
