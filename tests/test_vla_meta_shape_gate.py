"""CPU-only tiny meta modules; no real LingBot model or checkpoint values."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
import torch
from safetensors.torch import save_file
from vla_meta_shape_gate import compare_meta_model


class FakeV2Meta(torch.nn.Module):
    def __init__(self,*,width=2):
        super().__init__()
        self.config=type('Config',(),{'tactile_enabled':False})()
        self.tactile_adapter=None
        self.layers=torch.nn.Sequential(*(torch.nn.Linear(width,width) for _ in range(3)))


class MetaShapeGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        with torch.device('meta'):self.model=FakeV2Meta()
        self.mapping={}
        total=0
        for i,(key,tensor) in enumerate(self.model.state_dict().items(),start=1):
            name=f'model-{i:05d}-of-00006.safetensors'
            value=torch.ones(tuple(tensor.shape),dtype=torch.float32)
            save_file({key:value},str(self.root/name));self.mapping[key]=name
            total+=value.numel()*4
        self.index=self.root/'model.safetensors.index.json'
        self.index.write_text(json.dumps({'metadata':{'total_size':total},'weight_map':self.mapping}))
        self.digest=hashlib.sha256(self.index.read_bytes()).hexdigest()

    def test_exact_six_header_shapes_match_only_all_meta_visual_model(self):
        report=compare_meta_model(self.model,self.root,available_bytes=1000000,
                                  expected_index_sha=self.digest)
        self.assertEqual(report['matched_key_count'],6)
        self.assertTrue(report['model_is_meta_only'])
        self.assertFalse(report['values_loaded'])
        self.assertFalse(report['training_approval'])

    def test_keys_shapes_preinstalled_adapter_or_allocated_model_refuse(self):
        with torch.device('meta'):wrong=FakeV2Meta(width=3)
        with self.assertRaisesRegex(ValueError,'shape differs'):
            compare_meta_model(wrong,self.root,available_bytes=1000000,
                               expected_index_sha=self.digest)
        with torch.device('meta'):extra=FakeV2Meta()
        extra.register_buffer('bonus',torch.empty(1,device='meta'))
        with self.assertRaisesRegex(ValueError,'keys differ'):
            compare_meta_model(extra,self.root,available_bytes=1000000,
                               expected_index_sha=self.digest)
        self.model.tactile_adapter=torch.nn.Linear(1,1,device='meta')
        with self.assertRaisesRegex(ValueError,'before adapter'):
            compare_meta_model(self.model,self.root,available_bytes=1000000,
                               expected_index_sha=self.digest)
        with self.assertRaisesRegex(ValueError,'entirely meta'):
            compare_meta_model(FakeV2Meta(),self.root,available_bytes=1000000,
                               expected_index_sha=self.digest)


if __name__=='__main__':unittest.main()
