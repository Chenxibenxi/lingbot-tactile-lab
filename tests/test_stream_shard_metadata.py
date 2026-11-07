"""Six tiny synthetic shards prove header-only strictness, never load real 23GB weights."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
import torch
from safetensors.torch import save_file
from stream_shard_metadata import audit


class CheckpointHeaderTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.mapping={}
        for number in range(1,7):
            name=f'model-{number:05d}-of-00006.safetensors';key=f'layer{number}.weight'
            save_file({key:torch.ones(2,3,dtype=torch.float32)},str(self.root/name))
            self.mapping[key]=name
        self.index=self.root/'model.safetensors.index.json'
        self.index.write_text(json.dumps({'metadata':{'total_size':6*24},'weight_map':self.mapping}))

    def test_metadata_counts_dtype_index_sha_and_single_shard_not_full_model(self):
        digest=hashlib.sha256(self.index.read_bytes()).hexdigest()
        card=audit(self.root,available_bytes=2_000_000,expected_index_sha=digest,margin_bytes=1000)
        self.assertEqual(card['tensor_key_count'],6)
        self.assertEqual(card['dtype_counts'],{'F32':6})
        self.assertEqual(card['declared_tensor_bytes'],144)
        self.assertTrue(card['single_shard_ram_lower_bound_pass'])
        self.assertFalse(card['training_approval'])
        self.assertIn('not_weights_loaded',card['status'])
        low=audit(self.root,available_bytes=100,margin_bytes=200)
        self.assertFalse(low['single_shard_ram_lower_bound_pass'])
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):
            audit(self.root,available_bytes=2000000,expected_index_sha='0'*64)

    def test_extra_or_mismatched_tensor_and_declared_bytes_refuse(self):
        corrupted={'metadata':{'total_size':144},'weight_map':dict(self.mapping)}
        corrupted['weight_map']['layer1.weight']='model-00002-of-00006.safetensors'
        self.index.write_text(json.dumps(corrupted))
        with self.assertRaisesRegex(ValueError,'six unique|key mismatch'):
            audit(self.root,available_bytes=2000000)
        corrupted['weight_map']=self.mapping;corrupted['metadata']['total_size']=143
        self.index.write_text(json.dumps(corrupted))
        with self.assertRaisesRegex(ValueError,'byte count'):
            audit(self.root,available_bytes=2000000)
        corrupted['metadata']['total_size']=144
        self.index.write_text(json.dumps(corrupted))
        other=self.root/'extra.safetensors';other.symlink_to(self.root/'model-00001-of-00006.safetensors')
        with self.assertRaisesRegex(ValueError,'six unique'):
            audit(self.root,available_bytes=2000000)


if __name__=='__main__':unittest.main()
