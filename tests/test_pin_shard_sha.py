"""Whole-file hashes of tiny six-shard synthetic reference only."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import torch
from safetensors.torch import save_file
from pin_shard_sha import pin


class ShardHashTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);weight_map={}
        for number in range(1,7):
            name=f'model-{number:05d}-of-00006.safetensors'
            key=f'key{number}'
            save_file({key:torch.ones(2,3,dtype=torch.float32)},str(self.root/name))
            weight_map[key]=name
        self.index=self.root/'model.safetensors.index.json'
        self.index.write_text(json.dumps({'metadata':{'total_size':6*24},'weight_map':weight_map}))
    def test_manifest_pins_every_synthetic_shard_without_model_values(self):
        result=pin(self.root,available_bytes=1_000_000,expected_index_sha=hashlib.sha256(self.index.read_bytes()).hexdigest())
        self.assertEqual(result['tensor_key_count'],6)
        self.assertEqual(len(result['shard_files']),6)
        self.assertFalse(result['values_loaded_into_model'])
        self.assertFalse(result['training_approval'])
        for name,entry in result['shard_files'].items():
            data=(self.root/name).read_bytes()
            self.assertEqual(entry['sha256'],hashlib.sha256(data).hexdigest())
            self.assertEqual(entry['file_bytes'],len(data))


if __name__=='__main__':unittest.main()
