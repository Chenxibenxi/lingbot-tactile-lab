"""Never allocate GPU or load old weights in RAM preflight tests."""
from pathlib import Path
import tempfile
import unittest
from vla_full_load_preflight import inspect_shards


class V2FullLoaderPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.dir=Path(self.temp.name)
        for n in range(1,7):
            (self.dir/f'model-{n:05d}-of-00006.safetensors').write_bytes(b'0'*100)

    def test_exact_original_six_shards_refuse_when_cpu_memory_below_lower_bound(self):
        report=inspect_shards(self.dir,available_bytes=500,margin_bytes=100)
        self.assertFalse(report['safe_to_attempt_full_loader'])
        self.assertEqual(report['total_shard_bytes'],600)
        self.assertEqual(report['status'],'refuse_full_weight_load')
        self.assertFalse(report['training_approval'])
        adequate=inspect_shards(self.dir,available_bytes=1000,margin_bytes=100)
        self.assertTrue(adequate['safe_to_attempt_full_loader'])
        self.assertEqual(adequate['status'],'ram_lower_bound_only_not_training_approval')
        self.assertFalse(adequate['training_approval'])

    def test_missing_or_symlinked_shards_cannot_fake_memory_ok(self):
        missing=self.dir/'model-00006-of-00006.safetensors'
        assert missing.resolve()==self.dir.resolve()/'model-00006-of-00006.safetensors'
        missing.unlink()
        with self.assertRaisesRegex(ValueError,'six original'):
            inspect_shards(self.dir,available_bytes=1000,margin_bytes=0)
        target=self.dir/'model-00001-of-00006.safetensors'
        missing.symlink_to(target)
        with self.assertRaisesRegex(ValueError,'symlinked'):
            inspect_shards(self.dir,available_bytes=1000,margin_bytes=0)


if __name__=='__main__':unittest.main()
