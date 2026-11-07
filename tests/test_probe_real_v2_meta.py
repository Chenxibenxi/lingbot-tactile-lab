"""Source/hash and visual-configuration gates; never imports large patched model."""
import unittest
from unittest.mock import patch
from probe_real_v2_meta import verify_sources, build_old_config, TOKENIZER,REFERENCE_INDEX_SHA


class RealMetaPreflightTests(unittest.TestCase):
    def test_all_external_visual_sources_pin_and_local_qwen_config_is_explicit(self):
        evidence=verify_sources()
        self.assertEqual(evidence['checkpoint_index_sha256'],REFERENCE_INDEX_SHA)
        self.assertEqual(len(evidence),4)
        config=build_old_config()
        self.assertEqual(config.tokenizer_path,str(TOKENIZER))
        self.assertFalse(config.tactile_enabled)
        self.assertEqual(config.chunk_size,50)
        self.assertEqual(config.max_state_dim,55)

    def test_original_visual_yaml_or_tokenizer_sha_tamper_stops_before_gpu(self):
        with patch('probe_real_v2_meta.TRAIN_SHA','0'*64):
            with self.assertRaisesRegex(ValueError,'source SHA changed'):
                verify_sources()
        with patch('probe_real_v2_meta.TOKENIZER_SHA','0'*64):
            with self.assertRaisesRegex(ValueError,'source SHA changed'):
                verify_sources()


if __name__=='__main__':unittest.main()
