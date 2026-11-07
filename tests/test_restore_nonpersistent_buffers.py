import unittest
import torch
from restore_nonpersistent_buffers import restore_nonpersistent_buffers
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionRotaryEmbedding,Qwen3VLTextRotaryEmbedding
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
from transformers import Qwen2Config

class RestoreTests(unittest.TestCase):
    def test_meta_rotary_matches_real_constructors_and_repairs_alias(self):
        factories=[lambda:Qwen3VLVisionRotaryEmbedding(32),
                   lambda:Qwen3VLTextRotaryEmbedding(Qwen3VLTextConfig(rope_scaling={'rope_type':'default'})),
                   lambda:Qwen2RotaryEmbedding(Qwen2Config())]
        for factory in factories:
            real=factory().to(dtype=torch.bfloat16)
            with torch.device('meta'):empty=factory()
            empty.to(dtype=torch.bfloat16).to_empty(device='cpu')
            empty.inv_freq.fill_(float('nan'))
            restore_nonpersistent_buffers(empty)
            self.assertTrue(torch.equal(real.inv_freq,empty.inv_freq))
            if hasattr(empty,'original_inv_freq'):
                self.assertIs(empty.original_inv_freq,empty.inv_freq)
    def test_unknown_buffer_refused(self):
        m=torch.nn.Module();m.register_buffer('unknown',torch.zeros(2),persistent=False)
        with self.assertRaisesRegex(ValueError,'unhandled'):restore_nonpersistent_buffers(m)
