"""
Ensure that we can load huggingface/transformer GPTs into minGPT
"""

import unittest
import torch
from dataclasses import dataclass
from mingpt.model_ref import CausalMultiheadSelfAttentionRef
from mingpt.model_handout import CausalMultiheadSelfAttention
# -----------------------------------------------------------------------------

@dataclass
class MHAConfig:
    n_embd: int = 128
    n_head: int = 8
    attn_pdrop: float = 0.
    resid_pdrop: float = 0.
    block_size: int = 32

class TestMHAImpl(unittest.TestCase):

    @torch.no_grad()
    def test_mha(self):
        config = MHAConfig()

        stu_impl = CausalMultiheadSelfAttention(config)
        ref_impl = CausalMultiheadSelfAttentionRef(config)

        for stu_param, ref_param in zip(stu_impl.parameters(), ref_impl.parameters()):
            ref_param.data = stu_param.data
        
        stu_impl.eval()
        ref_impl.eval()

        batch = torch.randn(4, config.block_size, config.n_embd)
        stu_out = stu_impl(batch)
        ref_out = ref_impl(batch)
        
        self.assertTrue(torch.allclose(stu_out, ref_out))

if __name__ == '__main__':
    unittest.main()
