"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from mingpt.utils import CfgNode as CN
    
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, config.block_size, config.block_size))

    def forward(self, q: torch.FloatTensor, k: torch.FloatTensor, v: torch.FloatTensor) -> torch.FloatTensor:
        """TODO: implement the forward pass of a causal self attention layer
        Inputs:
            q, k, v: query/key/value tensors of size (batch_size, seq_len, head_dim)
        Outputs:
            y: attention output of size (batch_size, seq_len, head_dim)
        """
        raise NotImplementedError
    
class CausalMultiheadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # attention heads
        self.attns = [CausalSelfAttention(config) for _ in range(self.n_head)]
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)


    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """TODO: implement the forward pass of a causal multi-head attention layer. 
                 You should make use of CausalSelfAttention in your implementation!
        Inputs:
            x: previous layer representation of size (batch_size, seq_len, embedding_dim)
        Outputs:
            y: post-MHA representation of size (batch_size, seq_len, embedding_dim)
        """
        raise NotImplementedError


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # TODO: initialize the causal multi-head self-attention layer
        # self.attn = ?
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # TODO: initialize the MLP layer
        """
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(?, ?),
            c_proj  = nn.Linear(?, ?),
            act     = ?
        ))
        """
        # Initialize the MLP forward function
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x))) # MLP forward

    def forward(self, x):
        # TODO: implement the forward pass of a transformer block
        # x = x + ? # residual connection + self-attention
        # x = x + ? # residual connection + MLP
        raise NotImplementedError

class GPT(nn.Module):
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = 12
        C.n_head = 12
        C.n_embd =  768
        # these options must be filled in externally
        C.vocab_size = 50257
        C.block_size = 512
        return C

    def __init__(self, config):
        super().__init__()
        # Initialize model components
        self.block_size = config.block_size # maximum sequence length
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd), # positional encoding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # transformer blocks
            ln_f = nn.LayerNorm(config.n_embd), # final layer norm before output
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # language model head

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.shape

        # TODO: initialize the position indices
        # pos = torch.arange(0, ?, dtype=torch.long, device=device).unsqueeze(0)

        # TODO: initialize the token embeddings and the positional encodings
        # tok_emb = self.transformer.wte(?)
        # pos_emb = self.transformer.wpe(?)

        # TODO: prepare input tensor by adding the two embeddings
        # x = ? + ?

        # TODO: compute transformer layers, starting from x
        for block in self.transformer.h:
            # x = ?
            raise NotImplementedError
        
        # TODO: apply final layer norm
        # x = ?

        # TODO: compute logits from lm head
        logits = None
        # logits = ?


        # TODO: if we are given some desired targets also compute the loss
        loss = None
        if targets is not None:
            # loss = ?
            raise NotImplementedError

        return logits, loss


    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            raise NotImplementedError
