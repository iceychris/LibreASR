from functools import partial, lru_cache

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F

import numpy as np

from fastai.torch_core import Module

from IPython.core.debugger import set_trace


class LinearTransformerEncoder(Module):
    def __init__(
        self,
        feature_sz,
        hidden_sz,
        out_sz,
        dropout=0.01,
        dropout_input=0.0,
        dropout_inner=0.0,
        num_layers=8,
        trace=True,
        device="cuda:0",
        rnn_type="LSTM",
        norm="bn",
        attention=False,
        use_tmp_state_pcent=0.9,
        reversible=False,
        bidirectional=False,
        **kwargs,
    ):
        self.num_layers = num_layers
        self.input_norm = nn.LayerNorm(feature_sz)
        from fast_transformers.builders import (
            TransformerEncoderBuilder,
            RecurrentEncoderBuilder,
        )
        from fast_transformers.utils import make_mirror
        from fast_transformers.masking import TriangularCausalMask

        n_heads = 16
        params = dict(
            attention_type="causal-linear",
            n_layers=num_layers,
            n_heads=n_heads,
            feed_forward_dimensions=hidden_sz,
            query_dimensions=hidden_sz // n_heads,
            value_dimensions=hidden_sz // n_heads,
        )
        self.transformer = TransformerEncoderBuilder.from_dictionary(params).get()
        self.recurrent_transformer = RecurrentEncoderBuilder.from_dictionary(
            params
        ).get()
        make_mirror(self.transformer, self.recurrent_transformer)
        self.tcm = TriangularCausalMask

        # input projection
        self.ff1 = nn.Linear(feature_sz, hidden_sz)

        # output projection
        if not hidden_sz == out_sz:
            self.ff2 = nn.Linear(hidden_sz, out_sz)
        else:
            self.ff2 = nn.Sequential()

    def param_groups(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x, state=None, lengths=None, return_state=False):
        x = x.reshape((x.size(0), x.size(1), -1))
        x = self.input_norm(x)
        x = self.ff1(x)
        if state is None:
            x = self.transformer(x, attn_mask=self.tcm(x.size(0), device=x.device))
        else:
            raise NotImplementedError("recurrent transformer not implemented")
        x = self.ff2(x)
        if return_state:
            return x, state
        return x
