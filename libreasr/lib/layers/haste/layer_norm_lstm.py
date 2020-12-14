# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modifications by @iceychris:
# - run CPU-only (for inference)
#

"""Layer Normalized Long Short-Term Memory"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_rnn import BaseRNN


__all__ = ["LayerNormLSTM"]


# @torch.jit.script
def LayerNormLSTMScript(
    training: bool,
    zoneout_prob: float,
    input,
    h0,
    c0,
    kernel,
    recurrent_kernel,
    bias,
    gamma,
    gamma_h,
    beta_h,
    zoneout_mask,
):
    time_steps = input.shape[0]
    batch_size = input.shape[1]
    hidden_size = recurrent_kernel.shape[0]

    h = [h0]
    c = [c0]
    Wx = F.layer_norm(input @ kernel, (hidden_size * 4,), weight=gamma[0])
    for t in range(time_steps):
        v = (
            F.layer_norm(h[t] @ recurrent_kernel, (hidden_size * 4,), weight=gamma[1])
            + Wx[t]
            + bias
        )
        i, g, f, o = torch.chunk(v, 4, 1)
        i = torch.sigmoid(i)
        g = torch.tanh(g)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        c.append(f * c[t] + i * g)
        h.append(
            o
            * torch.tanh(
                F.layer_norm(c[-1], (hidden_size,), weight=gamma_h, bias=beta_h)
            )
        )
        if zoneout_prob:
            if training:
                h[-1] = (h[-1] - h[-2]) * zoneout_mask[t] + h[-2]
            else:
                h[-1] = zoneout_prob * h[-2] + (1 - zoneout_prob) * h[-1]
    h = torch.stack(h)
    c = torch.stack(c)
    return h, c


class LayerNormLSTM(BaseRNN):
    """
  Layer Normalized Long Short-Term Memory layer.

  This LSTM layer applies layer normalization to the input, recurrent, and
  output activations of a standard LSTM. The implementation is fused and
  GPU-accelerated. DropConnect and Zoneout regularization are built-in, and
  this layer allows setting a non-zero initial forget gate bias.

  Details about the exact function this layer implements can be found at
  https://github.com/lmnt-com/haste/issues/1.

  See [\_\_init\_\_](#__init__) and [forward](#forward) for usage.
  """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_first=False,
        forget_bias=1.0,
        dropout=0.0,
        zoneout=0.0,
        return_state_sequence=False,
    ):
        """
    Initialize the parameters of the LSTM layer.

    Arguments:
      input_size: int, the feature dimension of the input.
      hidden_size: int, the feature dimension of the output.
      batch_first: (optional) bool, if `True`, then the input and output
        tensors are provided as `(batch, seq, feature)`.
      forget_bias: (optional) float, sets the initial bias of the forget gate
        for this LSTM cell.
      dropout: (optional) float, sets the dropout rate for DropConnect
        regularization on the recurrent matrix.
      zoneout: (optional) float, sets the zoneout rate for Zoneout
        regularization.
      return_state_sequence: (optional) bool, if `True`, the forward pass will
        return the entire state sequence instead of just the final state. Note
        that if the input is a padded sequence, the returned state will also
        be a padded sequence.

    Variables:
      kernel: the input projection weight matrix. Dimensions
        (input_size, hidden_size * 4) with `i,g,f,o` gate layout. Initialized
        with Xavier uniform initialization.
      recurrent_kernel: the recurrent projection weight matrix. Dimensions
        (hidden_size, hidden_size * 4) with `i,g,f,o` gate layout. Initialized
        with orthogonal initialization.
      bias: the projection bias vector. Dimensions (hidden_size * 4) with
        `i,g,f,o` gate layout. The forget gate biases are initialized to
        `forget_bias` and the rest are zeros.
      gamma: the input and recurrent normalization gain. Dimensions
        (2, hidden_size * 4) with `gamma[0]` specifying the input gain and
        `gamma[1]` specifying the recurrent gain. Initialized to ones.
      gamma_h: the output normalization gain. Dimensions (hidden_size).
        Initialized to ones.
      beta_h: the output normalization bias. Dimensions (hidden_size).
        Initialized to zeros.
    """
        super().__init__(
            input_size, hidden_size, batch_first, zoneout, return_state_sequence
        )

        if dropout < 0 or dropout > 1:
            raise ValueError("LayerNormLSTM: dropout must be in [0.0, 1.0]")
        if zoneout < 0 or zoneout > 1:
            raise ValueError("LayerNormLSTM: zoneout must be in [0.0, 1.0]")

        self.forget_bias = forget_bias
        self.dropout = dropout

        self.kernel = nn.Parameter(torch.empty(input_size, hidden_size * 4))
        self.recurrent_kernel = nn.Parameter(torch.empty(hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.empty(hidden_size * 4))
        self.gamma = nn.Parameter(torch.empty(2, hidden_size * 4))
        self.gamma_h = nn.Parameter(torch.empty(hidden_size))
        self.beta_h = nn.Parameter(torch.empty(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        """Resets this layer's parameters to their initial values."""
        hidden_size = self.hidden_size
        for i in range(4):
            nn.init.xavier_uniform_(
                self.kernel[:, i * hidden_size : (i + 1) * hidden_size]
            )
            nn.init.orthogonal_(
                self.recurrent_kernel[:, i * hidden_size : (i + 1) * hidden_size]
            )
        nn.init.zeros_(self.bias)
        nn.init.constant_(
            self.bias[hidden_size * 2 : hidden_size * 3], self.forget_bias
        )
        nn.init.ones_(self.gamma)
        nn.init.ones_(self.gamma_h)
        nn.init.zeros_(self.beta_h)

    def forward(self, input, state=None, lengths=None):
        """
    Runs a forward pass of the LSTM layer.

    Arguments:
      input: Tensor, a batch of input sequences to pass through the LSTM.
        Dimensions (seq_len, batch_size, input_size) if `batch_first` is
        `False`, otherwise (batch_size, seq_len, input_size).
      lengths: (optional) Tensor, list of sequence lengths for each batch
        element. Dimension (batch_size). This argument may be omitted if
        all batch elements are unpadded and have the same sequence length.

    Returns:
      output: Tensor, the output of the LSTM layer. Dimensions
        (seq_len, batch_size, hidden_size) if `batch_first` is `False` (default)
        or (batch_size, seq_len, hidden_size) if `batch_first` is `True`. Note
        that if `lengths` was specified, the `output` tensor will not be
        masked. It's the caller's responsibility to either not use the invalid
        entries or to mask them out before using them.
      (h_n, c_n): the hidden and cell states, respectively, for the last
        sequence item. Dimensions (1, batch_size, hidden_size).
    """
        input = self._permute(input)
        state_shape = [1, input.shape[1], self.hidden_size]
        state_shape = (state_shape, state_shape)
        h0, c0 = self._get_state(input, state, state_shape)
        h, c = self._impl(input, (h0[0], c0[0]), self._get_zoneout_mask(input))
        state = self._get_final_state((h, c), lengths)
        output = self._permute(h[1:])
        return output, state

    def _impl(self, input, state, zoneout_mask):
        return LayerNormLSTMScript(
            self.training,
            self.zoneout,
            input.contiguous(),
            state[0].contiguous(),
            state[1].contiguous(),
            self.kernel.contiguous(),
            F.dropout(self.recurrent_kernel, self.dropout, self.training).contiguous(),
            self.bias.contiguous(),
            self.gamma.contiguous(),
            self.gamma_h.contiguous(),
            self.beta_h.contiguous(),
            zoneout_mask.contiguous(),
        )
