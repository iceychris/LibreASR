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

"""Gated Recurrent Unit"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_rnn import BaseRNN


__all__ = ["GRU"]


# @torch.jit.script
def GRUScript(
    training: bool,
    zoneout_prob: float,
    input,
    h0,
    kernel,
    recurrent_kernel,
    bias,
    recurrent_bias,
    zoneout_mask,
):
    time_steps = input.shape[0]
    batch_size = input.shape[1]
    hidden_size = recurrent_kernel.shape[0]

    h = [h0]
    Wx = input @ kernel + bias
    for t in range(time_steps):
        Rh = h[t] @ recurrent_kernel + recurrent_bias
        vx = torch.chunk(Wx[t], 3, 1)
        vh = torch.chunk(Rh, 3, 1)

        z = torch.sigmoid(vx[0] + vh[0])
        r = torch.sigmoid(vx[1] + vh[1])
        g = torch.tanh(vx[2] + r * vh[2])

        h.append(z * h[t] + (1 - z) * g)
        if zoneout_prob:
            if training:
                h[-1] = (h[-1] - h[-2]) * zoneout_mask[t] + h[-2]
            else:
                h[-1] = zoneout_prob * h[-2] + (1 - zoneout_prob) * h[-1]

    h = torch.stack(h)
    return h


class GRU(BaseRNN):
    """
  Gated Recurrent Unit layer.

  This GRU layer offers a fused, GPU-accelerated PyTorch op for inference
  and training. There are two commonly-used variants of GRU cells. This one
  implements 1406.1078v1 which applies the reset gate to the hidden state
  after matrix multiplication. cuDNN also implements this variant. The other
  variant, 1406.1078v3, applies the reset gate before matrix multiplication
  and is currently unsupported.

  This layer has built-in support for DropConnect and Zoneout, which are
  both techniques used to regularize RNNs.

  See [\_\_init\_\_](#__init__) and [forward](#forward) for usage.
  See [from_native_weights](#from_native_weights) and
  [to_native_weights](#to_native_weights) for compatibility with PyTorch GRUs.
  """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_first=False,
        dropout=0.0,
        zoneout=0.0,
        return_state_sequence=False,
    ):
        """
    Initialize the parameters of the GRU layer.

    Arguments:
      input_size: int, the feature dimension of the input.
      hidden_size: int, the feature dimension of the output.
      batch_first: (optional) bool, if `True`, then the input and output
        tensors are provided as `(batch, seq, feature)`.
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
        (input_size, hidden_size * 3) with `z,r,h` gate layout. Initialized
        with Xavier uniform initialization.
      recurrent_kernel: the recurrent projection weight matrix. Dimensions
        (hidden_size, hidden_size * 3) with `z,r,h` gate layout. Initialized
        with orthogonal initialization.
      bias: the input projection bias vector. Dimensions (hidden_size * 3) with
        `z,r,h` gate layout. Initialized to zeros.
      recurrent_bias: the recurrent projection bias vector. Dimensions
        (hidden_size * 3) with `z,r,h` gate layout. Initialized to zeros.
    """
        super().__init__(
            input_size, hidden_size, batch_first, zoneout, return_state_sequence
        )

        if dropout < 0 or dropout > 1:
            raise ValueError("GRU: dropout must be in [0.0, 1.0]")
        if zoneout < 0 or zoneout > 1:
            raise ValueError("GRU: zoneout must be in [0.0, 1.0]")

        self.dropout = dropout

        self.kernel = nn.Parameter(torch.empty(input_size, hidden_size * 3))
        self.recurrent_kernel = nn.Parameter(torch.empty(hidden_size, hidden_size * 3))
        self.bias = nn.Parameter(torch.empty(hidden_size * 3))
        self.recurrent_bias = nn.Parameter(torch.empty(hidden_size * 3))
        self.reset_parameters()

    def to_native_weights(self):
        """
    Converts Haste GRU weights to native PyTorch GRU weights.

    Returns:
      weight_ih_l0: Parameter, the input-hidden weights of the GRU layer.
      weight_hh_l0: Parameter, the hidden-hidden weights of the GRU layer.
      bias_ih_l0: Parameter, the input-hidden bias of the GRU layer.
      bias_hh_l0: Parameter, the hidden-hidden bias of the GRU layer.
    """

        def reorder_weights(w):
            z, r, n = torch.chunk(w, 3, dim=-1)
            return torch.cat([r, z, n], dim=-1)

        kernel = reorder_weights(self.kernel).permute(1, 0).contiguous()
        recurrent_kernel = (
            reorder_weights(self.recurrent_kernel).permute(1, 0).contiguous()
        )
        bias1 = reorder_weights(self.bias).contiguous()
        bias2 = reorder_weights(self.recurrent_bias).contiguous()

        kernel = torch.nn.Parameter(kernel)
        recurrent_kernel = torch.nn.Parameter(recurrent_kernel)
        bias1 = torch.nn.Parameter(bias1)
        bias2 = torch.nn.Parameter(bias2)
        return kernel, recurrent_kernel, bias1, bias2

    def from_native_weights(self, weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0):
        """
    Copies and converts the provided PyTorch GRU weights into this layer.

    Arguments:
      weight_ih_l0: Parameter, the input-hidden weights of the PyTorch GRU layer.
      weight_hh_l0: Parameter, the hidden-hidden weights of the PyTorch GRU layer.
      bias_ih_l0: Parameter, the input-hidden bias of the PyTorch GRU layer.
      bias_hh_l0: Parameter, the hidden-hidden bias of the PyTorch GRU layer.
    """

        def reorder_weights(w):
            r, z, n = torch.chunk(w, 3, axis=-1)
            return torch.cat([z, r, n], dim=-1)

        kernel = reorder_weights(weight_ih_l0.permute(1, 0)).contiguous()
        recurrent_kernel = reorder_weights(weight_hh_l0.permute(1, 0)).contiguous()
        bias = reorder_weights(bias_ih_l0).contiguous()
        recurrent_bias = reorder_weights(bias_hh_l0).contiguous()

        self.kernel = nn.Parameter(kernel)
        self.recurrent_kernel = nn.Parameter(recurrent_kernel)
        self.bias = nn.Parameter(bias)
        self.recurrent_bias = nn.Parameter(recurrent_bias)

    def reset_parameters(self):
        """Resets this layer's parameters to their initial values."""
        hidden_size = self.hidden_size
        for i in range(3):
            nn.init.xavier_uniform_(
                self.kernel[:, i * hidden_size : (i + 1) * hidden_size]
            )
            nn.init.orthogonal_(
                self.recurrent_kernel[:, i * hidden_size : (i + 1) * hidden_size]
            )
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.recurrent_bias)

    def forward(self, input, state=None, lengths=None):
        """
    Runs a forward pass of the GRU layer.

    Arguments:
      input: Tensor, a batch of input sequences to pass through the GRU.
        Dimensions (seq_len, batch_size, input_size) if `batch_first` is
        `False`, otherwise (batch_size, seq_len, input_size).
      lengths: (optional) Tensor, list of sequence lengths for each batch
        element. Dimension (batch_size). This argument may be omitted if
        all batch elements are unpadded and have the same sequence length.

    Returns:
      output: Tensor, the output of the GRU layer. Dimensions
        (seq_len, batch_size, hidden_size) if `batch_first` is `False` (default)
        or (batch_size, seq_len, hidden_size) if `batch_first` is `True`. Note
        that if `lengths` was specified, the `output` tensor will not be
        masked. It's the caller's responsibility to either not use the invalid
        entries or to mask them out before using them.
      h_n: the hidden state for the last sequence item. Dimensions
        (1, batch_size, hidden_size).
    """
        input = self._permute(input)
        state_shape = [1, input.shape[1], self.hidden_size]
        h0 = self._get_state(input, state, state_shape)
        h = self._impl(input, h0[0], self._get_zoneout_mask(input))
        state = self._get_final_state(h, lengths)
        output = self._permute(h[1:])
        return output, state

    def _impl(self, input, state, zoneout_mask):
        return GRUScript(
            self.training,
            self.zoneout,
            input.contiguous(),
            state.contiguous(),
            self.kernel.contiguous(),
            F.dropout(self.recurrent_kernel, self.dropout, self.training).contiguous(),
            self.bias.contiguous(),
            self.recurrent_bias.contiguous(),
            zoneout_mask.contiguous(),
        )
