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

import torch
import torch.nn as nn


__all__ = ["BaseRNN"]


class BaseRNN(nn.Module):
    def __init__(
        self, input_size, hidden_size, batch_first, zoneout, return_state_sequence
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.zoneout = zoneout
        self.return_state_sequence = return_state_sequence

    def _permute(self, x):
        if self.batch_first:
            return x.permute(1, 0, 2)
        return x

    def _get_state(self, input, state, state_shape):
        if state is None:
            state = _zero_state(input, state_shape)
        else:
            _validate_state(state, state_shape)
        return state

    def _get_final_state(self, state, lengths):
        if isinstance(state, tuple):
            return tuple(self._get_final_state(s, lengths) for s in state)
        if isinstance(state, list):
            return [self._get_final_state(s, lengths) for s in state]
        if self.return_state_sequence:
            return self._permute(state[1:]).unsqueeze(0)
        if lengths is not None:
            cols = range(state.size(1))
            return state[[lengths, cols]].unsqueeze(0)
        return state[-1].unsqueeze(0)

    def _get_zoneout_mask(self, input):
        if self.zoneout:
            zoneout_mask = input.new_empty(
                input.shape[0], input.shape[1], self.hidden_size
            )
            zoneout_mask.bernoulli_(1.0 - self.zoneout)
        else:
            zoneout_mask = input.new_empty(0, 0, 0)
        return zoneout_mask

    def _is_cuda(self):
        is_cuda = [tensor.is_cuda for tensor in list(self.parameters())]
        if any(is_cuda) and not all(is_cuda):
            raise ValueError(
                "RNN tensors should all be CUDA tensors or none should be CUDA tensors"
            )
        return any(is_cuda)


def _validate_state(state, state_shape):
    """
  Checks to make sure that `state` has the same nested structure and dimensions
  as `state_shape`. `None` values in `state_shape` are a wildcard and are not
  checked.

  Arguments:
    state: a nested structure of Tensors.
    state_shape: a nested structure of integers or None.

  Raises:
    ValueError: if the structure and/or shapes don't match.
  """
    if isinstance(state, (tuple, list)):
        # Handle nested structure.
        if not isinstance(state_shape, (tuple, list)):
            raise ValueError(
                "RNN state has invalid structure; expected {}".format(state_shape)
            )
        for s, ss in zip(state, state_shape):
            _validate_state(s, ss)
    else:
        shape = list(state.size())
        if len(shape) != len(state_shape):
            raise ValueError(
                "RNN state dimension mismatch; expected {} got {}".format(
                    len(state_shape), len(shape)
                )
            )

        for i, (d1, d2) in enumerate(zip(list(state.size()), state_shape)):
            if d2 is not None and d1 != d2:
                raise ValueError(
                    "RNN state size mismatch on dim {}; expected {} got {}".format(
                        i, d2, d1
                    )
                )


def _zero_state(input, state_shape):
    """
  Returns a nested structure of zero Tensors with the same structure and shape
  as `state_shape`. The returned Tensors will have the same dtype and be on the
  same device as `input`.

  Arguments:
    input: Tensor, to specify the device and dtype of the returned tensors.
    shape_state: nested structure of integers.

  Returns:
    zero_state: a nested structure of zero Tensors.

  Raises:
    ValueError: if `state_shape` has non-integer values.
  """
    if isinstance(state_shape, (tuple, list)) and isinstance(state_shape[0], int):
        state = input.new_zeros(*state_shape)
    elif isinstance(state_shape, tuple):
        state = tuple(_zero_state(input, s) for s in state_shape)
    elif isinstance(state_shape, list):
        state = [_zero_state(input, s) for s in state_shape]
    else:
        raise ValueError("RNN state_shape is invalid")
    return state
