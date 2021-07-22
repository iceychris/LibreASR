import math
import random
from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import Parameter, ParameterList
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from IPython.core.debugger import set_trace


ZONEOUT = 0.01
DEVICES = ["CPU", "GPU"]
RNN_TYPES = ["LSTM", "GRU", "NBRC"]
USE_PYTORCH = True


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


NORMS = {
    "bn": nn.BatchNorm1d,
    "ln": nn.LayerNorm,
    "sn": ScaleNorm,
    "rn": RMSNorm,
}


def get_rnn_impl(rnn_type):
    return getattr(nn, rnn_type)


def get_initial_state(rnn_type, hidden_size, init=torch.zeros):
    if rnn_type == "LSTM":
        h = nn.Parameter(init(2, 1, 1, hidden_size))
        tmp = init(2, 1, 1, hidden_size)
    else:
        h = nn.Parameter(init(1, 1, 1, hidden_size))
        tmp = init(1, 1, 1, hidden_size)
    return h, tmp


class WithPermute(nn.Module):
    def __init__(
        self,
        module,
        permutation,
    ):
        super().__init__()
        self.mod = module
        self.perm = permutation

    def forward(self, x):
        x = x.permute(self.perm)
        x = self.mod(x)
        x = x.permute(self.perm)
        return x


class StackedRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        batch_first=True,
        rnn_type="LSTM",
        reduction_indices=[],
        reduction_factors=[],
        reduction_drop=True,
        rezero=False,
        utsp=0.9,
        norm="bn",
        dropout=0.1,
    ):
        super().__init__()
        self.batch_first = batch_first
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self._is = [input_size] + [hidden_size] * (num_layers - 1)
        self._os = [hidden_size] * num_layers
        self.rnn_type = rnn_type

        # reduction
        assert len(reduction_indices) == len(reduction_factors)
        self.reduction_indices = reduction_indices
        self.reduction_factors = reduction_factors

        # learnable state & temporary state
        t = torch.zeros(len(self._os), 2, 1, 1, self._os[-1])
        for i, hidden_size in enumerate(self._os):
            h, tmp = get_initial_state(rnn_type, hidden_size)
            t[i][:] = h
            # self.hs.append(h)
        self.hs = nn.Parameter(t.contiguous())

        # state cache (key: bs, value: state)
        self.cache = {}
        self.reinit_cache = True

        # norm (BN or LN)
        self.norm = norm
        norm_cls = NORMS[norm]
        self.bns = nn.ModuleList()
        for i, o in zip(self._is, self._os):
            n = norm_cls(o)
            norm = n  # WithPermute(n, (0, 2, 1)) if norm_cls == nn.BatchNorm1d else n
            self.bns.append(norm)

        # rezero
        self.rezero = rezero

        # percentage of carrying over last state
        self.utsp = utsp

        # inititalize rnn stack
        rnn_cls = get_rnn_impl(self.rnn_type)
        self.rnns = nn.ModuleList()
        for i, o in zip(self._is, self._os):
            r = rnn_cls(i, o, batch_first=self.batch_first)
            r.flatten_parameters()
            self.rnns.append(r)

        # dropouts
        self.dropout = dropout
        if dropout > 0.0:
            self.dropouts = nn.ModuleList(
                [nn.Dropout(dropout) for _ in range(num_layers)]
            )

    def to_jit(self):
        self.__class__ = StackedRNNJit

    def initial_state(self, bs=1):
        s = []
        for i in range(self.num_layers):
            a = self.hs[i][0].expand(1, bs, self._os[i]).contiguous()
            b = self.hs[i][0].expand(1, bs, self._os[i]).contiguous()
            s.append((a, b))
        s = tuple(s)
        return s

    def forward_one_rnn(
        self, x, i, state=None, should_use_tmp_state=False, lengths=None
    ):
        bs = x.size(0)
        if state is None:
            s = self.cache[bs][i] if self.cache.get(bs) is not None else None
            same_dev_and_not_none = False if s is None else s[0].device == x.device
            is_tmp_state_possible = (
                self.training and s is not None and same_dev_and_not_none
            )
            if is_tmp_state_possible and should_use_tmp_state:
                # temporary state
                pass
            else:
                # learnable state
                if self.hs[i].size(0) == 2:
                    s = []
                    for h in self.hs[i]:
                        s.append(h.expand(1, bs, self._os[i]).contiguous())
                    s = tuple(s)
                else:
                    s = self.hs[i][0].expand(1, bs, self._os[i]).contiguous()
        else:
            s = state[i]

        if self.rnn_type == "LSTM" or self.rnn_type == "GRU":
            # PyTorch
            t_idx = 1 if self.batch_first else 0
            if False:  # lengths is not None:
                seq = pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                seq, new_state = self.rnns[i](seq, s)
                x, _ = pad_packed_sequence(
                    seq, batch_first=True, total_length=x.size(t_idx)
                )
                return (x, new_state)
            else:
                return self.rnns[i](x, s)
        else:
            # haste
            return self.rnns[i](x, s, lengths=lengths if lengths is not None else None)

    def forward(self, x, state=None, lengths=None):
        if self.reinit_cache:
            self.cache = {}
            self.reinit_cache = False
        bs = x.size(0)
        residual = 0.0
        new_states = []
        if self.utsp > 0.0:
            suts = random.random() > (1.0 - self.utsp)
        else:
            suts = False
        for i, rnn in enumerate(self.rnns):

            # reduce if necessary
            if i in self.reduction_indices:
                idx = self.reduction_indices.index(i)
                r_f = self.reduction_factors[idx]

                # to [N, H, T]
                x = x.permute(0, 2, 1)

                x = x.unfold(-1, r_f, r_f)
                x = x.permute(0, 2, 1, 3)

                # keep last
                # x = x[:,:,:,-1]
                # or take the mean
                x = x.mean(-1)

                # also reduce lengths
                if lengths is not None:
                    lengths = lengths // r_f

            # apply rnn
            inp = x
            x, new_state = self.forward_one_rnn(
                inp,
                i,
                state=state,
                should_use_tmp_state=suts,
                lengths=lengths,
            )

            # apply norm
            if self.norm == "bn":
                x = x.permute(0, 2, 1)
                x = self.bns[i](x)
                x = x.permute(0, 2, 1)
            else:
                x = self.bns[i](x)

            # apply dropout
            if self.dropout > 0.0:
                x = self.dropouts[i](x)

            # apply residual
            if self.rezero:
                if torch.is_tensor(residual) and residual.shape == x.shape:
                    x = x + residual

            # store new residual
            residual = inp

            new_states.append(new_state)
        if self.training:
            if len(new_states[0]) == 2:
                self.cache[bs] = [
                    (h.detach().contiguous(), c.detach().contiguous())
                    for (h, c) in new_states
                ]
            else:
                self.cache[bs] = [h.detach() for h in new_states]
        return x, new_states


class StackedRNNJit(StackedRNN):
    def forward(
        self, x: Tensor, lengths: Tensor, state: Optional[List[Tuple[Tensor, Tensor]]]
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        new_states: List[Tuple[Tensor, Tensor]] = []
        for i, (rnn, norm) in enumerate(zip(self.rnns, self.bns)):
            # x, s = rnn(x, state[i])
            if state is None:
                # use learnable state
                s = (self.hs[i][0], self.hs[i][1])
            else:
                s = state[i]
            x, s = rnn(x, s)
            new_states.append(s)
            x = norm(x)
        return x, new_states
