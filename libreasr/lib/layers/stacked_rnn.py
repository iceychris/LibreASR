import math
import random

import torch
from torch.nn import Parameter, ParameterList
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from IPython.core.debugger import set_trace


ZONEOUT = 0.01
DEVICES = ["CPU", "GPU"]
RNN_TYPES = ["LSTM", "GRU", "NBRC"]
USE_PYTORCH = True


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
        layer_norm=False,
        utsp=0.9,
        norm_cls=nn.BatchNorm1d,
    ):
        super().__init__()
        self.batch_first = batch_first
        self.hidden_size = hidden_size
        self._is = [input_size] + [hidden_size] * (num_layers - 1)
        self._os = [hidden_size] * num_layers
        self.rnn_type = rnn_type

        # reduction
        assert len(reduction_indices) == len(reduction_factors)
        self.reduction_indices = reduction_indices
        self.reduction_factors = reduction_factors

        # learnable state & temporary state
        self.hs = nn.ParameterList()
        for hidden_size in self._os:
            h, tmp = get_initial_state(rnn_type, hidden_size)
            self.hs.append(h)

        # state cache (key: bs, value: state)
        self.cache = {}

        # norm (BN or LN)
        self.bns = nn.ModuleList()
        for i, o in zip(self._is, self._os):
            norm = norm_cls(o)
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
            self.rnns.append(r)

    def forward_one_rnn(
        self, x, i, state=None, should_use_tmp_state=False, lengths=None
    ):
        bs = x.size(0)
        if state is None:
            s = self.cache[bs][i] if self.cache.get(bs) is not None else None
            is_tmp_state_possible = self.training and s is not None
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
            if lengths is not None:
                seq = pack_padded_sequence(
                    x, lengths, batch_first=True, enforce_sorted=False
                )
                seq, new_state = self.rnns[i](seq, s)
                x, _ = pad_packed_sequence(seq, batch_first=True)
                return (x, new_state)
            else:
                return self.rnns[i](x, s)
        else:
            # haste
            return self.rnns[i](x, s, lengths=lengths if lengths is not None else None)

    def forward(self, x, state=None, lengths=None):
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
            x = x.permute(0, 2, 1)
            x = self.bns[i](x)
            x = x.permute(0, 2, 1)

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
