from functools import partial
from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class ResidualSequence(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, **kwargs):
        for layer in self.layers:
            x = x + layer(x, **kwargs)
        return x

    def gather_state(self):
        states = []
        for layer in self.layers:
            s = layer.fn.fn.state
            states.append(s)
        return states

    def to_jit(self):
        self.__class__ = ResidualSequenceJit
        for m in self.layers:
            m.to_jit()


class ResidualSequenceJit(ResidualSequence):
    def forward(
        self, x: Tensor, lengths: Tensor, state: Optional[List[Tuple[Tensor, Tensor]]]
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        l: List[Tuple[Tensor, Tensor]] = []
        for layer in self.layers:
            residual = x
            x, s = layer(x, lengths, state)
            l.append(s)
            x = residual + x
        return x, l


# https://arxiv.org/abs/2103.17239
class LayerScale(nn.Module):
    def __init__(self, dim, depth, fn):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

    def to_jit(self):
        self.__class__ = LayerScaleJit
        self.fn.to_jit()


class LayerScaleJit(LayerScale):
    def forward(
        self, x: Tensor, lengths: Tensor, state: Optional[List[Tuple[Tensor, Tensor]]]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        x, s = self.fn(x, lengths, state)
        return x * self.scale, s


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

    def to_jit(self):
        self.__class__ = PreNormJit
        self.fn.to_jit()


class PreNormJit(PreNorm):
    def forward(
        self, x: Tensor, lengths: Tensor, state: Optional[List[Tuple[Tensor, Tensor]]]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        return self.fn(self.norm(x), lengths, state)


class LSTMWrapper(nn.Module):
    def __init__(self, fn, idx):
        super().__init__()
        self.fn = fn
        self.idx = idx
        self.state = None

    def forward(self, x, state=None, **kwargs):
        if state is not None:
            s = state[self.idx]
        else:
            s = None
        x, s = self.fn(x, s, **kwargs)
        self.state = s
        return x

    def to_jit(self):
        self.__class__ = LSTMWrapperJit


class LSTMWrapperJit(LSTMWrapper):
    def forward(
        self, x: Tensor, lengths: Tensor, state: Optional[List[Tuple[Tensor, Tensor]]]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if state is None:
            s = None
        else:
            s = state[self.idx]
        xn, sn = self.fn(x, s)
        return xn, sn


class SlimEncoder(nn.Module):
    def __init__(
        self,
        feature_sz,
        hidden_sz,
        out_sz,
        dropout=0.01,
        dropout_input=0.0,
        dropout_inner=0.0,
        num_layers=2,
        trace=True,
        device="cuda:0",
        rnn_type="LSTM",
        norm="bn",
        attention=False,
        use_tmp_state_pcent=0.9,
        reversible=False,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.drop_input = nn.Dropout(dropout_input)
        self.input_norm = nn.LayerNorm(feature_sz)
        self.drop = nn.Dropout(dropout)

        # ff layer at start
        self.ff1 = nn.Linear(feature_sz, hidden_sz)

        # ff layer at end
        if not hidden_sz == out_sz:
            self.ff2 = nn.Linear(hidden_sz, out_sz)
        else:
            self.ff2 = nn.Sequential()

        # initialize recurrent layers
        dim = hidden_sz
        self.dim = dim
        lstm = partial(nn.LSTM, dim, dim, batch_first=True, num_layers=1)
        layers = []
        for ind in range(num_layers):
            layers.extend(
                [
                    LayerScale(dim, ind + 1, PreNorm(dim, LSTMWrapper(lstm(), ind))),
                ]
            )
        execute_type = ResidualSequence
        self.net = execute_type(layers)
        self.norm = nn.LayerNorm(out_sz)

    def initial_state(self):
        nl, dim = self.num_layers, self.dim

        def mk_zeros(*a):
            return torch.zeros(*a)

        return [(mk_zeros(1, 1, dim), mk_zeros(1, 1, dim)) for _ in range(nl)]

    def to_jit(self):
        self.net.to_jit()
        self.__class__ = SlimEncoderJit

    def param_groups(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x, state=None, lengths=None, return_state=False):
        x = x.reshape((x.size(0), x.size(1), -1))
        x = self.drop_input(x)
        x = self.input_norm(x)

        # main block
        x = self.ff1(x)
        x = self.net(x, state=state)
        x = self.ff2(x)
        x = self.drop(x)

        # final norm
        x = self.norm(x)

        if return_state:
            s = self.net.gather_state()
            return x, s
        return x


class SlimEncoderJit(SlimEncoder):
    def forward(
        self, x: Tensor, lengths: Tensor, state: Optional[List[Tuple[Tensor, Tensor]]]
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        x = x.reshape((x.size(0), x.size(1), -1))
        x = self.drop_input(x)
        x = self.input_norm(x)

        # main block
        x = self.ff1(x)
        x, new_state = self.net(x, lengths, state)
        x = self.ff2(x)
        x = self.drop(x)

        # final norm
        x = self.norm(x)

        return x, new_state
