from functools import partial

import torch
from torch import nn
import torch.nn.functional as F


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


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


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
