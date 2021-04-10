"""
LocalAttention Implementation by lucidrains:
https://github.com/lucidrains/local-attention
"""

import math
from operator import mul
from functools import reduce

import torch
from torch import nn
import torch.nn.functional as F


class DualModeConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size):
        super().__init__()
        k = kernel_size
        assert k % 2 != 0 and k >= 3
        self.k = k
        self.mask = (torch.arange(k) <= k // 2).expand(chan_in, chan_out, k)
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size)

    def forward(self, x, stream=True):
        if stream:
            # apply mask & convolve
            weight = self.conv.weight * self.mask
            x = F.conv1d(x, weight, self.conv.bias, self.conv.stride,
                        self.conv.padding, self.conv.dilation, self.conv.groups)
            return x
        return self.conv(x)


# constant

TOKEN_SELF_ATTN_VALUE = -5e4 # carefully set for half precision to work

# helper functions

def default(value, d):
    return d if value is None else value

def to(t):
    return {'device': t.device, 'dtype': t.dtype}

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)

def expand_dim(t, dim, k, unsqueeze=True):
    if unsqueeze:
        t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value=value)

def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value= pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim=dim)

# Shaw's relative positional encoding per window

def shift(x):
    *_, i, j = x.shape
    zero_pad = torch.zeros((*_, i, i), **to(x))
    x = torch.cat([x, zero_pad], -1)
    l = i + j - 1
    x = x.view(*_, -1)
    zero_pad = torch.zeros(*_, -x.size(-1) % l, **to(x))
    shifted = torch.cat([x, zero_pad], -1).view(*_, -1, l)
    return shifted[..., :i, i - 1:]

class RelativePositionalEmbedding(nn.Module):
    def __init__(self, dim, heads, length):
        super().__init__()
        self.scale = dim ** -0.5
        self.weights = nn.Parameter(torch.zeros(length, heads, dim))

    def forward(self, q):
        emb = torch.einsum('bhnid,jhd->bhnij', q, self.weights.type(q.dtype)) * self.scale
        return shift(emb)

# main class

class LocalAttention(nn.Module):
    def __init__(self, window_size, causal = False, look_backward = 1, look_forward = None, dropout = 0., shared_qk = False, rel_pos_emb_config = None, autopad = False, exact_windowsize = False):
        super().__init__()
        look_forward = default(look_forward, 0 if causal else 1)
        assert not (causal and look_forward > 0), 'you cannot look forward if causal'

        self.window_size = window_size
        self.causal = causal
        self.look_backward = look_backward
        self.look_forward = look_forward
        self.exact_windowsize = exact_windowsize
        self.autopad = autopad

        self.dropout = nn.Dropout(dropout)

        self.shared_qk = shared_qk

        self.rel_pos = None
        if rel_pos_emb_config is not None:
            dim_head, heads = rel_pos_emb_config
            rel_pos_length = window_size * (1 + look_forward + look_backward)
            self.heads = heads
            self.rel_pos = RelativePositionalEmbedding(dim_head, heads, rel_pos_length)

    def forward(self, q, k, v, input_mask = None):
        shape = q.shape

        merge_into_batch = lambda t: t.reshape(-1, *t.shape[-2:])
        q, k, v = map(merge_into_batch, (q, k, v))

        if self.autopad:
            orig_t = q.shape[1]
            q, k, v = map(lambda t: pad_to_multiple(t, self.window_size, dim = -2), (q, k, v))

        window_size, causal, look_backward, look_forward, shared_qk = self.window_size, self.causal, self.look_backward, self.look_forward, self.shared_qk
        b, t, e, device, dtype = *q.shape, q.device, q.dtype
        assert (t % window_size) == 0, f'sequence length {t} must be divisible by window size {window_size} for local attention'

        windows = t // window_size

        if shared_qk:
            k = F.normalize(k, 2, dim=-1).type_as(q)

        ticker = torch.arange(t, device=device, dtype=dtype)[None, :]
        b_t = ticker.reshape(1, windows, window_size)

        bucket_fn = lambda t: t.reshape(b, windows, window_size, -1)
        bq, bk, bv = map(bucket_fn, (q, k, v))

        look_around_kwargs = {'backward': look_backward, 'forward': look_forward}
        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)

        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)

        dots = torch.einsum('bhie,bhje->bhij', bq, bk) * (e ** -0.5)

        if self.rel_pos is not None:
            rel_attn = self.rel_pos(bq.view(-1, self.heads, *bq.shape[1:])).reshape_as(dots)
            dots = dots + rel_attn

        mask_value = max_neg_value(dots)

        if shared_qk:
            mask = bq_t[:, :, :, None] == bq_k[:, :, None, :]
            dots.masked_fill_(mask, TOKEN_SELF_ATTN_VALUE)
            del mask

        if causal:
            mask = bq_t[:, :, :, None] < bq_k[:, :, None, :]

            if self.exact_windowsize:
                max_causal_window_size = (self.window_size * self.look_backward)
                mask = mask | (bq_t[:, :, :, None] > (bq_k[:, :, None, :] + max_causal_window_size))

            dots.masked_fill_(mask, mask_value)
            del mask

        mask = bq_k[:, :, None, :] == -1
        dots.masked_fill_(mask, mask_value)
        del mask

        if input_mask is not None:
            h = b // input_mask.shape[0]
            if self.autopad:
                input_mask = pad_to_multiple(input_mask, window_size, dim=-1, value=False)
            input_mask = input_mask.reshape(-1, windows, window_size)
            mq = mk = input_mask
            mk = look_around(mk, pad_value=False, **look_around_kwargs)
            mask = (mq[:, :, :, None] * mk[:, :, None, :])
            mask = merge_dims(0, 1, expand_dim(mask, 1, h))
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhij,bhje->bhie', attn, bv)
        out = out.reshape(-1, t, e)

        if self.autopad:
            out = out[:, :orig_t, :]

        return out.reshape(*shape)


class DualModeSelfAttention(nn.Module):
    def __init__(self, dim, inner_dim, **kwargs):
        super().__init__()
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        # causal = False,          # auto-regressive or not
        # look_backward = 1,       # each window looks at the window before
        # look_forward = 1,        # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
        self.attn_stream = LocalAttention(causal=True, look_backward=1, look_forward=0, **kwargs)
        self.attn_full = LocalAttention(causal=False, look_backward=1, look_forward=1, **kwargs)

    def forward(self, x, stream=True):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        if stream:
            x = self.attn_stream(q, k, v)
        else:
            x = self.attn_full(q, k, v)
        return x


class DualModeMultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, residual=True, **kwargs):
        super().__init__()
        inner_dim = dim // n_heads
        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_k = nn.Linear(dim, dim, bias = False)
        self.to_v = nn.Linear(dim, dim, bias = False)
        self.to_o = nn.Linear(dim, dim, bias = True)
        self.norm = nn.LayerNorm(dim)
        # causal = False,          # auto-regressive or not
        # look_backward = 1,       # each window looks at the window before
        # look_forward = 1,        # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
        self.attn_stream = LocalAttention(causal=True, look_backward=1, look_forward=0, **kwargs)
        self.attn_full = LocalAttention(causal=False, look_backward=1, look_forward=1, **kwargs)
        self.n_heads = n_heads
        self.residual = residual
        assert (
            dim // n_heads
        ) * n_heads == dim, "dim must be divisible by n_heads"

    def forward(self, x, stream=True):
        
        # maybe store residual
        residual = x if self.residual else 0.0
        
        # create queries, keys and values
        q, k, v = map(lambda x: x[0](x[1]), zip((self.to_q, self.to_k, self.to_v), (x,) * 3))
        
        # chunk
        q, k, v = (x.chunk(self.n_heads, -1) for x in (q, k, v))
        
        # choose attn func
        if stream:
            attn = self.attn_stream
        else:
            attn = self.attn_full
            
        # attend
        x = torch.cat(
                [attn(q, k, v) for q, k, v in zip(q, k, v)], -1
            )
        
        # project
        x = self.to_o(x)

        # norm
        x = self.norm(x)
        
        # add residual
        x = x + residual
        return x
