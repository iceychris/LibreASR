import torch
import torch.nn as nn
import torch.nn.functional as F


def _mish_fwd(x):
    return x.mul(torch.tanh(F.softplus(x)))


def _mish_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    x_tanh_sp = F.softplus(x).tanh()
    return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))


class MishAutoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _mish_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        return _mish_bwd(x, grad_output)


class _Mish(nn.Module):
    def forward(self, x):
        return MishAutoFn.apply(x)


try:
    from fastai2.layers import Mish

    Mish = Mish
    print("Using Mish activation from fastai2.")
except:
    Mish = _Mish
    print("Using Mish activation from lib.layers.")
