from functools import partial
import multiprocessing
import math
import sys

# fastai v2 stuff
from fastai2.torch_basics import *
from fastai2.layers import *
from fastai2.data.all import *
from fastai2.optimizer import *
from fastai2.learner import *
from fastai2.metrics import *
from fastai2.text.core import *
from fastai2.text.data import *
from fastai2.text.models.core import *
from fastai2.text.models.awdlstm import *
from fastai2.text.learner import *
from fastai2.callback.rnn import *
from fastai2.callback.all import *

from fastai2.vision.learner import *
from fastai2.vision.models.xresnet import *

from fastai2_audio.core import *

import torchaudio
import numpy as np
import pandas as pd

from .utils import *


def get_loss_func(
    loss_type,
    device,
    reduction_factor,
    debug=True,
    perf=True,
    entropy_loss=False,
    zero_loss=False,
    div_by_len=False,
    zero_nan=True,
    zero_inf=True,
    keep_best_pcent=-0.75,
    keep_best_largest=True,
):

    if debug:
        print("loss_type:", loss_type)

    def reducer(_loss, reduction="mean", zero_infinity=False):
        if debug:
            print("unreduced loss:", _loss)
        # replace inf with 0.
        if zero_infinity:
            _loss[_loss == float("Inf")] = 0.0
        # reduce
        if reduction == "mean":
            return _loss.mean()
        else:
            return _loss

    if loss_type == "ctc":
        from torch.nn import CTCLoss

        loss_func = CTCLoss(zero_infinity=True, reduction="none").to(device)

    elif loss_type == "rnnt":
        # from warprnnt_pytorch import RNNTLoss
        # loss_func = RNNTLoss(reduction='none').to(device)

        # faster
        from warp_rnnt import rnnt_loss

        loss_func = partial(rnnt_loss, average_frames=False)

    else:
        raise Exception(f"no such loss type: {loss_type}")

    def _loss_func(inp, tgt, reduction="mean", **kwargs):
        if perf:
            t = time.time()

        # unpack
        tgt, tgt_lens, inp_lens = tgt
        tgt = tgt.type(torch.int32)
        inp_lens = inp_lens.type(torch.int32)
        tgt_lens = tgt_lens.type(torch.int32)

        # preprocess inp_lens
        # factor reduction in
        inp_lens = inp_lens // reduction_factor

        # avoid NaN
        if zero_nan:
            inp = torch.where(torch.isnan(inp), torch.zeros_like(inp), inp)

        # avoid Inf
        if zero_inf:
            inp[inp == float("Inf")] = 0
            inp[inp == float("-Inf")] = 0

        # do loss calculation
        inp, tgt = inp.to(device), tgt.to(device)
        inp_lens, tgt_lens = inp_lens.to(device), tgt_lens.to(device)
        loss = loss_func(inp, tgt, inp_lens, tgt_lens)
        if entropy_loss:
            el = entropy_crit(inp)
            # print("el   ", el.mean())
            # print("loss ", loss.mean())
            loss += el
        if zero_loss:
            zl = (1 / (inp[:, :, 0].abs() + 1e-5)).mean(-1) * tgt_lens * 1.0
            # print(loss.mean(), zl.mean())
            loss += zl

        # divide by tgt lens
        if div_by_len:
            loss /= tgt_lens + 1e-5

        # drop lowest losses
        if keep_best_pcent >= 0.0 and keep_best_pcent < 1.0:
            s, _ = loss.sort(descending=keep_best_largest)
            loss = s[: int(len(loss) * keep_best_pcent)]

        # reduce
        loss = reducer(loss, reduction)
        if perf:
            t = (time.time() - t) * 1000.0
            print(f"loss took {t:2.2f}ms")

        return loss

    return _loss_func
