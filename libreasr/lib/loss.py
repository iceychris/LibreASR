from functools import partial
import multiprocessing
import math
import sys

import torchaudio
import numpy as np
import pandas as pd

from libreasr.lib.utils import *


def get_loss_func(
    loss_type,
    device,
    reduction_factor,
    noisystudent=False,
    debug=True,
    perf=True,
    div_by_len=False,
    zero_nan=True,
    zero_inf=False,
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

        loss_func = CTCLoss(zero_infinity=True, reduction="none")  # .to(device)

    elif loss_type == "UNUSED":
        from warp_rnnt import rnnt_loss

        loss_func = partial(rnnt_loss, average_frames=False)
        if noisystudent:
            loss_func_ns = nn.KLDivLoss(log_target=True, reduction="none")

    elif loss_type == "contrastive" or loss_type == "rnnt":

        def l(loss, *args, reduction="mean"):
            if reduction == "mean":
                return loss.mean()
            else:
                return loss

        return l

    else:
        raise Exception(f"no such loss type: {loss_type}")

    def _loss_func(inp, tgt, reduction="mean", **kwargs):
        if perf:
            t = time.time()

        # extract if noisystudent training
        if isinstance(inp, tuple):
            teacher_logits, inp = inp

        # unpack
        tgt, tgt_lens, inp_lens = tgt
        tgt = tgt.type(torch.int32)
        inp_lens = inp_lens.type(torch.int32)
        tgt_lens = tgt_lens.type(torch.int32)

        # preprocess inp_lens
        # factor reduction in
        inp_lens = torch.clamp((inp_lens // reduction_factor) - 1, min=1, max=inp.size(1))

        # avoid NaN
        if zero_nan:
            inp = torch.where(torch.isnan(inp), torch.zeros_like(inp), inp)

        # avoid Inf
        if zero_inf:
            inp[inp == float("Inf")] = 0
            inp[inp == float("-Inf")] = 0

        # do loss calculation
        if device.startswith("cuda"):
            inp, tgt = inp.cuda(), tgt.cuda()
            inp_lens, tgt_lens = inp_lens.cuda(), tgt_lens.cuda()
        else:
            inp, tgt = inp.cpu(), tgt.cpu()
            inp_lens, tgt_lens = inp_lens.cpu(), tgt_lens.cpu()
        if noisystudent:
            # perform log softmax as we did not do it before
            inp_sm = F.log_softmax(inp, dim=-1)
        loss = loss_func(inp_sm if noisystudent else inp, tgt, inp_lens, tgt_lens)

        # noisy student loss
        if noisystudent:

            # params
            alpha = 0.001
            T = 1.0

            # weighted loss
            loss_rnnt = loss * (1.0 - alpha)
            loss_ns = (
                loss_func_ns(
                    F.log_softmax(inp / T, dim=-1),
                    F.log_softmax(teacher_logits / T, dim=-1),
                )
                * (alpha * T * T)
            )
            loss_ns = loss_ns.mean((1, 2, 3)) * 20000.0 * 5

            # print(f"RNN-T: {loss_rnnt.mean():.2f}, NS: {loss_ns.mean():.2f}")

            loss = loss_rnnt + loss_ns

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
