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
