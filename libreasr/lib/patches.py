from functools import partial

# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
from fastcore.foundation import patch
from fastai2.learner import Learner
import numpy as np

from matplotlib import pyplot as plt
from fastai2.torch_basics import *
from fastai2.data.all import *
from fastai2_audio.core.signal import AudioTensor

from libreasr.lib.metrics import cer, wer
from IPython.core.debugger import set_trace


@patch
def test(
    self: Learner,
    pcent=0.5,
    min_samples=320,
    device="cuda:0",
    train=False,
    mp=False,
    save_best=True,
):
    lang = self.model.lang
    m = self.model.to(device)

    # choose train or valid dl
    if train:
        dl = self.dls.train
    else:
        dl = self.dls.valid

    # store best attr (at start)
    if save_best and not hasattr(self, "best_wer"):
        self.best_wer = 50.0

    # save & mutate
    back_to_train = self.training
    back_to_shuffle = dl.shuffle
    back_to_half = mp
    if back_to_half:
        self.model.float()
    dl.shuffle = False

    bs = 8
    n_batches = dl.n // bs
    min_batches = min_samples // bs
    _for = min(min_batches, int(n_batches * pcent) + 1)
    _iter_list = list(range(_for))
    _cers, _wers, _metrics, _xlens, _ylens = [], [], [], [], []
    iterator = iter(dl)
    for batch, _ in tqdm(zip(iterator, _iter_list), total=_for):
        for X, Y in zip(batch[0][0], batch[1][0]):
            # set_trace()
            utterance = X.to(device)
            label = Y.detach().cpu().numpy().tolist()

            _pred, _metric = m.transcribe(utterance)
            _true = lang.denumericalize(label)

            _metric = {"metrics/" + k: v for k, v in _metric.items()}
            _metrics.append(_metric)
            _cer = cer(_pred, _true)
            _wer = wer(_pred, _true)
            _cers.append(_cer)
            _wers.append(_wer)
            _xlens.append(X.size(-2))
            _ylens.append(len(_true))
            d = {
                **_metric,
                "metrics/cer": _cer,
                "metrics/wer": _wer,
                "text/prediction": _pred,
                "text/ground_truth": _true,
            }
            yield d
    aligns = [M["metrics/alignment_score"] for M in _metrics]
    _cer = np.array(_cers).mean()
    _wer = np.array(_wers).mean()

    # maybe save best
    if hasattr(self, "best_wer"):
        if _wer < self.best_wer:
            if save_best:
                self.save("best_wer", with_opt=True)
                print("New best WER saved:", _wer)
            else:
                print("New best WER:", _wer)
            self.best_wer = _wer

    # plot lens
    # plt.hist(_xlens, bins=20)
    # plt.show()
    # plt.hist(_ylens, bins=20)
    # plt.show()

    last = f"true: {_true} | pred: {_pred}"
    print(f"CER={_cer:.3f} | WER={_wer:.3f} | {last}")
    yield {
        "metrics/mean_alignment_score": np.array(aligns).mean(),
        "metrics/mean_cer": _cer,
        "metrics/mean_wer": _wer,
        "text/last": last,
    }

    # restore
    if back_to_train:
        self.model.train()
    if back_to_half:
        self.model.half()
    dl.shuffle = back_to_shuffle


# no sr required
def __new__(cls, x, sr=16000, **kwargs):
    return TensorBase.__new__(cls, x, sr=sr, **kwargs)


AudioTensor.__new__ = __new__
