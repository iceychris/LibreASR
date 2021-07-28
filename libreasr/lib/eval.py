from tqdm import tqdm
from fastai.learner import Learner
import numpy as np

from libreasr.lib.metrics import cer, wer
from IPython.core.debugger import set_trace


def eval_speech_model(
    self: Learner,
    pcent=1.0,
    min_samples=200,  # 800,
    device=None,
    train=False,
    ddp=False,
    lang_name="unknown",
    save_best=True,
    save_with_opt=True,
    save_multiple=True,
):
    lang = self.lang
    m = self.model
    if ddp:
        m = m.module

    def to_device(x):
        if device is None:
            if torch.cuda.is_available():
                return x.cuda()
            else:
                return x.to("cpu")
        return x.to(device)

    # put model in eval mode
    m = to_device(m).eval()

    # choose train or valid dl
    if train:
        dl = self.dls.train
    else:
        dl = self.dls.valid
    if ddp:
        dl = dl.dl

    # store best attr (at start)
    if save_best and not hasattr(self, "best_wer"):
        self.best_wer = 50.0

    # save & mutate
    back_to_train = self.training
    back_to_shuffle = dl.shuffle
    dl.shuffle = False

    # grab batch size
    if hasattr(dl, "bs"):
        bs = dl.bs
    else:
        bs = 8

    try:
        n_batches = dl.n // bs
        min_batches = min_samples // bs
        _for = min(min_batches, int(n_batches * pcent) + 1)
        _iter_list = list(range(_for))
        _cers, _wers, _metrics, _xlens, _ylens = [], [], [], [], []
        iterator = iter(dl)
        for batch, _ in tqdm(zip(iterator, _iter_list), total=_for):
            for X, Y in zip(batch[0][0], batch[1][0]):
                # set_trace()
                utterance = to_device(X)
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
                if save_best and _wer < 1.0:
                    self.save(f"{lang_name}-best_wer", with_opt=save_with_opt)
                    if save_multiple:
                        self.save(f"{lang_name}-{_wer:.3f}wer", with_opt=save_with_opt)
                    print("New best WER saved:", _wer)
                else:
                    print("New best WER:", _wer)
                self.best_wer = _wer

        last = f"true: {_true} | pred: {_pred}"
        print(f"CER={_cer:.4f} | WER={_wer:.4f} | {last}")
        yield {
            "metrics/mean_alignment_score": np.array(aligns).mean(),
            "metrics/mean_cer": _cer,
            "metrics/mean_wer": _wer,
            "text/last": last,
        }

    # bubble up
    except Exception as e:
        raise e

    # cleanup & restore
    finally:
        if back_to_train:
            self.model.train()
        dl.shuffle = back_to_shuffle
