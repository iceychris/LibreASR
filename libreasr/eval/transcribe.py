from tqdm import tqdm
import torch
import numpy as np

from sklearn import linear_model

from libreasr.lib.metrics import cer, wer
from libreasr.eval.evaluator import Evaluator


class TranscribeEvaluator(Evaluator):
    def __init__(self, device="cpu", lang_name="unknown"):
        self.device = device
        self.lang_name = lang_name
        self.best = float("inf")

    def eval(
        self,
        learn,
        ddp=False,
        save_fn=lambda x: None,
        num_samples=800,
        pcent=1.0,
        train=False,
        save_best=True,
        save_with_opt=True,
        save_multiple=True,
        **kwargs,
    ):

        # grab stuff
        lang = learn.lang
        device = self.device
        lang_name = self.lang_name
        m = learn.model
        if ddp:
            m = m.module

        def to_device(x):
            if device is None:
                if torch.cuda.is_available():
                    return x.cuda()
                else:
                    return x.to("cpu")
            return x.to(device)

        # choose train or valid dl
        if train:
            dl = learn.dls.train
        else:
            dl = learn.dls.valid
        if ddp:
            dl = dl.dl

        # save & mutate
        back_to_shuffle = dl.shuffle
        dl.shuffle = False

        # grab batch size
        if hasattr(dl, "bs"):
            bs = dl.bs
        else:
            bs = 8

        try:
            n_batches = dl.n // bs
            min_batches = num_samples // bs
            _for = min(min_batches, int(n_batches * pcent) + 1)
            _iter_list = list(range(_for))
            _cers, _wers, _metrics, _xlens, _ylens = [], [], [], [], []
            iterator = iter(dl)
            for batch, _ in tqdm(zip(iterator, _iter_list), total=_for):
                for X, Y in zip(batch[0][0], batch[1][0]):
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
            if _wer < self.best:
                if save_best and _wer < 1.0:
                    save_fn(f"{lang_name}-best_wer", with_opt=save_with_opt)
                    if save_multiple:
                        save_fn(f"{lang_name}-{_wer:.3f}wer", with_opt=save_with_opt)
                    print("New best WER saved:", _wer)
                else:
                    print("New best WER:", _wer)
                self.best = _wer

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
            dl.shuffle = back_to_shuffle
