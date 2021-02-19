from functools import partial
import multiprocessing
import math
import time
import sys
import random
from pathlib import Path
from typing import Tuple

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
import tqdm

import pandas as pd
import numpy as np

from libreasr.lib.utils import *
from libreasr.lib.transforms import update_tfms, update_tfms_multi


# x: maximum batch capacity
#  n stacked frames
X_MAX = 8 * 7000  # 9000  # 6000 # 6500 # 7750

# y: maximum batch capacity
#  n BPE tokens
Y_MAX = 8 * 90  # 85
Y_MAX_ONE = 90

# x: time dimension
DIM_TIME = 1

# pre sorting capacity (n samples)
DL_ADVANCE_BY_SORTISH = 100
DL_ADVANCE_BY_BUCKETING = 4000

# debugging
PRINT_BATCH_STATS = False

# rng issues
HOME = "/home/chris"


@delegates(TfmdDL)
class SortishDL(TfmdDL):
    def __init__(
        self,
        dataset,
        tpls,
        sort_func=None,
        res=None,
        reverse=True,
        **kwargs,
    ):
        super().__init__(dataset, **kwargs)
        self.sort_func = _default_sort if sort_func is None else sort_func
        self.res = (
            [self.sort_func(tpls[it], noop=True) for it in self.items]
            if res is None
            else res
        )
        self.idx_max = np.argmax(self.res)
        self.reverse = reverse

    def get_idxs(self):
        idxs = super().get_idxs()
        if self.shuffle:
            return idxs
        return sorted(idxs, key=lambda i: self.res[i], reverse=not self.reverse)

    def shuffle_fn(self, idxs):
        idxs = np.random.permutation(idxs)
        idx_max = np.extract(idxs == self.idx_max, idxs)[0]
        idxs[0], idxs[idx_max] = idxs[idx_max], idxs[0]
        sz = self.bs * DL_ADVANCE_BY_SORTISH
        chunks = [idxs[i : i + sz] for i in range(0, len(idxs), sz)]
        chunks = [sorted(s, key=lambda i: self.res[i], reverse=True) for s in chunks]
        sort_idx = np.concatenate(chunks)

        sz = self.bs
        batches = [sort_idx[i : i + sz] for i in range(0, len(sort_idx), sz)]
        sort_idx = (
            np.concatenate(np.random.permutation(batches[1:-1]))
            if len(batches) > 2
            else np.array([], dtype=np.int)
        )
        sort_idx = np.concatenate(
            (batches[0], sort_idx)
            if len(batches) == 1
            else (batches[0], sort_idx, batches[-1])
        )
        return iter(sort_idx)


@delegates(TfmdDL)
class DynamicBucketingDL(TfmdDL):
    def __init__(
        self,
        dataset,
        tpls,
        sort_func=None,
        res=None,
        reverse=True,
        bs_max=32,
        bs_mul=1.0,
        **kwargs,
    ):
        """
        mul_bs contrastive: 7.5
        mul_bs rnnt: 1.
        """
        super().__init__(dataset, **kwargs)
        self.sort_func = _default_sort if sort_func is None else sort_func
        self.res = (
            [self.sort_func(tpls[it]) for it in self.items] if res is None else res
        )
        self.res_y = [self.sort_func(tpls[it], y=True) for it in self.items]
        self.idx_max = np.argmax(self.res)
        self.reverse = reverse
        self.tpls = tpls
        self.bs_max = bs_max
        self.bs_mul = bs_mul

    def get_idxs(self):
        idxs = super().get_idxs()
        if self.shuffle:
            return idxs
        return sorted(idxs, key=lambda i: self.res[i], reverse=not self.reverse)

    def __len__(self):
        if hasattr(self, "_l"):
            return self._l
        return self.n

    def shuffle_fn(self, idxs):
        nprng = np.random.default_rng(42)
        if torch.utils.data.get_worker_info() is not None:
            # try load rng state
            try:
                with open(f"{HOME}/rng-{self.offs}", "rb") as f:
                    seed = pickle.load(f)
                    nprng = np.random.default_rng(seed)
            except Exception as e:
                # print(e)
                pass
        # print("id", self.offs, nprng.permutation([1,2,3,4,5]))
        idxs = nprng.permutation(idxs)
        idx_max = np.extract(idxs == self.idx_max, idxs)[0]
        idxs[0], idxs[idx_max] = idxs[idx_max], idxs[0]
        sz = DL_ADVANCE_BY_BUCKETING
        chunks = [idxs[i : i + sz] for i in range(0, len(idxs), sz)]
        chunks = [sorted(s, key=lambda i: self.res[i], reverse=True) for s in chunks]

        # variable batch bucketing
        def is_adding_one_okay(xlen, ylen, xmax, ymax, bs):
            xmaxok = xlen <= X_MAX * self.bs_mul
            ymaxok = ylen <= Y_MAX * self.bs_mul
            multok = bs * xmax * ymax <= X_MAX * Y_MAX_ONE * self.bs_mul
            bsok = bs <= self.bs_max
            return xmaxok and ymaxok and multok and bsok

        batches = []
        xmax = 0.0
        ymax = 0.0
        xlen = 0.0
        ylen = 0.0
        batch = []
        for chunk in chunks:
            for i, one in enumerate(chunk):
                x, y = int(self.res[one]), int(self.res_y[one])
                if is_adding_one_okay(
                    xlen + x, ylen + y, max(xmax, x), max(ymax, y), len(batch) + 1
                ):
                    xmax = max(xmax, x)
                    ymax = max(ymax, y)
                    xlen += x
                    ylen += y
                    # print("+", one)
                    batch.append(one)
                else:
                    if len(batch) > 0:
                        batches.append(batch)
                    xmax = x
                    ymax = y
                    xlen = x
                    ylen = y
                    # print("+", one)
                    batch = [one]
        if len(batch) > 0:
            batches.append(batch)

        # drop batches with length 1
        # as that does not work with BatchNorm
        batches = list(filter(lambda batch: len(batch) != 1, batches))

        # intra-batch shuffling
        batches = [nprng.permutation(batch).tolist() for batch in batches]

        # batches shuffling
        batches = nprng.permutation(batches).tolist()

        # set _l
        self._l = len(batches)

        # save rng state
        if torch.utils.data.get_worker_info() is not None:
            seed = nprng.integers(0, 2 ** 32 - 1)
            with open(f"{HOME}/rng-{self.offs}", "wb") as f:
                pickle.dump(seed, f)

        return iter(batches)


def pad_collate_float(
    samples,
    dim_T,
    lang,
    print_stats=PRINT_BATCH_STATS,
    raw_audio=False,
    blank_y=0,
    p_y_rand=0.1,
    **kwargs,
) -> Tuple[
    Tuple[FloatTensor, LongTensor, LongTensor, LongTensor],
    Tuple[LongTensor, LongTensor, LongTensor],
]:
    "Function that collect samples and adds padding"

    # set_trace()
    n_samples = len(samples)
    x_lens = [int(s[0].size(dim_T)) for s in samples]
    y_lens = [s[1][1] for s in samples]

    # set limits
    # max_x_len = max(501, max(x_lens))
    # max_y_len = max(16, max(y_lens))
    max_x_len = max(x_lens)
    max_y_len = max(y_lens)

    if print_stats:
        xlens = x_lens
        n_x_elem = max_x_len * n_samples
        print(
            f"xlens | mean: {np.mean(xlens):3.0f}, min: {min(xlens):3.0f}, max: {max(xlens):3.0f}, wasted computation: {(n_x_elem - sum(xlens)) / n_x_elem * 100.:.2f}%"
        )

    def _pad(q, _max):
        to_pad = _max - len(q)
        return q[:_max] + [blank_y] * to_pad

    # X
    x_shape = list(samples[0][0].shape)
    if raw_audio:
        # [N, C, T, H]
        new_x_shape = (n_samples, x_shape[0], max_x_len, x_shape[-1])
        X = torch.zeros(*new_x_shape, dtype=torch.float32)
        for i, s in enumerate(samples):
            t, l = s[0][:, :max_x_len], s[0].size(dim_T)
            X[i, :, :l] = t
    else:
        # [N, T, H, W]
        new_x_shape = (n_samples, max_x_len, *x_shape[2:])
        X = torch.zeros(*new_x_shape, dtype=torch.float32)
        for i, s in enumerate(samples):
            t, l = s[0][:, :max_x_len], s[0].size(dim_T)
            try:
                # pad back
                X[i, :l] = t
                # pad front
                # X[i, max_x_len-l:] = t
            except:
                print(s[0].shape)
                print(t.shape)
                print(l)
                set_trace()
    X_padded = X
    X_lens = LongTensor([s[0].size(dim_T) for s in samples])
    check(X_padded)

    # Y
    Y_padded = LongTensor([_pad(s[1][0], max_y_len) for s in samples])
    Y_lens = LongTensor([s[1][1] for s in samples])
    Y = (Y_padded, Y_lens, X_lens)
    Y = [t.data for t in Y]
    check(Y_padded)

    # X
    X = (X_padded, Y_padded, X_lens, Y_lens)
    X = [t.data for t in X]
    return X, Y


def sorter(tpl, y=False, noop=False):
    if noop:
        return 1
    if y:
        return TupleGetter.ylen(tpl)
    return TupleGetter.xlen(tpl)


def preload_tfms(tfm_funcs, tfm_args):
    tfms = update_tfms(tfm_funcs, tfm_args)
    return tfms


def grab_asr_databunch(
    builder_train,
    builder_valid,
    seed,
    tfms,
    tfms_args,
    sorted_dl_args,
    shuffle_valid,
    pad_collate_float_args={},
    bs_valid=8,
    after_batch=[],
    splitter=partial(RandomSplitter, seed=42),
    name="unknown_asr_dataset",
) -> DataLoaders:

    # get all audio files
    _, idxs_train, tpls_train, _ = builder_train.get()
    _, idxs_valid, tpls_valid, _ = builder_valid.get()

    # convert tuples to numpy
    # to make sure multiprocessing works
    # as desired
    tpls_train = np.array(tpls_train, dtype=np.object).astype(np.string_)
    tpls_valid = np.array(tpls_valid, dtype=np.object).astype(np.string_)

    # pass information to the transforms
    # and create them later
    extra_train = OrderedDict(tfms_args, tpls=tpls_train)
    extra_valid = OrderedDict(tfms_args, tpls=tpls_valid)
    extra_train = OrderedDict(
        extra_train,
        random=True,
    )
    extra_valid = OrderedDict(
        extra_valid,
        random=False,
    )

    # update MySortedDL args
    sorted_dl_args.update(
        OrderedDict(
            create_batch=partial(
                pad_collate_float, dim_T=DIM_TIME, **pad_collate_float_args
            ),
            sort_func=sorter,
            after_batch=after_batch,
        )
    )
    sorted_dl_args_train = sorted_dl_args.copy()
    sorted_dl_args_train["tpls"] = tpls_train

    sorted_dl_args_valid = sorted_dl_args.copy()
    sorted_dl_args_valid["bs"] = bs_valid

    # rnnt: True
    # contrastive: False
    # set this to True for rnnt training to make sure
    # learn.test() happens on unsorted data
    sorted_dl_args_valid["shuffle"] = shuffle_valid
    sorted_dl_args_valid["tpls"] = tpls_valid

    # create tfms
    tfms_train, tfms_valid = update_tfms_multi((tfms,) * 2, (extra_train, extra_valid))

    # Datasets
    dsrc_train = Datasets(idxs_train, tfms_train)
    dsrc_valid = Datasets(idxs_valid, tfms_valid)

    # Special DataLoader
    train = DynamicBucketingDL(dsrc_train.subset(0), **sorted_dl_args_train)
    valid = SortishDL(dsrc_valid.subset(0), **sorted_dl_args_valid)

    # DataLoaders
    db = DataLoaders(train, valid)

    # add some helpful methods
    def statistics(self, norm_file, batches=50, norm_dims=[0, 1], loader=0):
        idx_t_x = 1
        idx_t_y = -1
        means = []
        stds = []
        ratios = []
        feat_norm_means = []
        feat_norm_stds = []
        dl = self.loaders[loader]
        # for (_, b) in zip(range(batches), dl):
        for _ in tqdm.tqdm(range(batches)):
            b = db.one_batch()
            means.append(b[0][0].mean().cpu().numpy())
            stds.append(b[0][0].std().cpu().numpy())
            ratios.append(b[0][0].shape[idx_t_x] / b[0][1].shape[idx_t_y])
            feat_norm_means.append(b[0][0].mean(norm_dims).cpu())
            feat_norm_stds.append(b[0][0].std(norm_dims).cpu())

        print(f"{self.n} Utterances (train)")
        print("Results after Pipeline (train):")
        print(f" mean of means: {np.array(means).mean()}")
        print(f" mean of stds: {np.array(stds).mean()}")
        print(
            f"  data_norm_stats = ({np.array(means).mean()}, {np.array(stds).mean()})"
        )

        plt.hist(means, bins=30)
        plt.title("means")
        plt.show()

        plt.hist(stds, bins=30)
        plt.title("stds")
        plt.show()

        plt.hist([len(l) for l in dl.sample()], bins=20)
        plt.title("# Batches per batch_size")
        plt.show()

    def augmentation(self, n=1, only=None, benchmark=False):
        aud_pipe = tfms_train[0]
        lbl_pipe = tfms_train[1]
        if only is None:
            only = list(range(len(aud_pipe)))
        else:
            if only == -1 or only == "last":
                only = [len(aud_pipe) - 2]
            elif only < 0:
                only = [list(range(len(aud_pipe)))[only]]
            else:
                only = [only]
        if benchmark:
            times = {q.__class__.__name__: 0.0 for q in aud_pipe}
        for i in range(n):
            label = lbl_pipe[0](i)
            if not benchmark:
                print("-" * 75)
                print(f"{i}th item:")
                print("label:", label)
            for j, aud_step in enumerate(aud_pipe[1:]):
                t1 = time.perf_counter()
                if j == 0:
                    item = aud_pipe[0](i)
                else:
                    item = aud_step(item)
                t2 = time.perf_counter()
                if j in only:
                    if benchmark:
                        t = t2 - t1
                        n = aud_step.__class__.__name__
                        times[n] = times[n] + t
                        # s = f"Step ({n.ljust(18)}): {t:.5f}s"
                        # print(s)
                    else:
                        s = f"After step #{j} ({aud_step.__class__}):"
                        print(s)

                    from librosa.display import specshow

                    def pplot(x):
                        if isinstance(x, torch.Tensor) and len(x.shape) == 3:
                            # specshow()?
                            if x.size(-1) == 128:
                                plt.imshow(x.squeeze(0).T.cpu().numpy())
                            else:
                                plt.imshow(x.squeeze(0)[..., :100].cpu().numpy())
                            plt.title(f"mean={x.mean()}, std={x.std()}")
                            plt.show()
                            return str(x.shape)
                        else:
                            raise Exception("not plottable")

                    if not benchmark:
                        desc = chained_try(
                            [
                                pplot,
                                lambda x: (x.shape, x.mean(), x.std()),
                                lambda x: (x.sig.shape, x.sig.mean(), x.sig.std()),
                                lambda x: what(x),
                            ],
                            item,
                        )
                        print("item info:", desc)
                        try:
                            item.show()
                        except:
                            pass
        if benchmark:
            return times

    from types import MethodType

    db.statistics = MethodType(statistics, db)
    db.stats = MethodType(statistics, db)
    db.augmentation = MethodType(augmentation, db)
    db.aug = MethodType(augmentation, db)

    return db


class ASRDatabunch:
    @staticmethod
    def from_config(conf, lang, builder_train, builder_valid, tfms):
        tfms_args = OrderedDict(
            lang=lang,
            channels=conf["channels"],
            sr=conf["sr"],
            feature_sz=conf["model"]["feature_sz"],
            target_sr=conf["sr"],
            audio_len_min_sec=conf["almins"],
            audio_len_max_sec=conf["almaxs"],
            label_min_len=conf["y_min"],
            label_max_len=conf["y_max"],
            win_length=conf["win_length"],
            hop_length=conf["hop_length"],
            delta_win_length=conf["delta_win_length"],
            deltas=conf["deltas"],
            n_foward_frames=conf["n_forward_frames"],
            mfcc_args=conf["mfcc_args"],
            melkwargs=conf["melkwargs"],
            use_extra_features=False,
        )
        sorted_dl_args = OrderedDict(
            bs=None,
            num_workers=conf["num_workers"],
            shuffle=conf["shuffle"],
            reverse=conf["ascending"],
            **conf["dataloader_args"],
        )
        pad_collate_float_args = OrderedDict(
            lang=lang,
            raw_audio=False,
            print_stats=PRINT_BATCH_STATS,
            p_y_rand=0.0,
        )
        after_batch = []
        shuffle_valid = conf["loss"]["type"] != "contrastive"
        print("shuffle_valid:", shuffle_valid)

        db = grab_asr_databunch(
            builder_train,
            builder_valid,
            conf["seed"],
            tfms,
            tfms_args,
            sorted_dl_args,
            shuffle_valid,
            pad_collate_float_args,
            bs_valid=conf["batching"]["batch_size_valid"],
            after_batch=after_batch,
        )
        return db
