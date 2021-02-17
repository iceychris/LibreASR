from functools import partial
import multiprocessing
import math
import sys
import os
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
from fastcore.transform import _TfmMeta

from fastai2_audio.core.signal import *
from fastai2_audio.augment.signal import SignalShifter, AddNoise

import pandas as pd
import numpy as np
from scipy.signal import decimate, resample_poly

import torchaudio

from libreasr.lib.utils import *

DUMMY_AUDIO = AudioTensor(torch.randn(1, 16000) / 100.0, 16000)
DUMMY_TEXT = Text(" dummy ")
DEBUG_TRANSFORMS = False


def warn(msg):
    m = f"WARN | {msg}"
    mn = m + "\n"
    print(m)
    print(m, file=sys.stderr)
    try:
        sys.__stdout__.write(mn)
        sys.__stderr__.write(mn)
        sys.__stdout__.flush()
        sys.__stderr__.flush()
    except:
        pass


def debug(self, inp=None):
    if DEBUG_TRANSFORMS:
        rnd = self.random if hasattr(self, "random") else None
        inp = "" if inp is None else inp.shape
        print(f"Tfm '{self.__class__.__name__}' (random={rnd}, inp={inp}) ran.")


# use sox_io as default backend
torchaudio.set_audio_backend("sox_io")


class OpenAudioSpan(Transform):
    def __init__(self, tpls):
        self.tpls = tpls

    def encodes(self, i) -> AudioTensor:
        tpl = self.tpls[i]
        fname = TupleGetter.file(tpl)

        sr = 16000
        sr_csv = TupleGetter.sr(tpl)
        if sr_csv == -1:
            # determine sr while loading
            use_sr_csv = False
            warn("OpenAudioSpan: sr_csv = -1")
        else:
            # sr is fixed
            use_sr_csv = True
            sr = sr_csv

        # crop
        xstart = TupleGetter.xstart(tpl)
        if not math.isnan(xstart):
            xstart = int((xstart / 1000.0) * sr)
        else:
            xstart = 0
        pad = int(0.5 * sr)
        xlen = TupleGetter.xlen(tpl)
        if int(xlen) == -1 or math.isnan(xlen):
            xlen = 800000
        else:
            xlen = int((xlen / 1000.0) * sr) + pad
        sig, sr_sig = torchaudio.load(fname, frame_offset=xstart, num_frames=xlen)
        return AudioTensor(sig[:1, :], sr=sr_csv if use_sr_csv else sr_sig)


class MyOpenAudio(Transform):
    order = 0

    def __init__(self, tpls, **kwargs):
        self.tpls = tpls
        self.oa = OpenAudioSpan(tpls)

    def encodes(self, i) -> AudioTensor:
        try:
            ai = self.oa(i)
            # TODO add lang
            return ai
        except Exception as e:
            warn(f"encountered an audio error at i={i}: {e}")
            return DUMMY_AUDIO


class ChannelCut(Transform):
    order = 1

    def __init__(self, channels, **kwargs):
        self.n_chans = channels

    def encodes(self, i: AudioTensor) -> AudioTensor:
        debug(self)
        # if not i.size(0) == self.n_chans:
        #     return AudioTensor(i[range(self.n_chans)], i.sr)
        # return i
        return AudioTensor(i[:1], i.sr)


class Resample(Transform):
    order = 2

    def __init__(self, target_sr, **kwargs):
        self.sr = target_sr

    def encodes(self, i: AudioTensor) -> AudioTensor:
        debug(self)
        if i.sr % self.sr == 0 or i.sr < 16000:
            smpl = torchaudio.transforms.Resample(orig_freq=i.sr, new_freq=self.sr)
            res = smpl(i)
        else:
            res = torch.from_numpy(resample_poly(i.numpy()[0], self.sr, i.sr)).float()[
                None
            ]
        return AudioTensor(res, self.sr)


class Dummy(Transform):
    order = 3

    def __init__(self, **kwargs):
        pass

    def encodes(self, i: AudioTensor) -> AudioTensor:
        return i


class MultiCrop(Transform):
    order = 4

    def __init__(self, random=True, duration=1.5, crops=2, silence=0.25, **kwargs):
        self.random = random
        self.d = duration
        self.crops = crops
        self.silence = silence
        assert crops >= 2

    def encodes(self, i: AudioTensor) -> AudioTensor:
        debug(self)
        ds = int(self.d * i.sr)
        ss = int(self.silence * i.sr)
        n = self.crops
        fs = n * ds + (n - 1) * ss
        offset = 0
        chans = i.size(0)
        if self.random:
            _max = i.size(1) - n * ds
            if not _max <= 0:
                offset = torch.randint(low=0, high=_max, size=(1,)).item()
        ts = []
        for j in range(n):
            _from, _to = offset + ds * j, offset + ds * (j + 1)
            ts.append(i[:, _from:_to])
            if j != n - 1:
                ts.append(torch.zeros(chans, ss))
        return AudioTensor(torch.cat(ts, dim=1), i.sr)


class ResamplePoly(Transform):
    order = 3
    "This takes > ~30ms for one item"

    def __init__(self, random=True, delta=20, **kwargs):
        self.random = random
        self.delta = delta

    def encodes(self, i: AudioTensor) -> AudioTensor:
        debug(self)
        if not self.random:
            return i
        sig, sr = i, i.sr
        rand = torch.randint(low=-self.delta, high=self.delta + 1, size=(1,))
        new_audio = torch.from_numpy(
            resample_poly(sig.numpy()[0], 200, 200 + int(rand))
        ).float()[None]
        return AudioTensor(new_audio, i.sr)


class ChangeVolume(Transform):
    order = 4
    "Change the overall (and elementwise) volume of an AudioTensor"

    def __init__(self, random=True, pcent=0.1, **kwargs):
        self.random = random
        self.pcent = pcent

    def encodes(self, i: AudioTensor) -> AudioTensor:
        debug(self)
        if not self.random:
            return i
        aud = i
        rand = FloatTensor([1]).uniform_(1.0 - self.pcent, 1.0 + self.pcent)
        return AudioTensor(aud * rand, i.sr)


class MyAddNoise(Transform):
    order = 5

    def __init__(self, random=True, *args, **kwargs):
        self.random = random
        self.tfm = AddNoise(*args, **kwargs)

    def encodes(self, i: AudioTensor) -> AudioTensor:
        debug(self)
        if not self.random:
            return i
        return self.tfm(i)


class MySignalShifter(Transform):
    order = 6

    def __init__(self, random=True, *args, **kwargs):
        self.random = random
        self.tfm = SignalShifter(*args, **kwargs)

    def encodes(self, i: AudioTensor) -> AudioTensor:
        debug(self)
        if not self.random:
            return i
        return self.tfm(i)


class PadderCutter(Transform):
    order = 10

    def __init__(self, audio_len_min_sec=1.0, audio_len_max_sec=7.0, **kwargs):
        self.al_min = audio_len_min_sec
        self.al_max = audio_len_max_sec

    def encodes(self, ai: AudioTensor) -> AudioTensor:
        debug(self)
        sig, sr = ai.data, ai.sr
        min_sec = self.al_min
        max_sec = self.al_max
        # pad | first make sure audio is at least one second
        if sig.size(1) <= sr * min_sec:
            sig = torch.cat(
                [sig, torch.zeros(sig.size(0), int(sr * min_sec) - sig.size(1))], -1
            )
        # cut | not too long
        if sig.size(1) / sr >= max_sec:
            sig = sig[:, : int(sr * max_sec)]
        return sig


class FixRawAudio(Transform):
    order = 11

    def __init__(self, **kwargs):
        pass

    def encodes(self, t: Tensor) -> None:
        debug(self)
        return t.data.unsqueeze(-1).unsqueeze(-1)


def pad_around(t1, _prev=0, _next=0, l=0.01, sr=16000):
    l = int(sr * l)
    a, b, c = None, t1, None

    if torch.is_tensor(_prev):
        a = _prev[:, -l:]
    else:
        a = torch.empty(0)
        b = torch.nn.functional.pad(b[None], (l, 0), mode="reflect")[0]

    if torch.is_tensor(_next):
        c = _next[:, :l]
    else:
        c = torch.empty(0)
        b = torch.nn.functional.pad(b[None], (0, l), mode="reflect")[0]

    return torch.cat([a, b, c], 1)


class StreamPreprocess(Transform):
    order = 15

    def __init__(
        self,
        sr,
        win_length=0.025,
        hop_length=0.01,
        **kwargs,
    ):
        self.sr = sr
        self.hop_length = hop_length

    def encodes(self, t) -> None:
        debug(self)
        t = t[:, : -int(0.07 * self.sr)]
        return t


class TransformTime(Transform):
    order = 20

    def __init__(
        self,
        mfcc_args,
        melkwargs,
        n_forward_frames=4,
        sr=16000,
        win_length=0.025,
        hop_length=0.01,
        deltas=2,
        delta_win_length=2,
        use_extra_features=False,
        log_mels=True,
        **kwargs,
    ):
        length = win_length
        shift = hop_length
        self.wl = int(length * sr)
        self.hl = int(shift * sr)
        self.op = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            win_length=self.wl,
            hop_length=self.hl,
            **melkwargs,
            **mfcc_args,
        )
        self.delta_op = torchaudio.transforms.ComputeDeltas(delta_win_length)
        self.deltas = deltas
        self.next_n = n_forward_frames
        self.use_extra_features = use_extra_features
        self.log_mels = log_mels

    def encodes(self, ai) -> None:
        debug(self)
        # returns [C, T, H]
        sig = ai.data

        # use mfcc & compute deltas
        with torch.no_grad():
            res = self.op(sig)
            if self.log_mels:
                log_offset = 1e-6
                res = torch.log(res + log_offset)
        d = res
        ds = [res]
        for _ in range(self.deltas):
            q = self.delta_op(d)
            ds.append(q)
            d = q
        res = torch.cat(ds, 1)
        res = res.permute(0, 2, 1)  # .detach()

        # only take first channel
        return res[:1]


class StreamPostprocess(Transform):
    "Cut the spectrogram in 3 parts and return only the middle"
    order = 25

    def __init__(
        self,
        n_stack,
        **kwargs,
    ):
        self.n_stack = n_stack

    def encodes(self, spectro) -> None:
        debug(self)
        shape = spectro.shape
        t = shape[1]
        l = t // 3
        a = l + 1
        spectro_new = spectro[:, a:, :][:, : self.n_stack, :]
        return spectro_new


class MyCutFrames(Transform):
    "Cut a spectrogram at the start (front) and the end (back)."
    order = 30

    def __init__(self, random=True, max_front=2, max_back=2, **kwargs):
        self.random = random
        self.max_front = max_front
        self.max_back = max_back

    def encodes(self, spectro) -> None:
        debug(self)
        if not self.random:
            return spectro
        f = random.randint(0, self.max_front)
        b = random.randint(0, self.max_back)
        spectro = spectro[:, f:, ...]
        if b != 0:
            spectro = spectro[:, :-b, ...]
        return spectro


class MyMaskTime(Transform):
    order = 31

    def __init__(
        self,
        random=True,
        num_masks=1,
        size=20,
        start=None,
        val=None,
        adaptive=True,
        **kwargs,
    ):
        self.random = random
        self.num_masks = num_masks
        self.size = size
        self.start = start
        self.val = val
        self.adaptive = adaptive

    def encodes(self, spectro) -> None:
        """Google SpecAugment time masking from https://arxiv.org/abs/1904.08779."""
        debug(self, spectro)
        if not self.random:
            return spectro
        num_masks = self.num_masks
        size = self.size
        start = self.start
        val = self.val
        sg = spectro.clone()
        channel_mean = sg.contiguous().view(sg.size(0), -1).mean(-1)[:, None, None]
        mask_val = channel_mean if val is None else val
        c, y, x = sg.shape

        def mk_masks(_min, _max):
            for _ in range(num_masks):
                mask = torch.ones(size, x, device=spectro.device) * mask_val
                start = random.randint(_min, _max - size)
                if not 0 <= start <= y - size:
                    raise ValueError(
                        f"Start value '{start}' out of range for AudioSpectrogram of shape {sg.shape}"
                    )
                sg[:, start : start + size, :] = mask

        if self.adaptive:
            sz = 100
            for a in range(0, y, sz):
                _min, _max = a, min(a + sz, y)
                if _max - _min != sz:
                    continue
                mk_masks(_min, _max)
        else:
            mk_masks(0, y)
        return sg


class MyMaskFreq(Transform):
    order = 32

    def __init__(
        self,
        random=True,
        num_masks=1,
        size=20,
        start=None,
        val=None,
        adaptive=False,
        **kwargs,
    ):
        self.random = random
        self.num_masks = num_masks
        self.size = size
        self.start = start
        self.val = val
        self.adaptive = adaptive
        self.kwargs = kwargs

    def encodes(self, spectro) -> None:
        debug(self)
        if not self.random:
            return spectro
        sg = spectro.clone()
        sg = torch.einsum("...ij->...ji", sg)
        sg = MyMaskTime(
            self.random,
            self.num_masks,
            self.size,
            self.start,
            self.val,
            self.adaptive,
            **self.kwargs,
        )(sg)
        return torch.einsum("...ij->...ji", sg)


class StackDownsample(Transform):
    order = 80

    def __init__(self, n_stack=6, downsample=3, **kwargs):
        self.n_stack = n_stack
        self.downsample = downsample

    def encodes(self, t: Tensor) -> None:
        debug(self)
        # stack & downsample using unfold
        uf = t.unfold(-2, self.n_stack, self.downsample).contiguous()
        viewed = uf.view(uf.size(0), uf.size(1), -1).contiguous()
        return viewed


class FixDimensions(Transform):
    order = 99

    def __init__(self, **kwargs):
        pass

    def encodes(self, t: Tensor) -> None:
        debug(self)
        return t.data.unsqueeze(-1)


class Buffer(Transform):
    "Buffer incoming tensors and concat them when n_buffer is reached"
    order = 150

    def __init__(self, n_buffer, **kwargs):
        self.n_buffer = n_buffer
        self.saved = []

    def encodes(self, t: Tensor) -> None:
        debug(self)
        self.saved.append(t)
        if len(self.saved) == self.n_buffer:
            catted = torch.cat(self.saved, dim=1)
            self.saved.clear()
            # print("Buffer catted:", catted.shape)
            return catted[0]
        return None


class MyOpenLabel(Transform):
    order = 0

    def __init__(self, tpls, **kwargs):
        self.tpls = tpls

    def encodes(self, i) -> str:
        debug(self)
        return TupleGetter.label(self.tpls[i])


class PadCutLabel(Transform):
    order = 1

    def __init__(self, label_min_len=1, label_max_len=80, **kwargs):
        self.lmin = label_min_len
        self.lmax = label_max_len

    def encodes(self, o: str) -> str:
        debug(self)
        o = sanitize_str(o)
        if len(o) < self.lmin:
            return o + " " * (self.lmin - len(o))
        return o[: self.lmax]


class MyNumericalize(Transform):
    order = 2

    def __init__(self, lang, random, **kwargs):
        self.lang = lang
        self.random = random

    def encodes(self, o: str):
        debug(self)
        if self.random:
            res = self.lang.numericalize(o)  # , dropout=0.25)
        else:
            res = self.lang.numericalize(o)
        assert 0 not in res
        return res


class AddLen(Transform):
    order = 3

    def __init__(self, **kwargs):
        pass

    def encodes(self, o):
        debug(self)
        return (o, len(o))


def update_tfms(l, extra_args):
    "this is running twice but somehow works"
    if l is None:
        return l
    new_l = []
    for x in l:
        if not isinstance(x, _TfmMeta) and not callable(x):
            # recurse
            new_l.append(update_tfms(x, extra_args))
        else:
            # update & build x
            if x is not None:
                try:
                    new_l.append(partial(x, **extra_args)())
                except Exception as e:
                    print("Exception at:", x)
                    print(e)
                    set_trace()
            else:
                new_l.append(None)
    return new_l


def update_tfms_multi(list_of_tfms, list_of_extra_args):
    return [
        update_tfms(tfms, extra_args)
        for (tfms, extra_args) in zip(list_of_tfms, list_of_extra_args)
    ]
