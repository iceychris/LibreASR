import contextlib
import operator
import time
import random
from queue import PriorityQueue
from functools import partial, lru_cache
import itertools
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F

import numpy as np

from fastai.vision.models.xresnet import xresnet18
from fastai.layers import Debugger, ResBlock
from fastai.torch_core import Module
from fastai.learner import CancelBatchException

from IPython.core.debugger import set_trace

from libreasr.lib.utils import *
from libreasr.lib.layers import StackedRNN, DualModeMultiHeadSelfAttention
from libreasr.lib.lm import LMFuser, LMFuserBatch
from libreasr.lib.defaults import (
    LM_ALPHA,
    LM_TEMP,
    MODEL_TEMP,
    DEFAULT_MAX_ITERS,
    DEFAULT_BEAM_SEARCH_OPTS,
)
from libreasr.modules.encoders import (
    SlimEncoder,
    LinearTransformerEncoder,
    Wav2Vec2Encoder,
)
from libreasr.lib.inference.beamsearch import (
    Beamsearch,
)
from libreasr.modules.ssl import pi_loss


class ResidualAdapter(Module):
    """
    ResidualAdapter according to
    https://ai.googleblog.com/2019/09/large-scale-multilingual-speech.html?m=1
    """

    def __init__(
        self, hidden_sz, projection="fcnet", projection_factor=3.2, activation=F.relu
    ):
        self.hidden_sz = hidden_sz

        self.activation = activation()
        self.ln = nn.LayerNorm(hidden_sz)

        if projection == "conv":
            pass
        else:
            bottleneck_sz = int(hidden_sz / projection_factor)
            # find next multiple of 8 for performance
            bottleneck_sz = bottleneck_sz + (8 - bottleneck_sz % 8)
            self.down = nn.Linear(hidden_sz, bottleneck_sz)
            self.up = nn.Linear(bottleneck_sz, hidden_sz)

    def forward(self, x):
        inp = x

        # layer norm
        x = self.ln(x)

        # down projection
        x = self.down(x)
        x = self.activation(x)

        # up projection
        x = self.up(x)

        # residual connection
        return x + inp


class SpecAugment(Module):
    """
    A differentiable implementation of
    Google SpecAugment from https://arxiv.org/abs/1904.08779.
    Contains time and frequency masking.
    """

    def __init__(
        self,
        time_mask_n=2,
        time_mask_sz=4,
        freq_mask_n=4,
        freq_mask_sz=2,
        start=None,
        val=None,
        enable_eval=False,
    ):
        self.time_mask_n = time_mask_n
        self.time_mask_sz = time_mask_sz
        self.freq_mask_n = freq_mask_n
        self.freq_mask_sz = freq_mask_sz
        self.start = start
        self.val = val
        self.ee = enable_eval

    def mask_time(self, spectro, adaptive=True):
        num_masks = self.time_mask_n
        size = self.time_mask_sz
        start = self.start
        val = self.val
        sg = spectro.clone()
        channel_mean = sg.contiguous().view(sg.size(0), -1).mean(-1)[:, None, None]
        mask_val = channel_mean if val is None else val
        c, x, y = sg.shape

        def mk_masks(_min, _max):
            for _ in range(num_masks):
                mask = torch.ones(x, size, device=spectro.device) * mask_val
                start = random.randint(_min, _max - size)
                if not 0 <= start <= y - size:
                    raise ValueError(
                        f"Start value '{start}' out of range for AudioSpectrogram of shape {sg.shape}"
                    )
                sg[:, :, start : start + size] = mask

        if adaptive:
            sz = 100
            for a in range(0, y, sz):
                _min, _max = a, min(a + sz, y)
                if _max - _min != sz:
                    continue
                mk_masks(_min, _max)
        else:
            mk_masks(0, y)
        return sg

    def mask_freq(self, spectro):
        sg = spectro.clone()
        sg = torch.einsum("...ij->...ji", sg)
        sg = self.mask_time(sg, adaptive=False)
        return torch.einsum("...ij->...ji", sg)

    def forward(self, x, mask_time_adaptive=True, no_augmentation=False):
        if ((not self.training) and (not self.ee)) or no_augmentation:
            return x
        x = self.mask_time(x, adaptive=mask_time_adaptive)
        x = self.mask_freq(x)
        return x


class Preprocessor(Module):
    def __init__(
        self,
        enable=True,
        sr=16000,
        n_mels=80,
        n_fft=1024,
        downsample=8,
        stack=8,
        trainable=False,
        **kwargs,
    ):
        from nnAudio.Spectrogram import MelSpectrogram

        self.enable = enable
        self.sr = sr
        self.hl = int(0.01 * sr)
        self.spec = MelSpectrogram(
            sr=sr,
            trainable_mel=trainable,
            trainable_STFT=trainable,
            n_fft=n_fft,
            n_mels=n_mels,
            win_length=int(0.025 * sr),
            hop_length=self.hl,
            verbose=False,
        )
        self.specaugment = SpecAugment(**kwargs)
        self.n_mels = n_mels
        self.downsample = downsample
        self.stack = stack

    def to_jit(self):
        jittify_mel_spectrogram(self.spec)
        jittify_stft(self.spec.stft)
        self.specaugment = None
        self.__class__ = PreprocessorJit

    def param_groups(self):
        return [p for p in self.parameters() if p.requires_grad]

    def stack_downsample(self, x):
        x = x.unfold(-2, self.stack, self.downsample).contiguous()
        x = x.view(x.size(0), x.size(1), -1).contiguous()
        return x

    def _adjust_lengths(self, x, xl):
        xl = ((xl / self.hl).floor() + 1).long()
        xl = xl // self.stack
        xl = torch.clamp(xl, min=1, max=x.size(1))
        if xl.max() < x.size(1):
            xl[-1] = x.size(1)
        return xl

    def augment(self, x, xl):
        x = self.spec(x[..., :, 0, 0])

        # augment using mask
        ones = torch.ones_like(x)
        mask = self.specaugment(ones)

        # permute
        x = x.permute(0, 2, 1).contiguous()
        mask = mask.permute(0, 2, 1).contiguous()
        raw_x, raw_mask = x, mask

        # stack/downsample
        x = self.stack_downsample(x)
        mask = self.stack_downsample(mask)

        # calc lengths
        xl = self._adjust_lengths(x, xl)
        if self.training:
            pass
        return x, xl, mask, (raw_x, raw_mask)

    def forward(self, x, xl=None, inference=False, **kwargs):
        if not self.enable:
            if xl is not None:
                return x, xl
            else:
                return x
        x = self.spec(x[..., :, 0, 0])
        x = self.specaugment(x, **kwargs)
        x = x.permute(0, 2, 1).contiguous()
        if not inference:
            x = self.stack_downsample(x)
        if xl is not None:
            xl = self._adjust_lengths(x, xl)
            return x, xl
        return x


def jittify_stft(stft):
    from nnAudio.Spectrogram import STFT

    # fix padding issue
    stft.padder = nn.ReflectionPad1d(stft.pad_amount)

    # replace class
    class STFTJit(STFT):
        def forward(self, x: Tensor) -> Tensor:
            # assume: self.freq_bins is None
            output_format = self.output_format
            freq_bins: int = int(10e6)
            trainable: bool = self.trainable
            num_samples = x.size(-1)

            # assume: x.shape == [B, T]
            x = x[:, None, :]

            # assume: center + reflect
            if num_samples < self.pad_amount:
                raise AssertionError(
                    "Signal length shorter than reflect padding length (n_fft // 2)."
                )
            x = self.padder(x)

            spec_imag = F.conv1d(x, self.wsin, stride=self.stride)
            spec_real = F.conv1d(
                x, self.wcos, stride=self.stride
            )  # Doing STFT by using conv1d

            # remove redundant parts
            spec_real = spec_real[:, :freq_bins, :]
            spec_imag = spec_imag[:, :freq_bins, :]

            # assume: output_format=='Magnitude'
            spec = spec_real.pow(2) + spec_imag.pow(2)
            if trainable:
                return torch.sqrt(
                    spec + 1e-8
                )  # prevent Nan gradient when sqrt(0) due to output=0
            else:
                return torch.sqrt(spec)

    stft.__class__ = STFTJit


def jittify_mel_spectrogram(spec):
    from nnAudio.Spectrogram import MelSpectrogram

    # replace class
    class MelSpectrogramJit(MelSpectrogram):
        def forward(self, x: Tensor) -> Tensor:
            spec = self.stft(x) ** self.power
            melspec = torch.matmul(self.mel_basis, spec)
            return melspec

    spec.__class__ = MelSpectrogramJit


class PreprocessorJit(Preprocessor):
    def forward(self, x: Tensor, xl: Tensor, inference: bool) -> Tuple[Tensor, Tensor]:
        x = x[..., :, 0, 0]
        x = self.spec(x)
        x = x.permute(0, 2, 1).contiguous()
        if not inference:
            x = self.stack_downsample(x)
        # TODO: use correct formula for this
        fac = self.sr // 100
        xl = torch.clamp(xl // (fac * self.downsample), min=1, max=x.size(1))
        return x, xl


class Encoder(Module):
    def __init__(
        self,
        feature_sz,
        hidden_sz,
        out_sz,
        dropout=0.01,
        dropout_input=0.0,
        dropout_inner=0.0,
        num_layers=2,
        trace=True,
        device="cuda:0",
        rnn_type="LSTM",
        norm="bn",
        attention=False,
        use_tmp_state_pcent=0.9,
        **kwargs,
    ):
        self.num_layers = num_layers
        self.drop_input = nn.Dropout(dropout_input)
        self.input_norm = nn.LayerNorm(feature_sz)
        self.rnn_stack = StackedRNN(
            feature_sz,
            hidden_sz,
            num_layers,
            rnn_type=rnn_type,
            reduction_indices=[],  # 1
            reduction_factors=[],  # 2
            rezero=False,
            utsp=use_tmp_state_pcent,
            norm=norm,
            dropout=dropout_inner,
        )
        if attention:
            self.attention = DualModeMultiHeadSelfAttention(
                hidden_sz,
                n_heads=8,
                window_size=8,  # window size. 512 is optimal, but 256 or 128 yields good enough results
                dropout=0.1,  # post-attention dropout
                exact_windowsize=True,  # if this is set to true, in the causal setting, each query will see at maximum the number of keys equal to the window size
                autopad=True,
            )
        else:
            self.attention = nn.Sequential()
        self.drop = nn.Dropout(dropout)
        if not hidden_sz == out_sz:
            self.linear = nn.Linear(hidden_sz, out_sz)
        else:
            self.linear = nn.Sequential()

    def param_groups(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x, state=None, lengths=None, return_state=False):
        x = x.reshape((x.size(0), x.size(1), -1))
        x = self.drop_input(x)
        x = self.input_norm(x)
        x, state = self.rnn_stack(x, state=state, lengths=lengths)
        x = self.attention(x)
        x = self.drop(x)
        x = self.linear(x)
        if return_state:
            return x, state
        return x


class Predictor(Module):
    def __init__(
        self,
        vocab_sz,
        embed_sz,
        hidden_sz,
        out_sz,
        dropout=0.01,
        num_layers=2,
        blank=0,
        rnn_type="LSTM",
        norm="bn",
        final_norm=False,
        use_tmp_state_pcent=0.9,
        **kwargs,
    ):
        self.vocab_sz = vocab_sz
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_sz, embed_sz, padding_idx=blank)
        if not embed_sz == hidden_sz:
            self.ffn = nn.Linear(embed_sz, hidden_sz)
        else:
            self.ffn = nn.Sequential()
        self.rnn_stack = StackedRNN(
            hidden_sz,
            hidden_sz,
            num_layers,
            rnn_type=rnn_type,
            rezero=False,
            utsp=use_tmp_state_pcent,
            norm=norm,
            dropout=0.0,
        )
        self.drop = nn.Dropout(dropout)
        if not hidden_sz == out_sz:
            self.linear = nn.Linear(hidden_sz, out_sz)
        else:
            self.linear = nn.Sequential()
        if final_norm:
            self.final_norm = nn.LayerNorm(out_sz)
        else:
            self.final_norm = nn.Sequential()

    def to_jit(self):
        self.rnn_stack.to_jit()
        self.__class__ = PredictorJit

    def initial_state(self):
        return self.rnn_stack.initial_state()

    def param_groups(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x, state=None, lengths=None):
        x = self.embed(x)
        x = self.ffn(x)
        x, state = self.rnn_stack(x, state=state, lengths=lengths)
        x = self.drop(x)
        x = self.linear(x)
        x = self.final_norm(x)
        return x, state


class PredictorJit(Predictor):
    def forward(
        self, x: Tensor, lengths: Tensor, state: Optional[List[Tuple[Tensor, Tensor]]]
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        # TODO: remove or use `lengths` properly...
        x = self.embed(x)
        x = self.ffn(x)
        x, new_states = self.rnn_stack(x, lengths, state)
        x = self.drop(x)
        x = self.linear(x)
        return x, new_states


class Joint(Module):
    def __init__(
        self,
        out_sz,
        joint_sz,
        vocab_sz,
        method,
        arch="regular",
        reversible=False,
        bias=True,
        act="tanh",
        dropout=0.0,
        inplace=True,
    ):
        assert dropout == 0.0, "Dropout is not used in Joint"
        assert not reversible
        self.method = method
        self.reversible = reversible
        if act == "tanh":
            activation = nn.Tanh(inplace=inplace)
        elif act == "relu":
            activation = nn.ReLU(inplace=inplace)
        else:
            raise Exception(f"No such activation '{act}'")

        # custom LibreASR
        if arch == "regular":
            self.joint = nn.Sequential(
                nn.Linear(out_sz, joint_sz, bias=bias),
                activation,
                nn.Linear(joint_sz, vocab_sz, bias=bias),
            )

        # less memory heavy?
        elif arch == "slim":
            assert act == "relu", "Only ReLU supported for Joint with arch=slim"
            self.joint = nn.Sequential(
                activation,
                nn.Linear(out_sz, vocab_sz, bias=bias),
            )

    def to_jit(self):
        self.__class__ = JointJit

    def param_groups(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(
        self,
        h_pred,
        h_enc,
        temp=MODEL_TEMP,
        softmax=True,
        log=True,
        normalize_grad=False,
        expand=False,
    ):
        # expand: pass [N, T, U, H] to the joint network
        if expand:
            N, U, H_p = h_pred.shape
            N, T, H_e = h_enc.shape
            sz_e = (N, T, U, H_e)
            sz_p = (N, T, U, H_p)
            h_enc = h_enc.unsqueeze(2).expand(sz_e).contiguous()
            h_pred = h_pred.unsqueeze(1).expand(sz_p).contiguous()

        # https://arxiv.org/pdf/2011.01576.pdf
        if normalize_grad:
            h_pred.register_hook(lambda grad: grad / h_enc.size(1))
            h_enc.register_hook(lambda grad: grad / h_pred.size(1))
        if self.method == "add":
            x = h_pred + h_enc
        elif self.method == "mul":
            x = h_pred * h_enc
        elif self.method == "comb":
            x = torch.cat((h_pred + h_enc, h_pred * h_enc), dim=-1)
        elif self.method == "concat":
            x = torch.cat((h_pred, h_enc), dim=-1)
        else:
            raise Exception("No such method")
        if self.reversible:
            if self.training:
                x = self.joint(x)
            else:
                self.joint.disable = True
                x = self.joint(x)
                self.joint.disable = False
        else:
            x = self.joint(x)
        if softmax:
            f = F.softmax
            if log:
                f = F.log_softmax
            x = f(x / temp, dim=-1)
        return x


class JointJit(Joint):
    def forward(
        self, h_pred: Tensor, h_enc: Tensor, softmax: bool, log: bool, expand: bool
    ) -> Tensor:
        if expand:
            N, U, H_p = h_pred.shape
            N, T, H_e = h_enc.shape
            sz_e = (N, T, U, H_e)
            sz_p = (N, T, U, H_p)
            h_enc = h_enc.unsqueeze(2).expand(sz_e).contiguous()
            h_pred = h_pred.unsqueeze(1).expand(sz_p).contiguous()
        x = torch.cat((h_pred, h_enc), dim=-1)
        x = self.joint(x)
        if softmax:
            if log:
                x = F.log_softmax(x, dim=-1)
            else:
                x = F.softmax(x, dim=-1)
        return x


def get_model(conf, *args, **kwargs):
    clazz = eval(conf["model"]["name"])
    conf_ns = conf.get("training", {}).get("noisystudent", {})
    use_noisystudent = conf_ns.get("enable", False)
    if use_noisystudent:
        ovr = conf_ns.get("overrides", {})
        conf_overrides_t = ovr.get("teacher", {})
        conf_overrides_s = ovr.get("student", {})
        conf_teacher = update(conf, conf_overrides_t, deepcpy=True)
        conf_student = update(conf, conf_overrides_s, deepcpy=True)
        teacher = clazz.from_config(conf_teacher, *args, **kwargs)
        student = clazz.from_config(conf_student, *args, **kwargs)
        extra = conf_ns.get("extra", {})
        ns = NoisyStudent(teacher, student, **extra)
        return ns
    return clazz.from_config(conf, *args, **kwargs)


class RNNTLoss(Module):
    """
    RNN-T loss function
    with auxiliary CTC loss
    """

    def __init__(self, gather=True, aux_ctc=False, reduction=1.0, zero_nan=False):
        from warp_rnnt import rnnt_loss

        self.rnnt_loss = partial(
            rnnt_loss, average_frames=False, gather=gather, fastemit_lambda=0.002
        )
        self.reduction = reduction
        self.zero_nan = zero_nan
        self.aux_ctc = aux_ctc
        if aux_ctc:
            self.ctc_loss = nn.CTCLoss(zero_infinity=True)

    def forward(self, x, y, xl, yl, ctc_out=None, return_dict=True):
        # trim lens
        xl = torch.clamp(xl // self.reduction, min=1, max=x.size(1))

        # fix types
        y = y.type(torch.int32)
        xl = xl.type(torch.int32)
        yl = yl.type(torch.int32)

        # avoid NaN
        if self.zero_nan:
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        # dict of losses
        losses = dict()

        # calc rnnt loss
        rnnt_loss_value = self.rnnt_loss(x, y, xl, yl)
        losses["rnnt_loss"] = rnnt_loss_value
        losses["loss"] = rnnt_loss_value

        # calc ctc loss
        ctc_loss_value = 0.0
        if self.aux_ctc and ctc_out is not None:
            ctc_out = ctc_out.permute(1, 0, 2).contiguous().log_softmax(-1)
            y = y.type(torch.long)
            xl = xl.type(torch.long)
            yl = yl.type(torch.long)
            ctc_loss_value = self.ctc_loss(ctc_out, y, xl, yl)
            losses["ctc_loss"] = ctc_loss_value
            losses["loss"] = (rnnt_loss_value + ctc_loss_value) / 2.0
        if return_dict:
            return losses
        return losses["loss"]


class NoisyStudentLoss(Module):
    def __init__(
        self,
        temp=1.0,
        alpha=1.0,
        beta=1e-3,
        gamma=1e-3,
        interpolate=True,
        use_mse=True,
        shift=0,
    ):
        self.temp = temp
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.interpolate = interpolate
        self.use_mse = use_mse
        self.shift = shift
        self.rnnt_loss = RNNTLoss()
        self.mse = torch.nn.MSELoss(reduction="none")

    def _gather_one(self, x, labels, blank=0):
        N, T, U, V = x.size()
        index = torch.full([N, T, U, 2], blank, device=labels.device, dtype=torch.long)
        index[:, :, : U - 1, 1] = labels.unsqueeze(dim=1)
        x = x.gather(dim=3, index=index)
        return x

    def _gather(self, t, s, labels, blank=0):
        t = self._gather_one(t, labels, blank=blank)
        s = self._gather_one(s, labels, blank=blank)
        return t, s

    def _shift(self, t, s):
        if self.shift == 0:
            return t, s
        N, _, U, V = s.shape
        s = s[:, self.shift :]
        pad = torch.zeros((N, self.shift, U, V), dtype=s.dtype, device=s.device)
        s = torch.cat([s, pad], dim=1)
        return t, s

    def _kd_kldiv_loss(self, t, s, dim=-1, mul_temp=False):
        p_t = F.softmax(t / self.temp, dim=dim)
        p_s = F.log_softmax(s / self.temp, dim=dim)
        kd_loss_value = F.kl_div(p_s, p_t, reduction="none").sum((1, 2, 3))
        if mul_temp:
            kd_loss_value *= self.temp ** 2
        return kd_loss_value

    def _l2_loss(self, e1, e2, dims=(1, 2)):
        return self.mse(e1, e2).mean(dims)

    def _l2_loss_encoder(self, e1, e2, dims=(1, 2), swap=True, mean=True):
        if self.interpolate:
            if swap:
                e1, e2 = e2, e1
            e1 = e1.permute(0, 2, 1).contiguous()
            e2 = e2.permute(0, 2, 1).contiguous()
            e1 = F.interpolate(e1, size=e2.size(-1), mode="nearest")
        if mean:
            return self.mse(e1, e2).mean(dims)
        return self.mse(e1, e2).sum(dims)

    def forward(self, x, y, xl, yl, t_logits, s_logits, t_e, s_e, **kwargs):
        # dict of losses
        losses = dict()

        # rnnt loss
        if self.alpha > 0.0:
            rnnt_logits = s_logits.log_softmax(dim=-1)
            rnnt_loss_value = self.rnnt_loss(rnnt_logits, y, xl, yl, return_dict=False)
            losses["rnnt_loss"] = rnnt_loss_value
        else:
            rnnt_loss_value = 0.0

        # encoder l2 loss
        if self.gamma > 0.0:
            l2_loss_value = self._l2_loss_encoder(t_e, s_e)
            losses["l2_loss"] = l2_loss_value
        else:
            l2_loss_value = 0.0

        # teacher student knowledge distillation loss
        if self.beta > 0.0:
            t_logits, s_logits = self._gather(t_logits, s_logits, labels=y)
            t_logits, s_logits = self._shift(t_logits, s_logits)
            if self.use_mse:
                kd_loss_value = self._l2_loss(t_logits, s_logits, dims=(1, 2, 3))
            else:
                kd_loss_value = self._kd_kldiv_loss(t_logits, s_logits)
            losses["kd_loss"] = kd_loss_value
        else:
            kd_loss_value = 0.0

        # sum up
        losses["loss"] = (
            rnnt_loss_value * self.alpha
            + l2_loss_value * self.gamma
            + kd_loss_value * self.beta
        )

        return losses


class Benchmark(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        torch.cuda.synchronize()
        now = time.time()
        print(f"{self.name}: {now - self.start:.3f}s")


class Transducer(Module):
    def __init__(
        self,
        feature_sz,
        embed_sz,
        vocab_sz,
        hidden_sz,
        out_sz_enc,
        out_sz_pre,
        joint_sz,
        lang,
        l_e=6,
        l_p=2,
        blank=0,
        perf=False,
        act=F.relu,
        use_tmp_bos=True,
        use_tmp_bos_pcent=0.99,
        preprocessor_kwargs={},
        encoder_kwargs={},
        predictor_kwargs={},
        joint_kwargs={},
        enable_encoder=True,
        enable_predictor=True,
        enable_joint=True,
        learnable_stft=False,
        device="cpu",
        loss=False,
        benchmark=False,
        auxiliary_ctc_loss=False,
        **kwargs,
    ):
        self.preprocessor = (
            Preprocessor(**preprocessor_kwargs) if learnable_stft else Noop()
        )
        if enable_encoder:
            self.encoder = eval(encoder_kwargs["name"])(
                feature_sz,
                hidden_sz=hidden_sz,
                out_sz=out_sz_enc,
                **encoder_kwargs,
            )
        if enable_predictor:
            self.predictor = eval(predictor_kwargs["name"])(
                vocab_sz,
                embed_sz=embed_sz,
                hidden_sz=hidden_sz,
                out_sz=out_sz_pre,
                **predictor_kwargs,
            )
        if enable_joint:
            assert joint_kwargs.get("method", "concat") == "concat"
            o_sz = out_sz_enc + out_sz_pre
            self.joint = Joint(o_sz, joint_sz, vocab_sz, **joint_kwargs)
        self.enable_predictor = enable_predictor
        self.enable_joint = enable_joint
        self.feature_sz = feature_sz
        self.lang = lang
        self.blank = blank
        # TODO: dont hardcode
        self.bos = 2
        self.bos_cache = {}
        self.use_tmp_bos = use_tmp_bos
        self.use_tmp_bos_pcent = use_tmp_bos_pcent
        self.vocab_sz = vocab_sz
        self.hidden_sz = hidden_sz
        self.device = device
        self.lm = None
        self.learnable_stft = learnable_stft
        if auxiliary_ctc_loss:
            self.ctc_head = nn.LSTM(
                out_sz_enc, vocab_sz, num_layers=1, batch_first=True
            )
        self.aux_ctc = auxiliary_ctc_loss
        self.loss = RNNTLoss(aux_ctc=auxiliary_ctc_loss) if loss else None
        self.ctx = Benchmark if benchmark else contextlib.suppress

    @staticmethod
    def from_config(conf, lang, lm=None, cls=None):
        if cls is None:
            cls = Transducer
        m = cls(
            conf["model"]["feature_sz"],
            conf["model"]["embed_sz"],
            conf["model"]["vocab_sz"],
            conf["model"]["hidden_sz"],
            conf["model"]["out_sz_enc"],
            conf["model"]["out_sz_pre"],
            conf["model"]["joint_sz"],
            lang,
            p_e=conf["model"]["encoder"]["dropout"],
            p_p=conf["model"]["predictor"]["dropout"],
            perf=False,
            raw_audio=False,
            use_tmp_bos=conf["model"]["use_tmp_bos"],
            use_tmp_bos_pcent=conf["model"]["use_tmp_bos_pcent"],
            preprocessor_kwargs=conf["model"]["preprocessor"],
            encoder_kwargs=conf["model"]["encoder"],
            predictor_kwargs=conf["model"]["predictor"],
            joint_kwargs=conf["model"]["joint"],
            enable_encoder=conf["model"].get("enable_encoder", True),
            enable_predictor=conf["model"].get("enable_predictor", True),
            enable_joint=conf["model"].get("enable_joint", True),
            learnable_stft=conf["model"]["learnable_stft"],
            device=conf["cuda"]["device"],
            loss=conf["model"].get("loss", False),
            **conf["model"].get("extra", {}),
        )  # .to(conf["cuda"]["device"])
        m.mp = conf.get("mp", False)
        return m

    def param_groups(self):
        l = [
            self.preprocessor.param_groups(),
            self.encoder.param_groups(),
        ]
        if self.enable_predictor:
            l.append(self.predictor.param_groups())
        if self.enable_joint:
            l.append(self.joint.param_groups())
        return l

    def quantization_fix(self):
        self.__class__ = QuantizedTransducer

    def grab_bos(self, y, yl, bs, device):
        if self.training and self.use_tmp_bos:
            r = random.random()
            thresh = 1.0 - self.use_tmp_bos_pcent
            cached_bos = self.bos_cache.get(bs)
            if torch.is_tensor(cached_bos) and r > thresh:
                # use end of last batch as bos
                bos = cached_bos
                return bos
            # store for next batch
            # is -1 acceptable here?
            try:
                q = torch.clamp(yl[:, None] - 1, min=0)
                self.bos_cache[bs] = y.gather(1, q).detach()
            except:
                pass
        # use regular bos
        bos = torch.zeros((bs, 1), device=device).long()
        bos = bos.fill_(self.bos)
        return bos

    def forward(
        self,
        tpl,
        softmax=True,
        return_logits=False,
        calc_loss=True,
        return_encoder=False,
    ):
        """
        (x, y)
        x: N tuples (audios of shape [N, n_chans, seq_len, H], x_lens)
        y: N tuples (y_padded, y_lens)
        """

        # unpack
        x, y, xl, yl = tpl

        # preprocess
        with self.ctx("preprocessor"):
            x, xl = self.preprocessor(x, xl)
            if torch.isinf(x).any() or torch.isnan(x).any():
                print("WARN: x is invalid after preprocessor...")

        # encoder
        with self.ctx("encoder"):
            encoder_out = self.encoder(x, lengths=xl)
            raw_encoder_out = encoder_out
            xl = xl * (encoder_out.size(1) / x.size(1))

        # bail if predictor is disabled
        if not self.enable_predictor:
            if return_encoder:
                return None, raw_encoder_out
            return None

        # N: batch size
        # T: n frames (time)
        # H: hidden features
        N, T, H_e = encoder_out.size()

        # predictor
        with self.ctx("predictor"):
            # concat first bos (yconcat is y shifted right by 1)
            bos = self.grab_bos(y, yl, bs=N, device=encoder_out.device)
            yconcat = torch.cat((bos, y), dim=1)
            # yl here because we want to omit the last label
            # in the resulting state (we had (yl + 1))
            predictor_out, _ = self.predictor(yconcat, lengths=yl + 1)
            N, U, H_p = predictor_out.size()

        # joint & project
        with self.ctx("joint"):
            joint_out = self.joint(
                predictor_out, encoder_out, softmax=softmax, log=True, expand=True
            )
            if self.loss is None or return_logits:
                return joint_out

        # aux ctc head
        ctc_out = None
        if self.aux_ctc:
            with self.ctx("aux_ctc"):
                ctc_out, _ = self.ctc_head(raw_encoder_out)

        # calc loss
        if calc_loss:
            with self.ctx("loss"):
                loss = self.loss(joint_out, y, xl, yl, ctc_out=ctc_out)
            return loss
        if return_encoder:
            return joint_out, raw_encoder_out
        return joint_out

    def transcribe_batch(
        self,
        tpl,
        max_iters=DEFAULT_MAX_ITERS,
        alpha=LM_ALPHA,
        temp_lm=LM_TEMP,
        temp_model=MODEL_TEMP,
        enc_rb_sz=0,
        enc_rb_trim=0,
        pre_rb_sz=0,
        pre_rb_trim=0,
    ):

        # unpack
        if isinstance(tpl, tuple) or isinstance(tpl, list):
            if len(tpl) == 4:
                x, _, xl, _ = tpl
            else:
                x, xl = tpl
        else:
            x, xl = tpl, None

        # preprocess
        x, xl = self.preprocessor(x, xl)

        # encoder
        x = x.reshape(x.size(0), x.size(1), -1)
        encoder_out = self.encoder(x, lengths=xl)

        # N: batch size
        # T: n frames (time)
        # U: n label tokens
        # V: vocab size
        # H: hidden features
        N, T, H = encoder_out.size()
        V = self.vocab_sz
        dtype, device = x.dtype, x.device

        # first
        pred_input = torch.zeros((N, 1), device=device).long()
        pred_input.fill_(self.bos)
        pred_output, pred_state = self.predictor(pred_input)

        # prepare vars
        outs = []
        predictions = []

        # lm
        fuser = LMFuserBatch(self.lm)

        # memoize joint
        mp, me = None, None
        rj = None

        def memo_joint(p, e):
            nonlocal mp, me, rj
            if ok([mp, me]) and ok(rj) and eq(mp, p) and eq(me, e):
                return rj, False
            rj = self.joint(p, e, temp=temp_model, softmax=True, log=False)
            mp, me, = (
                p,
                e,
            )
            return rj, True

        # memoize predictor
        mpo, mps = None, None
        rpo, rps = None, None

        def memo_predictor(po, ps):
            nonlocal mpo, mps, rpo, rps
            match_inp = eq(mpo, po)
            match_state = eq(mps, ps)
            if ok([mpo, mps, rpo, rps]) and match_inp and match_state:
                return rpo, rps, False
            rpo, rps = self.predictor(po, state=ps)
            mpo, mps = po, ps
            return rpo, rps, True

        # helper for predictor state
        def listify(l, cls=list):
            for i, sl in enumerate(l):
                l[i] = cls(sl)

        # iterate through time
        for t in range(T):

            # iterate through label
            for u in range(max_iters):

                # joint
                joint_out, _ = memo_joint(
                    pred_output.unsqueeze(1), encoder_out[:, t, None, None]
                )

                # decode one character
                #  (rnnt-only)
                prob, pred = joint_out.max(-1)

                # check for blanks
                #  (bail if all blank)
                pred_is_blank = pred == self.blank
                if pred_is_blank.all():
                    break

                # apply/fuse lm
                _, prob, pred = fuser.fuse(joint_out, prob, pred, alpha=alpha)

                # set as next input
                pred_input = pred[:, :, 0]

                # advance predictor (output & state)
                # issue: only advance predictor state
                #        when a non-blank has been decoded
                #        (use torch.where)...
                new_pred_output, new_pred_state, changed = memo_predictor(
                    pred_input, pred_state
                )

                if changed:
                    # update non-blanks for output & state
                    pred_output = torch.where(
                        pred_is_blank, pred_output, new_pred_output
                    )
                    qpred = pred[None, :, 0]
                    listify(pred_state, cls=list)
                    for i, (ps, nps) in enumerate(zip(pred_state, new_pred_state)):
                        for j, (psi, npsi) in enumerate(zip(ps, nps)):
                            pred_state[i][j] = torch.where(
                                qpred == self.blank, psi, npsi
                            )
                    listify(pred_state, cls=tuple)

                    # advance lm
                    fuser.advance(
                        pred[:, :, 0], keep_mask=pred == self.blank, temp=temp_lm
                    )

                # cap at xl
                if xl is not None:
                    pred[t >= xl] = self.blank

                # store prediction
                predictions.append(pred[:, 0, 0])

        # reset lm
        fuser.reset()

        # denumericalize
        strs = []
        if len(predictions) == 0:
            predictions = [torch.ones((x.size(0),), dtype=torch.int64) * self.blank]
        predictions = torch.stack(predictions, dim=1)
        for p in predictions:
            s = self.lang.denumericalize(p.cpu().numpy().tolist())
            strs.append(s)
        return strs

    def transcribe(self, *args, **kwargs):
        res, metrics, _ = self.decode_beam(*args, **kwargs)
        return res, metrics

    def decode_beam(
        self,
        x,
        max_iters=DEFAULT_MAX_ITERS,
        alpha=LM_ALPHA,
        temp_lm=LM_TEMP,
        temp_model=MODEL_TEMP,
        enc_rb_sz=0,
        enc_rb_trim=0,
        pre_rb_sz=0,
        pre_rb_trim=0,
        beam_search_opts={},
        **kwargs,
    ):
        "x must be of shape [C, T, H]"

        # keep stats
        metrics = {}
        extra = {
            "iters": [],
            "outs": [],
        }

        # move to correct device
        x = x.to(self.device)

        # set model to eval mode
        self.eval()

        # check shape of x
        if len(x.shape) == 2:
            # add channel dimension
            x = x[None]

        # reshape x to (1, C, T, H...)
        x = x[None]

        # preprocess
        with torch.no_grad():
            x = self.preprocessor(x)

        # encode full spectrogram (all timesteps)
        xl = torch.LongTensor([x.size(1)]).to(self.device)
        with torch.no_grad():
            encoder_out = self.encoder(x, lengths=xl)[0]

        # predictor: BOS goes first
        y_one_char = torch.LongTensor([[self.bos]]).to(encoder_out.device)
        h_t_pred, pred_state = self.predictor(y_one_char)

        # lm
        fuser = LMFuser(self.lm)

        # initiate beam search
        beam_search_opts = defaults(beam_search_opts, DEFAULT_BEAM_SEARCH_OPTS)
        impl = beam_search_opts["implementation"]
        blank, bos, lang = self.blank, self.bos, self.lang
        p, j = self.predictor, partial(
            self.joint, temp=temp_model, softmax=True, log=False
        )
        po, ps = h_t_pred, pred_state
        mi = max_iters
        dev = encoder_out.device
        beamer = Beamsearch(
            impl,
            beam_search_opts,
            blank,
            bos,
            lang,
            p,
            j,
            po,
            ps,
            mi,
            dev,
            lm=self.lm,
            lm_weight=alpha,
        )

        # iterate through all timesteps
        for t, h_t_enc in enumerate(encoder_out):

            # advance
            hyps = beamer(h_t_enc)

            # record how many iters we had
            extra["iters"].append(-1)

        # extract best hypothesis
        #  cut off BOS
        y_seq = hyps[0]

        # compute alignment score
        #  better if distributed along the sequence
        align = np.array(extra["iters"])
        _sum = align.sum()
        val, cnt = np.unique(align, return_counts=True)
        d = {v: c for v, c in zip(val, cnt)}
        _ones = d.get(1, 0)
        alignment_score = (_sum - _ones) / (_sum + 1e-4)
        metrics["alignment_score"] = alignment_score

        return self.lang.denumericalize(y_seq), metrics, extra


class Dummy(Module):
    def __init__(self, *args, **kwargs):
        self.mod = nn.Linear(1280, 20)

    @staticmethod
    def from_config(*args, **kwargs):
        return Dummy(root=True)

    def param_groups(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, tpl, **kwargs):
        x = tpl[0]
        x = self.mod(x[:, :, :, 0])
        # print("dummy forward x", x.shape, x.device)
        return (x ** 2).mean()


class NoisyStudent(Module):
    def __init__(self, teacher, student, **kwargs):
        self.teacher, self.student = teacher, student
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.loss = NoisyStudentLoss(**kwargs)

    def forward(self, tpl):

        # forward teacher
        self.teacher.eval()
        with torch.no_grad():
            t, t_e = self.teacher(
                tpl, softmax=False, calc_loss=False, return_encoder=True
            )

        # forward student
        s, s_e = self.student(tpl, softmax=False, calc_loss=False, return_encoder=True)

        # calculate loss
        return self.loss(*tpl, t, s, t_e, s_e)

    def param_groups(self):
        return self.student.param_groups()

    def transcribe(self, *args, **kwargs):
        return self.student.transcribe(*args, **kwargs)

    @property
    def encoder(self):
        return self.student.encoder


class QuantizedTransducer(Transducer):
    def eval(self):
        self.train(False)
        return self

    def train(self, _):
        try_eval(self)
        return self

    def to(self, _):
        return self


def masked_mean(t, mask, dim=1, thresh=0.5, random=False):
    if random:
        r = torch.zeros(t.size(dim), dtype=t.dtype, device=t.device).uniform_() > thresh
        r[0] = 1
        r = r.expand(mask.shape)
        mask = mask & r
    t = t.masked_fill(~mask[:, :, None], 0.0)
    return t.sum(dim=1) / mask.sum(dim=1)[..., None]


def crop(x, xl, size=2 * 16000, seq=1, random=False):
    N = x.size(0)

    # choose bounds
    until = torch.clamp(xl - size, min=1).long()
    new_xl = xl - until
    if random:
        start = (torch.rand((N,), device=x.device) * until).long()
    else:
        start = 0 if seq == 1 else size + 1
        start = (torch.ones((N,), device=x.device) * start).long()
    end = start + new_xl

    # new x
    shp = (N, size, *x.shape[2:])
    new_x = torch.zeros(*shp, device=x.device, dtype=x.dtype)
    for i in range(N):
        # set_trace()
        span = end[i] - start[i]
        new_x[i, :span] = x[i, start[i] : end[i]]

    return new_x, new_xl


class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size),
        )

    def forward(self, x):
        if len(x.shape) == 2:
            # (N x H)
            return self.net(x)
        else:
            for l in self.net:
                if isinstance(l, nn.BatchNorm1d):
                    x = x.permute(0, 2, 1)
                    x = l(x)
                    x = x.permute(0, 2, 1)
                else:
                    x = l(x)
            return x


def norm(x, dim=-1, eps=1e-5):
    "standardize x"
    return (x - x.mean(dim, keepdims=True)) / (x.std(dim, keepdims=True) + eps)


class Decoder(torch.nn.Module):
    def __init__(self, i, o, nl=2):
        super().__init__()
        h = o  # * 8
        start = [nn.Linear(i, h)]
        fat = [nn.Linear(h, h) for _ in range(nl)]
        end = [nn.Linear(h, o)]
        self.net = nn.Sequential(*(start + fat + end))

    def forward(self, x):
        x = self.net(x)
        x = x.contiguous()
        return x


class SemiSupervisedTransducer(Transducer):
    def __init__(
        self,
        *args,
        hidden_sz=1024,
        cache_sz=128,
        modalities=2,
        mode="simsiam",  # simcse or simsiam or contrastive or wav2vec2_contrastive
        random_masking=True,
        **kwargs,
    ):
        a, b = hidden_sz, hidden_sz
        if mode == "simsiam":
            self.projection = MLP(a, b, b)
            self.prediction = MLP(a, b, b)
        else:
            self.latents = nn.ModuleList(
                [nn.Linear(a, b, bias=False) for _ in range(modalities)]
            )
        if mode == "wav2vec2_contrastive":
            from transformers import (
                AutoTokenizer,
                AutoModel,
                Wav2Vec2FeatureExtractor,
                Wav2Vec2Processor,
            )

            name, cut_at = "facebook/wav2vec2-large-xlsr-53", kwargs.pop(
                "wav2vec2_cut", 15
            )
            model = AutoModel.from_pretrained(name)
            model.encoder.layers = model.encoder.layers[:cut_at]
            model = model.eval().cuda()
            self.wav2vec2 = [model]
            print("Contrastive Training with Wav2Vec2:")
            print(f" => Using '{name}' model")
            print(f" => Keeping {cut_at} attention layers")
        if mode == "dapc+mr+contrastive":
            self.w_c = 0.5
            self.w_pi = 0.1
            self.w_recon = 1.0
            self.decoder = Decoder(hidden_sz, 768)
            self.project = nn.Linear(hidden_sz, hidden_sz // 16, bias=False)
            print("dapc+mr model")
            print("  weights:", self.w_pi, self.w_recon)
        temps = 1 if modalities == 2 else 3
        self.temperature = nn.Parameter(torch.tensor([1.0 for _ in range(temps)]))
        self.cache_sz = cache_sz
        self.modalities = modalities
        self.mode = mode
        self.random_masking = random_masking
        self.cache = [[] for _ in range(modalities)]
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_config(conf, lang, lm=None, cls=None):
        cls = SemiSupervisedTransducer
        return Transducer.from_config(conf, lang, lm, cls)

    def param_groups(self):
        encoder_prepro_params = [
            *self.preprocessor.param_groups(),
            *self.encoder.param_groups(),
        ]
        if self.mode == "simsiam":
            return [
                encoder_prepro_params,
                [
                    self.temperature,
                    *list(self.projection.parameters()),
                    *list(self.prediction.parameters()),
                ],
            ]
        elif self.mode == "wav2vec2_contrastive":
            return [
                encoder_prepro_params,
                [
                    self.temperature,
                    *[l.weight for l in self.latents],
                ],
            ]
        elif self.mode == "dapc+mr+contrastive":
            return [
                encoder_prepro_params,
                [
                    self.temperature,
                    *[l.weight for l in self.latents],
                ],
                list(self.decoder.parameters()),
                list(self.project.parameters()),
            ]
        return [
            [
                self.temperature,
                *[l.weight for l in self.latents],
            ],
            self.preprocessor.param_groups(),
            self.encoder.param_groups(),
            self.predictor.param_groups(),
        ]

    def cache_and_extend(self, *mods, dim=0):
        new_mods = list(mods)
        if self.training:
            for i, mod in enumerate(mods):
                # if cache is non-empty
                # concat cache to new_mods
                if len(self.cache[0]) > 0:
                    new_mods[i] = torch.cat([*self.cache[i], mod], dim)

                # add to cache
                self.cache[i].insert(0, mod.detach())

                # cap cache size
                self.cache[i] = self.cache[i][: self.cache_sz]
            return new_mods
        return new_mods

    def augment(self, x, factor=0.1):
        if self.training:
            # add white noise
            x = x + (torch.randn_like(x) + x.mean()) * x.std() * factor
        return x

    def simsiam_loss(self, a, b, xl=None):
        b = b.detach()
        a, b = map(lambda t: F.normalize(t, p=2, dim=-1), (a, b))
        if xl is not None:
            # sequence-wise over all logits
            return -(a * b).sum(dim=(1, 2)) / xl
        else:
            # logits only
            return -(a * b).sum(dim=-1)

    def forward_simsiam(self, tpl, return_logits=False, sequence_wise=True):
        """
        SimSiam from https://arxiv.org/abs/2011.10566
        """

        # unpack
        x, y, xl, yl = tpl

        # grab augmentations
        x1, xl1 = self.preprocessor(self.augment(x), xl)
        x2, xl2 = self.preprocessor(self.augment(x), xl)

        # [N, T, H, W] -> [N, T, H]
        x1, x2 = map(lambda x: x.reshape(x.size(0), x.size(1), -1), (x1, x2))

        # encoder
        r1 = self.encoder(x1, lengths=xl1)
        r2 = self.encoder(x2, lengths=xl2)

        # N: batch size
        # T: n frames (time)
        # H: hidden features
        N, T, H = r1.size()

        # create masks
        r1mask = (
            torch.arange(T, dtype=xl1.dtype, device=xl1.device)[None, :] < xl1[:, None]
        )
        r2mask = (
            torch.arange(T, dtype=xl2.dtype, device=xl2.device)[None, :] < xl2[:, None]
        )

        if sequence_wise:
            # online (backbone + projection mlp)
            r1 = self.projection(r1)
            r2 = self.projection(r2)

            # target (prediction mlp)
            r1t = self.prediction(r1)
            r2t = self.prediction(r2)

            # make sure irrelevant values are zero in loss
            r1[~r1mask] = 0.0
            r2[~r2mask] = 0.0
            r1t[~r1mask] = 0.0
            r2t[~r2mask] = 0.0

            # pass lengths to loss
            xl_loss = xl1

        else:
            # reduce
            r1 = masked_mean(
                r1, r1mask, dim=1, random=self.random_masking and self.training
            )
            r2 = masked_mean(
                r2, r2mask, dim=1, random=self.random_masking and self.training
            )

            # online (backbone + projection mlp)
            r1 = self.projection(r1)
            r2 = self.projection(r2)

            # target (prediction mlp)
            r1t = self.prediction(r1)
            r2t = self.prediction(r2)

            # no lengths needed
            xl_loss = None

        # loss
        if return_logits:
            return r1, r2
        loss = (
            self.simsiam_loss(r1, r2t, xl=xl_loss) / 2.0
            + self.simsiam_loss(r2, r1t, xl=xl_loss) / 2.0
        )
        return loss

    def forward_contrastive(self, tpl, return_logits=False):
        """
        A contrastive objective following CLIP.
        https://arxiv.org/abs/2103.00020
        """

        # unpack
        x, y, xl, yl = tpl
        N = x.size(0)

        # preprocess signal
        x_raw, xl_raw = x, xl
        x, xl = self.preprocessor(self.augment(x), xl)

        # [N, T, H, W] -> [N, T, H]
        x = x.reshape(x.size(0), x.size(1), -1)

        # encoder A (r1)
        r1 = self.encoder(x, lengths=xl)
        T = r1.size(1)
        r1mask = (
            torch.arange(T, dtype=xl.dtype, device=xl.device)[None, :] < xl[:, None]
        )

        # encoder B (r2)
        if self.mode == "wav2vec2_contrastive":
            # zero mean + unit variance
            x, xl = x_raw, xl_raw
            x = (x - x.mean(0)) / (x.std(0) + 1e-5)

            # attention mask / mask for later
            attn_mask = (
                torch.arange(x.size(1), dtype=xl.dtype, device=xl.device)[None, :]
                < xl[:, None]
            )

            # assemble
            inp = {"input_values": x[:, :, 0, 0], "attention_mask": attn_mask}

            # extract
            with torch.no_grad():
                r2 = self.wav2vec2[0](**inp).last_hidden_state
            r2mask = (
                torch.arange(r2.size(1), dtype=xl.dtype, device=xl.device)[None, :] > -1
            )
        else:
            # predictor
            # concat first bos (yconcat is y shifted right by 1)
            bos = self.grab_bos(y, yl, bs=N, device=x.device)
            yconcat = torch.cat((bos, y), dim=1)
            # yl here because we want to omit the last label
            # in the resulting state (we had (yl + 1))
            r2, _ = self.predictor(yconcat, lengths=yl + 1)
            U = r2.size(1)
            r2mask = (
                torch.arange(U, dtype=yl.dtype, device=yl.device)[None, :] < yl[:, None]
            )

        ###
        # following lucidrains CLIP model:
        #  https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/dalle_pytorch.py
        ###

        # reduce
        r1 = masked_mean(
            r1, r1mask, dim=1, random=self.random_masking and self.training
        )
        r2 = masked_mean(
            r2, r2mask, dim=1, random=self.random_masking and self.training
        )

        # project
        r1 = self.latents[0](r1)
        r2 = self.latents[1](r2)

        # normalize
        r1, r2 = map(lambda t: F.normalize(t, p=2, dim=-1), (r1, r2))

        # cache for later
        # and extend batch
        r1, r2 = self.cache_and_extend(r1, r2)
        N = r1.size(0)

        # loss
        temp = self.temperature.exp()
        sim = einsum("i d, j d -> i j", r1, r2) * temp
        labels = torch.arange(N, device=x.device)
        if return_logits:
            return r1, r2, sim, labels

        # calculate losses
        l1 = F.cross_entropy(sim, labels)
        l2 = F.cross_entropy(sim.T, labels)
        return {"loss": (l1 + l2) / 2.0}

    def forward_dapc_mr_contrastive(self, tpl, return_logits=False):
        # unpack
        x, y, xl, yl = tpl
        N = x.size(0)

        # create two differently augmented instances of x
        xraw1, xl1, xmask1, _ = self.preprocessor.augment(x, xl)
        xraw2, xl2, xmask2, _ = self.preprocessor.augment(x, xl)
        x1 = xraw1 * xmask1
        x2 = xraw2 * xmask2
        xinv1 = norm(xraw1 * (1.0 - xmask1))
        xinv2 = norm(xraw2 * (1.0 - xmask2))

        # feed both through encoder
        enc1 = self.encoder(x1, lengths=xl1)
        enc2 = self.encoder(x2, lengths=xl2)
        # r1, r2 = enc1, enc2
        # T = r1.size(1)
        # r1mask = (
        #     torch.arange(T, dtype=xl.dtype, device=xl.device)[None, :] < xl[:, None]
        # )
        # r2mask = r1mask

        # # reduce
        # r1 = masked_mean(
        #     r1, r1mask, dim=1, random=self.random_masking and self.training
        # )
        # r2 = masked_mean(
        #     r2, r2mask, dim=1, random=self.random_masking and self.training
        # )

        # # project
        # r1 = self.latents[0](r1)
        # r2 = self.latents[1](r2)

        # # normalize
        # r1, r2 = map(lambda t: F.normalize(t, p=2, dim=-1), (r1, r2))

        # # cache for later
        # # and extend batch
        # r1, r2 = self.cache_and_extend(r1, r2)
        # N = r1.size(0)

        # # contrastive loss
        # temp = self.temperature.exp()
        # sim = einsum("i d, j d -> i j", r1, r2) * temp
        # labels = torch.arange(N, device=x.device)
        # l1 = F.cross_entropy(sim, labels)
        # l2 = F.cross_entropy(sim.T, labels)
        # loss_contrastive = ((l1 + l2) / 2.0).mean()
        loss_contrastive = torch.Tensor([0.0]).to(x.device)

        # pi (dapc) loss
        try:
            encp1 = self.project(enc1)
            encp2 = self.project(enc2)
            loss_pi1, _ = pi_loss(encp1, xl1)
            loss_pi2, _ = pi_loss(encp2, xl2)
            loss_pi = (loss_pi1 + loss_pi2) / 2.0
        except:
            loss_pi = torch.Tensor([0.0]).to(x.device)

        # reconstruction loss
        dec1 = norm(self.decoder(enc1))
        dec2 = norm(self.decoder(enc2))
        lr1 = (((dec1 - xinv1) ** 2) * (1 - xmask1)).mean((1, 2))
        lr2 = (((dec2 - xinv2) ** 2) * (1 - xmask2)).mean((1, 2))
        loss_recon = ((lr1 + lr2) / 2.0).mean()

        # full loss
        a = self.w_c * loss_contrastive
        b = self.w_pi * loss_pi
        c = self.w_recon * loss_recon
        loss = a + b + c
        d = {
            "loss": loss,
            "loss_contrastive": a,
            "loss_pi": b,
            "loss_recon": c,
        }
        if return_logits:
            return d, (xraw1, x1, enc1, dec1, xinv1, xmask1)
        return d

    def forward(self, tpl, return_logits=False):
        """
        (x, y)
        x: N tuples (audios of shape [N, n_chans, seq_len, H], x_lens)
        y: N tuples (y_padded, y_lens)
        """

        if self.mode == "simsiam":
            return self.forward_simsiam(tpl, return_logits=return_logits)

        if self.mode == "dapc+mr+contrastive":
            return self.forward_dapc_mr_contrastive(tpl, return_logits=return_logits)

        if "contrastive" in self.mode:
            return self.forward_contrastive(tpl, return_logits=return_logits)

        raise Exception(f"No such mode {self.mode}")


class ConformerEncoder(Module):
    def __init__(
        self,
        feature_sz,
        hidden_sz,
        out_sz,
        dropout=0.01,
        num_layers=2,
        trace=True,
        device="cuda:0",
        use_tmp_state_pcent=0.9,
        **kwargs,
    ):
        self.num_layers = num_layers
        self.input_norm = nn.LayerNorm(feature_sz)
        hidden_conformer = 384  # hidden_sz // 2
        self.pre_proj = nn.Linear(feature_sz, hidden_conformer)
        from libreasr.lib.layers.conformer import ConformerBlock

        self.conformer = nn.Sequential(
            *[
                ConformerBlock(
                    dim=hidden_conformer,
                    dim_head=64,
                    heads=8,
                    ff_mult=4,
                    conv_expansion_factor=2,
                    conv_kernel_size=31,
                    attn_dropout=0.0,
                    ff_dropout=0.0,
                    conv_dropout=0.0,
                )
                for _ in range(num_layers)
            ]
        )
        self.post_proj = nn.Linear(hidden_conformer, out_sz)

    def param_groups(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x, state=None, lengths=None, return_state=False):
        x = x.reshape((x.size(0), x.size(1), -1))
        x = self.input_norm(x)
        x = self.pre_proj(x)
        x = torch.utils.checkpoint.checkpoint_sequential(self.conformer, 8, x)
        x = self.post_proj(x)
        if return_state:
            return x, state
        return x
