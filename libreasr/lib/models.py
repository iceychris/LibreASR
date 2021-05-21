import contextlib
import operator
import time
import random
from queue import PriorityQueue
from functools import partial, lru_cache
import itertools
import math
from dataclasses import dataclass, field
from typing import List

import torch
from torch import nn, einsum
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
    DEFAULT_BEAM_SEARCH_OPTS,
)
from libreasr.lib.inference.beamsearch import (
    BeamStateBuilder,
    Beamer,
    print_beam_results,
    start_rnnt_beam_search,
)


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

    def __init__(self, time_mask_n=2, time_mask_sz=4, freq_mask_n=4, freq_mask_sz=2):
        self.time_mask_n = time_mask_n
        self.time_mask_sz = time_mask_sz
        self.freq_mask_n = freq_mask_n
        self.freq_mask_sz = freq_mask_sz
        self.start = None
        self.val = None

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

    def forward(self, x):
        if not self.training:
            return x
        x = self.mask_time(x)
        x = self.mask_freq(x)
        return x


class Preprocessor(Module):
    def __init__(
        self,
        sr=16000,
        n_mels=80,
        n_fft=1024,
        downsample=8,
        stack=8,
        trainable=False,
        **kwargs,
    ):
        from nnAudio.Spectrogram import MelSpectrogram

        self.spec = MelSpectrogram(
            sr=sr,
            trainable_mel=trainable,
            trainable_STFT=trainable,
            n_fft=n_fft,
            n_mels=n_mels,
            win_length=int(0.025 * sr),
            hop_length=int(0.01 * sr),
        )
        self.specaugment = SpecAugment(**kwargs)
        self.sr = sr
        self.n_mels = n_mels
        self.downsample = downsample
        self.stack = stack

    def param_groups(self):
        return [p for p in self.parameters() if p.requires_grad]

    def stack_downsample(self, x):
        x = x.unfold(-2, self.stack, self.downsample).contiguous()
        x = x.view(x.size(0), x.size(1), -1).contiguous()
        return x

    def forward(self, x, xl=None, inference=False):
        x = self.spec(x[..., :, 0, 0])
        x = self.specaugment(x)
        x = x.permute(0, 2, 1).contiguous()
        if not inference:
            x = self.stack_downsample(x)
        if xl is not None:
            fac = self.sr // 100
            xl = torch.clamp(xl // (fac * self.downsample), min=1, max=x.size(1))
            return x, xl
        return x


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

    def param_groups(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x, state=None, lengths=None):
        x = self.embed(x)
        x = self.ffn(x)
        x, state = self.rnn_stack(x, state=state, lengths=lengths)
        x = self.drop(x)
        x = self.linear(x)
        return x, state


class InnerJoint(Module):
    def __init__(self, i, h, v):
        self.net = nn.Sequential(
            nn.Linear(i, h),
            nn.Tanh(),
            nn.Linear(h, v),
        )

    def param_groups(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x):
        return self.net(x)


class Joint(Module):
    def __init__(self, out_sz, joint_sz, vocab_sz, joint_method, reversible=False):
        self.joint_method = joint_method
        self.reversible = reversible
        if joint_method == "add":
            input_sz = out_sz
        elif joint_method == "concat" or self.joint_method == "comb":
            input_sz = 2 * out_sz
        else:
            raise Exception("No such joint_method")
        if reversible:
            import memcnn

            inv_joint = memcnn.AdditiveCoupling(
                Fm=InnerJoint(input_sz // 2, joint_sz // 2, vocab_sz // 2),
                Gm=InnerJoint(input_sz // 2, joint_sz // 2, vocab_sz // 2),
                split_dim=3,
            )
            assert memcnn.is_invertible_module(
                inv_joint, test_input_shape=(4, 3, 2, input_sz)
            )
            inv_joint = memcnn.InvertibleModuleWrapper(
                fn=inv_joint, keep_input=False, keep_input_inverse=False
            )
            assert memcnn.is_invertible_module(
                inv_joint, test_input_shape=(4, 3, 2, input_sz)
            )
            self.joint = inv_joint
        else:
            # less memory usage
            # self.joint = nn.Sequential(
            #     nn.Tanh(),
            #     nn.Linear(input_sz, vocab_sz),
            # )
            # custom LibreASR
            self.joint = nn.Sequential(
                nn.Linear(input_sz, joint_sz),
                nn.Tanh(),
                nn.Linear(joint_sz, vocab_sz),
            )

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
    ):
        # https://arxiv.org/pdf/2011.01576.pdf
        if normalize_grad:
            h_pred.register_hook(lambda grad: grad / h_enc.size(1))
            h_enc.register_hook(lambda grad: grad / h_pred.size(1))
        if self.joint_method == "add":
            x = h_pred + h_enc
        elif self.joint_method == "mul":
            x = h_pred * h_enc
        elif self.joint_method == "comb":
            x = torch.cat((h_pred + h_enc, h_pred * h_enc), dim=-1)
        elif self.joint_method == "concat":
            x = torch.cat((h_pred, h_enc), dim=-1)
        else:
            raise Exception("No such joint_method")
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


def get_model(conf, *args, **kwargs):
    if conf.get("training", {}).get("noisystudent", False):
        teacher = eval(conf["model"]["name"]).from_config(conf, *args, **kwargs)
        ns = NoisyStudent(teacher, Transducer.from_config(conf, *args, **kwargs))
        return ns
    return eval(conf["model"]["name"]).from_config(conf, *args, **kwargs)


class RNNTLoss(Module):
    def __init__(self, reduction=1.0, zero_nan=False):
        from warp_rnnt import rnnt_loss

        self.loss = partial(rnnt_loss, average_frames=False, fastemit_lambda=0.004)
        self.reduction = reduction
        self.zero_nan = zero_nan

    def forward(self, x, y, xl, yl):
        # trim lens
        xl = torch.clamp(xl // self.reduction, min=1, max=x.size(1))

        # fix types
        y = y.type(torch.int32)
        xl = xl.type(torch.int32)
        yl = yl.type(torch.int32)

        # avoid NaN
        if self.zero_nan:
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        return self.loss(x, y, xl, yl)


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
        out_sz,
        joint_sz,
        lang,
        l_e=6,
        l_p=2,
        p_j=0.0,
        blank=0,
        joint_method="concat",
        perf=False,
        act=F.relu,
        use_tmp_bos=True,
        use_tmp_bos_pcent=0.99,
        preprocessor_kwargs={},
        encoder_kwargs={},
        predictor_kwargs={},
        enable_encoder=True,
        enable_predictor=True,
        enable_joint=True,
        joint_reversible=False,
        learnable_stft=False,
        device="cpu",
        loss=False,
        benchmark=False,
        **kwargs,
    ):
        self.preprocessor = (
            Preprocessor(**preprocessor_kwargs) if learnable_stft else Noop()
        )
        if enable_encoder:
            self.encoder = eval(encoder_kwargs["name"])(
                feature_sz,
                hidden_sz=hidden_sz,
                out_sz=out_sz,
                **encoder_kwargs,
            )
        if enable_predictor:
            self.predictor = eval(predictor_kwargs["name"])(
                vocab_sz,
                embed_sz=embed_sz,
                hidden_sz=hidden_sz,
                out_sz=out_sz,
                **predictor_kwargs,
            )
        if enable_joint:
            self.joint = Joint(
                out_sz, joint_sz, vocab_sz, joint_method, reversible=joint_reversible
            )
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
        self.loss = RNNTLoss() if loss else None
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
            conf["model"]["out_sz"],
            conf["model"]["joint_sz"],
            lang,
            p_e=conf["model"]["encoder"]["dropout"],
            p_p=conf["model"]["predictor"]["dropout"],
            p_j=conf["model"]["joint"]["dropout"],
            joint_method=conf["model"]["joint"]["method"],
            perf=False,
            raw_audio=False,
            use_tmp_bos=conf["model"]["use_tmp_bos"],
            use_tmp_bos_pcent=conf["model"]["use_tmp_bos_pcent"],
            preprocessor_kwargs=conf["model"]["preprocessor"],
            encoder_kwargs=conf["model"]["encoder"],
            predictor_kwargs=conf["model"]["predictor"],
            enable_encoder=conf["model"].get("enable_encoder", True),
            enable_predictor=conf["model"].get("enable_predictor", True),
            enable_joint=conf["model"].get("enable_joint", True),
            joint_reversible=conf["model"]["joint"]["reversible"],
            learnable_stft=conf["model"]["learnable_stft"],
            device=conf["cuda"]["device"],
            loss=conf["model"].get("loss", False),
            **conf["model"].get("extra", {}),
        )  # .to(conf["cuda"]["device"])
        m.mp = conf.get("mp", False)
        return m

    def param_groups(self):
        return [
            self.preprocessor.param_groups(),
            self.encoder.param_groups(),
            self.predictor.param_groups(),
            self.joint.param_groups(),
        ]

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

    def forward(self, tpl, softmax=True, return_logits=False):
        """
        (x, y)
        x: N tuples (audios of shape [N, n_chans, seq_len, H], x_lens)
        y: N tuples (y_padded, y_lens)
        """

        # unpack
        x, y, xl, yl = tpl
        # print("x", x.shape)

        # preprocess
        with self.ctx("preprocessor"):
            x, xl = self.preprocessor(x, xl)
            if torch.isinf(x).any() or torch.isnan(x).any():
                print("WARN: x is invalid after preprocessor...")

        # encoder
        with self.ctx("encoder"):
            x = x.reshape(x.size(0), x.size(1), -1)
            encoder_out = self.encoder(x, lengths=xl)

        # N: batch size
        # T: n frames (time)
        # H: hidden features
        N, T, H = encoder_out.size()

        # predictor
        with self.ctx("predictor"):
            # concat first bos (yconcat is y shifted right by 1)
            bos = self.grab_bos(y, yl, bs=N, device=encoder_out.device)
            yconcat = torch.cat((bos, y), dim=1)
            # yl here because we want to omit the last label
            # in the resulting state (we had (yl + 1))
            predictor_out, _ = self.predictor(yconcat, lengths=yl + 1)
            U = predictor_out.size(1)

        # expand:
        # we need to pass [N, T, U, H] to the joint network
        # NOTE: we might want to not include padding here?
        with self.ctx("expand"):
            M = max(T, U)
            sz = (N, T, U, H)
            encoder_out = encoder_out.unsqueeze(2).expand(sz).contiguous()
            predictor_out = predictor_out.unsqueeze(1).expand(sz).contiguous()
            # print(encoder_out.shape, predictor_out.shape)

        # joint & project
        with self.ctx("joint"):
            joint_out = self.joint(
                predictor_out, encoder_out, softmax=softmax, log=True
            )
            if self.loss is None or return_logits:
                return joint_out

        # calc loss
        with self.ctx("loss"):
            loss = self.loss(joint_out, y, xl, yl)
        return loss

    def transcribe_batch(
        self,
        tpl,
        max_iters=3,
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
        max_iters=2,
        alpha=LM_ALPHA,
        temp_lm=LM_TEMP,
        temp_model=MODEL_TEMP,
        enc_rb_sz=0,
        enc_rb_trim=0,
        pre_rb_sz=0,
        pre_rb_trim=0,
        beam_search_opts={},
    ):
        "x must be of shape [C, T, H]"

        # keep stats
        metrics = {}
        extra = {
            "iters": [],
            "outs": [],
        }

        # set model to eval mode
        self.eval()

        # check shape of x
        if len(x.shape) == 2:
            # add channel dimension
            x = x[None]

        # reshape x to (1, C, T, H...)
        x = x[None]

        # preprocess
        x = self.preprocessor(x)

        # encode full spectrogram (all timesteps)
        encoder_out = self.encoder(x)[0]

        # predictor: BOS goes first
        y_one_char = torch.LongTensor([[self.bos]]).to(encoder_out.device)
        h_t_pred, pred_state = self.predictor(y_one_char)

        # lm
        fuser = LMFuser(self.lm)

        # initiate beam search
        beam_search_opts = defaults(beam_search_opts, DEFAULT_BEAM_SEARCH_OPTS)
        blank, bos, lang = self.blank, self.bos, self.lang
        p, j = self.predictor, partial(
            self.joint, temp=temp_model, softmax=True, log=False
        )
        po, ps = h_t_pred, pred_state
        mi = max_iters
        beamer = start_rnnt_beam_search(
            beam_search_opts, blank, bos, lang, p, j, po, ps, mi
        )

        # iterate through all timesteps
        for t, h_t_enc in enumerate(encoder_out):

            # advance
            beamer(h_t_enc)

            # record how many iters we had
            extra["iters"].append(-1)

        # finalize beam search
        # print_beam_results(beamer.all, denumericalize_fn=self.lang.denumericalize)

        # extract best hypothesis
        #  cut off BOS
        y_seq = beamer.best.tokens[1:]

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

    def transcribe_stream(
        self,
        stream,
        denumericalizer,
        max_iters=2,
        alpha=LM_ALPHA,
        temp_lm=LM_TEMP,
        temp_model=MODEL_TEMP,
        enc_rb_sz=0,
        enc_rb_trim=0,
        pre_rb_sz=0,
        pre_rb_trim=0,
        beam_search_opts={},
    ):
        """
        stream is expected to yield chunks of shape (NCHANS, CHUNKSIZE)
        """
        # put model into evaluation mode
        self.eval()

        # state to hold while transcribing
        encoder_state = None
        predictor_state = None

        # variables
        dev = torch.device(self.device)
        y_first = torch.LongTensor([[self.bos]]).to(dev)
        y_one_char = torch.LongTensor([[self.bos]]).to(dev)
        h_t_pred = None

        # sequence of the whole stream
        y = []

        # lm
        fuser = LMFuser(self.lm)

        # functions
        etrb = TensorRingBuffer(enc_rb_sz, (1, 0, 1280), dim=1, device=dev)

        def enc(x, state, return_state):
            if etrb.append(x):
                x = etrb.get()
                r, s = self.encoder(x, return_state=return_state)
                if enc_rb_trim > 0:
                    etrb.trim_to(enc_rb_trim)
                r = r[:, -1:]
                return r, s
            return self.encoder(x, state=state, return_state=return_state)

        ptrb = TensorRingBuffer(pre_rb_sz, (1, 0), dim=1, device=dev, dtype=torch.int64)

        def pre(x, state):
            if ptrb.append(x):
                x = ptrb.get()
                x = torch.cat([y_first, x], dim=1)
                r, s = self.predictor(x)
                r = r[:, -1:]
                if pre_rb_trim > 0:
                    ptrb.trim_to(pre_rb_trim)
                return r, s
            return self.predictor(x, state=state)

        def reset_encoder():
            nonlocal encoder_state
            encoder_state = None

        def reset_predictor():
            nonlocal y_one_char, h_t_pred, predictor_state
            # initialize predictor
            #  blank goes first
            y_one_char = torch.LongTensor([[self.bos]]).to(dev)
            h_t_pred, predictor_state = self.predictor(y_one_char, None)

        def reset_lm():
            fuser.reset()

        def reset():
            # only reset lm periodically
            #  if using ring buffers
            # reset_encoder()
            # reset_predictor()
            reset_lm()
            pass

        # reset at start
        # reset()
        reset_predictor()

        last_x = None
        assert isinstance(self.preprocessor, Preprocessor)
        bs, n_mels, stack = 1, self.preprocessor.n_mels, self.preprocessor.stack
        last_remainder = torch.zeros(bs, 0, n_mels, device=dev)
        split = 0

        def fix_tiling(x):
            """
            Funky way to make sure:
            * spectrograms are properly tiled
              and correct at the boundary (kinda working)
            * time dimension is divisible by `stack`
              for `stack_downsample`
            """
            nonlocal last_x, last_remainder
            if last_x is None:
                # save
                last_x = x
                return None, True

            # 1. merge
            l = [last_remainder, last_x[:, :-1], x[:, :-1]]
            merged = torch.cat(l, dim=1)

            # 2. cut
            T = merged.size(1)
            nm = nearest_multiple(T, stack)
            out = merged[:, :nm]
            last_remainder = merged[:, nm:]
            last_x = None
            return out, False

        # initiate beam search
        beam_search_opts = defaults(beam_search_opts, DEFAULT_BEAM_SEARCH_OPTS)
        blank, bos, lang = self.blank, self.bos, self.lang
        p, j = pre, partial(self.joint, temp=temp_model, softmax=True, log=False)
        po, ps = h_t_pred, predictor_state
        mi = max_iters
        beamer = start_rnnt_beam_search(
            beam_search_opts, blank, bos, lang, p, j, po, ps, mi
        )

        # iterate through time
        # T > 1 is possible
        blanks = 0
        nonblanks = 0
        for chunk in stream:

            # in case we get a None, just continue
            if chunk is None:
                continue

            # to correct device
            chunk = chunk.to(dev)

            # preprocessor
            chunk = self.preprocessor(chunk, inference=True)

            # adjust for correct stft window
            #  and stack_downsample size
            chunk, skip = fix_tiling(chunk)
            if skip:
                continue

            # stack
            chunk = self.preprocessor.stack_downsample(chunk)

            # forward pass encoder
            encoder_out, encoder_state = enc(chunk, encoder_state, return_state=True)
            h_t_enc = encoder_out[0]

            # loop over encoder states (t)
            y_seq = []
            for i in range(h_t_enc.size(-2)):
                h_enc = h_t_enc[..., i, :]

                # perform beam search step
                best = beamer(h_enc)
                tok = best.tokens[1:]
                tok = list(filter(lambda x: x != self.blank, tok))
                if len(tok) > 0:
                    yield tok, reset


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
    def __init__(self, teacher, student):
        self.teacher, self.student = [teacher], student
        self.teacher[0].eval()
        for param in self.teacher[0].parameters():
            param.requires_grad = False

    def forward(self, tpl):

        # forward teacher
        with torch.no_grad():
            t = self.teacher[0](tpl, softmax=False)

        # apply noise
        if self.training:
            x = tpl[0]
            dtype, dev, std = x.dtype, x.device, x.std()
            tpl[0] += torch.randn(tpl[0].shape, dtype=dtype, device=dev) * std * 0.1

        # forward student
        s = self.student(tpl, softmax=False)

        return (t, s)

    def param_groups(self):
        return [
            self.student.encoder.param_groups(),
            self.student.predictor.param_groups(),
            self.student.joint.param_groups(),
        ]

    def transcribe(self, *args, **kwargs):
        return self.student.transcribe(*args, **kwargs)


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


class ContrastiveTransducer(Transducer):
    def __init__(
        self,
        *args,
        hidden_sz=1024,
        cache_sz=128,
        modalities=2,
        mode="simsiam",  # simcse or clip or simsiam
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
        cls = ContrastiveTransducer
        return Transducer.from_config(conf, lang, lm, cls)

    def param_groups(self):
        if self.mode == "simsiam":
            return [
                [
                    *self.preprocessor.param_groups(),
                    *self.encoder.param_groups(),
                ],
                [
                    self.temperature,
                    *list(self.projection.parameters()),
                    *list(self.prediction.parameters()),
                ],
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

        # preprocess signal
        x, xl = self.preprocessor(self.augment(x), xl)

        # [N, T, H, W] -> [N, T, H]
        x = x.reshape(x.size(0), x.size(1), -1)

        # encoder
        r1 = self.encoder(x, lengths=xl)

        # N: batch size
        # T: n frames (time)
        # H: hidden features
        N, T, H = r1.size()

        # predictor
        # concat first bos (yconcat is y shifted right by 1)
        bos = self.grab_bos(y, yl, bs=N, device=x.device)
        yconcat = torch.cat((bos, y), dim=1)
        # yl here because we want to omit the last label
        # in the resulting state (we had (yl + 1))
        r2, _ = self.predictor(yconcat, lengths=yl + 1)
        U = r2.size(1)

        ###
        # following lucidrains CLIP model:
        #  https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/dalle_pytorch.py
        ###

        # create masks
        r1mask = (
            torch.arange(T, dtype=xl.dtype, device=xl.device)[None, :] < xl[:, None]
        )
        r2mask = (
            torch.arange(U, dtype=yl.dtype, device=yl.device)[None, :] < yl[:, None]
        )

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
        return (l1 + l2) / 2.0

    def forward(self, tpl, return_logits=False):
        """
        (x, y)
        x: N tuples (audios of shape [N, n_chans, seq_len, H], x_lens)
        y: N tuples (y_padded, y_lens)
        """

        # special path for simsiam
        if self.mode == "simsiam":
            return self.forward_simsiam(tpl, return_logits=return_logits)

        # special path for contrastive
        if self.mode == "contrastive":
            return self.forward_contrastive(tpl, return_logits=return_logits)

        # no such method
        raise Exception(f"No such mode {self.mode}")


class LinearTransformerEncoder(Module):
    def __init__(
        self,
        feature_sz,
        hidden_sz,
        out_sz,
        dropout=0.01,
        num_layers=6,
        trace=True,
        device="cuda:0",
        use_tmp_state_pcent=0.9,
        **kwargs,
    ):
        self.num_layers = num_layers
        self.input_norm = nn.LayerNorm(feature_sz)
        self.pre_proj = nn.Linear(feature_sz, hidden_sz)
        from fast_transformers.builders import (
            TransformerEncoderBuilder,
            RecurrentEncoderBuilder,
        )
        from fast_transformers.utils import make_mirror
        from fast_transformers.masking import TriangularCausalMask

        n_heads = 16
        params = dict(
            attention_type="causal-linear",
            n_layers=num_layers,
            n_heads=n_heads,
            feed_forward_dimensions=hidden_sz,
            query_dimensions=hidden_sz // n_heads,
            value_dimensions=hidden_sz // n_heads,
        )
        self.transformer = TransformerEncoderBuilder.from_dictionary(params).get()
        self.recurrent_transformer = RecurrentEncoderBuilder.from_dictionary(
            params
        ).get()
        make_mirror(self.transformer, self.recurrent_transformer)
        self.tcm = TriangularCausalMask
        self.post_proj = nn.Linear(hidden_sz, out_sz)

    def param_groups(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x, state=None, lengths=None, return_state=False):
        x = x.reshape((x.size(0), x.size(1), -1))
        x = self.input_norm(x)
        x = self.pre_proj(x)
        if state is None:
            x = self.transformer(x, attn_mask=self.tcm(x.size(0), device=x.device))
        else:
            raise NotImplementedError("recurrent transformer not implemented")
        x = self.post_proj(x)
        if return_state:
            return x, state
        return x


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


class TransformerPredictor(Module):
    def __init__(
        self,
        vocab_sz,
        embed_sz,
        hidden_sz,
        out_sz,
        dropout=0.01,
        num_layers=2,
        blank=0,
        use_tmp_state_pcent=0.9,
        **kwargs,
    ):
        self.vocab_sz = vocab_sz
        self.num_layers = num_layers
        from x_transformers import TransformerWrapper, Encoder as XEncoder

        self.transformer = model = TransformerWrapper(
            num_tokens=vocab_sz,
            max_seq_len=1024,
            attn_layers=XEncoder(dim=hidden_sz, depth=num_layers, heads=8),  # 12
        )
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_sz * 2, hidden_sz)

    def param_groups(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x, state=None, lengths=None):
        x = self.transformer(x)
        x = self.drop(x)
        x = self.linear(x)
        return x, state
