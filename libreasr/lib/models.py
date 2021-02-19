import operator
import time
import random
from queue import PriorityQueue

import torch
from torch import nn, einsum
import torch.nn.functional as F

import numpy as np

from fastai2.vision.models.xresnet import xresnet18
from fastai2.layers import Debugger, ResBlock
from fastai2.torch_core import Module
from fastai2.learner import CancelBatchException

from IPython.core.debugger import set_trace

from libreasr.lib.utils import *
from libreasr.lib.layers import StackedRNN
from libreasr.lib.layers.conformer import ConformerBlock
from libreasr.lib.lm import LMFuser, LMFuserBatch
from libreasr.lib.defaults import LM_ALPHA, LM_TEMP, MODEL_TEMP


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


class Preprocessor(Module):
    def __init__(self):
        from nnAudio.Spectrogram import MelSpectrogram

        self.spec = MelSpectrogram(
            sr=16000,
            trainable_mel=True,
            trainable_STFT=True,
            n_fft=2048,
            n_mels=128,
            win_length=400,
            hop_length=160,
        )

    def param_groups(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x, xl=None):
        x = self.spec(x[..., :, 0, 0]).permute(0, 2, 1).contiguous()
        x = x.unfold(-2, 10, 8).contiguous()
        x = x.view(x.size(0), x.size(1), -1).contiguous()
        if xl is not None:
            xl = torch.clamp(xl // (160 * 8), min=0, max=x.size(1))
            return x, xl
        return x


class Encoder(Module):
    def __init__(
        self,
        feature_sz,
        hidden_sz,
        out_sz,
        dropout=0.01,
        num_layers=2,
        trace=True,
        device="cuda:0",
        rnn_type="LSTM",
        norm="bn",
        use_tmp_state_pcent=0.9,
        **kwargs,
    ):
        self.num_layers = num_layers
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
        )
        self.drop = nn.Dropout(dropout)
        if not hidden_sz == out_sz:
            self.linear = nn.Linear(hidden_sz, out_sz)
        else:
            self.linear = nn.Sequential()

    def param_groups(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x, state=None, lengths=None, return_state=False):
        x = x.reshape((x.size(0), x.size(1), -1))
        x = self.input_norm(x)
        x, state = self.rnn_stack(x, state=state, lengths=lengths)
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
        elif joint_method == "concat":
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
        encoder_kwargs={},
        predictor_kwargs={},
        joint=True,
        learnable_stft=False,
        device="cpu",
        **kwargs,
    ):
        self.preprocessor = Preprocessor() if learnable_stft else Noop()
        self.encoder = eval(encoder_kwargs["name"])(
            feature_sz,
            hidden_sz=hidden_sz,
            out_sz=out_sz,
            **encoder_kwargs,
        )
        self.predictor = eval(predictor_kwargs["name"])(
            vocab_sz,
            embed_sz=embed_sz,
            hidden_sz=hidden_sz,
            out_sz=out_sz,
            **predictor_kwargs,
        )
        if joint:
            self.joint = Joint(out_sz, joint_sz, vocab_sz, joint_method)
        self.lang = lang
        self.blank = blank
        # TODO: dont hardcode
        self.bos = 2
        self.mp = False
        self.bos_cache = {}
        self.use_tmp_bos = use_tmp_bos
        self.use_tmp_bos_pcent = use_tmp_bos_pcent
        self.vocab_sz = vocab_sz
        self.hidden_sz = hidden_sz
        self.device = device
        self.lm = None
        self.learnable_stft = learnable_stft

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
            encoder_kwargs=conf["model"]["encoder"],
            predictor_kwargs=conf["model"]["predictor"],
            joint=conf["model"]["joint"]["enable"],
            learnable_stft=conf["model"]["learnable_stft"],
            device=conf["cuda"]["device"],
            **conf["model"].get("extra", {}),
        ).to(conf["cuda"]["device"])
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

    def forward(self, tpl, softmax=True):
        """
        (x, y)
        x: N tuples (audios of shape [N, n_chans, seq_len, H], x_lens)
        y: N tuples (y_padded, y_lens)
        """

        # unpack
        x, y, xl, yl = tpl
        if self.mp:
            x = x.half()

        # preprocess
        x, xl = self.preprocessor(x, xl)

        # encoder
        x = x.reshape(x.size(0), x.size(1), -1)
        encoder_out = self.encoder(x, lengths=xl)

        # N: batch size
        # T: n frames (time)
        # H: hidden features
        N, T, H = encoder_out.size()

        # predictor
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
        M = max(T, U)
        sz = (N, T, U, H)
        encoder_out = encoder_out.unsqueeze(2).expand(sz).contiguous()
        predictor_out = predictor_out.unsqueeze(1).expand(sz).contiguous()
        # print(encoder_out.shape, predictor_out.shape)

        # joint & project
        joint_out = self.joint(predictor_out, encoder_out, softmax=softmax, log=True)

        return joint_out

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
        if self.mp:
            x = x.half()

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

        # check if sth is None
        def ok(a):
            if isinstance(a, list):
                return all([ok(x) for x in a])
            return a is not None

        # check if two things are the same
        def eq(a, b):
            if a is None or b is None:
                return False
            if isinstance(a, (list, tuple)):
                return all([eq(x, y) for x, y in zip(a, b)])
            return (a == b).all().item()

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

    def decode(self, *args, **kwargs):
        res, log_p, _ = self.decode_greedy(*args, **kwargs)
        return res, log_p

    def transcribe(self, *args, **kwargs):
        res, _, metrics, _ = self.decode_greedy(*args, **kwargs)
        return res, metrics

    def decode_greedy(
        self,
        x,
        max_iters=3,
        alpha=LM_ALPHA,
        temp_lm=LM_TEMP,
        temp_model=MODEL_TEMP,
        enc_rb_sz=0,
        enc_rb_trim=0,
        pre_rb_sz=0,
        pre_rb_trim=0,
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

        # iterate through all timesteps
        y_seq, log_p = [], 0.0
        for t, h_t_enc in enumerate(encoder_out):

            iters = 0
            while iters < max_iters:
                iters += 1

                # h_t_enc is of shape [H]
                # go through the joint network
                _h_t_pred = h_t_pred[None]
                _h_t_enc = h_t_enc[None, None, None]
                joint_out = self.joint(
                    _h_t_pred, _h_t_enc, temp=temp_model, softmax=True, log=False
                )

                # decode one character
                # extra["outs"].append(joint_out.clone())
                prob, pred = joint_out.max(-1)
                pred = int(pred)
                log_p += float(prob)

                # if blank,     advance encoder state
                # if not blank, add to the decoded sequence so far
                #               and advance predictor state
                if pred == self.blank:
                    break
                else:
                    # fuse with lm
                    _, prob, pred = fuser.fuse(joint_out, prob, pred, alpha=alpha)

                    # print(iters)
                    y_seq.append(pred)
                    y_one_char[0][0] = pred

                    # advance predictor
                    h_t_pred, pred_state = self.predictor(y_one_char, state=pred_state)

                    # advance lm
                    fuser.advance(y_one_char, temp=temp_lm)

            # record how many iters we had
            extra["iters"].append(iters)

        # compute alignment score
        #  better if distributed along the sequence
        align = np.array(extra["iters"])
        _sum = align.sum()
        val, cnt = np.unique(align, return_counts=True)
        d = {v: c for v, c in zip(val, cnt)}
        _ones = d.get(1, 0)
        alignment_score = (_sum - _ones) / (_sum + 1e-4)
        metrics["alignment_score"] = alignment_score

        return self.lang.denumericalize(y_seq), -log_p, metrics, extra

    def transcribe_stream(
        self,
        stream,
        denumericalizer,
        max_iters=10,
        alpha=LM_ALPHA,
        temp_lm=LM_TEMP,
        temp_model=MODEL_TEMP,
        enc_rb_sz=0,
        enc_rb_trim=0,
        pre_rb_sz=0,
        pre_rb_trim=0,
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

        # iterate through time
        # T > 1 is possible
        blanks = 0
        nonblanks = 0
        for chunk in stream:

            # in case we get a None, just continue
            if chunk is None:
                continue

            # -> [1, T, H, W]
            chunk = chunk.to(dev)[None, ..., 0]

            # forward pass encoder
            encoder_out, encoder_state = enc(chunk, encoder_state, return_state=True)
            h_t_enc = encoder_out[0]

            # loop over encoder states (t)
            y_seq = []
            for i in range(h_t_enc.size(-2)):
                h_enc = h_t_enc[..., i, :]

                iters = 0
                while iters < max_iters:
                    iters += 1

                    # h_enc is of shape [H]
                    # go through the joint network
                    _h_t_pred = h_t_pred[None]
                    _h_t_enc = h_enc[None, None, None]
                    # print(_h_t_pred.shape)
                    # print(_h_t_enc.shape)
                    joint_out = self.joint(
                        _h_t_pred, _h_t_enc, temp=temp_model, softmax=True, log=False
                    )

                    # decode one character
                    prob, pred = joint_out.max(-1)
                    pred = int(pred)

                    # if blank,     advance encoder state
                    # if not blank, add to the decoded sequence so far
                    #               and advance predictor state
                    if pred == self.blank:
                        blanks += 1
                        break
                    else:
                        # fuse with lm
                        joint_out, prob, pred = fuser.fuse(
                            joint_out, prob, pred, alpha=alpha
                        )

                        y_seq.append(pred)
                        y_one_char[0][0] = pred

                        # advance predictor
                        h_t_pred, predictor_state = pre(y_one_char, predictor_state)

                        # advance lm
                        fuser.advance(y_one_char, temp=temp_lm)

                        nonblanks += 1

            # add to y
            y = y + y_seq
            yield y, denumericalizer(y_seq), reset


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


def masked_mean(t, mask, dim=1, thresh=0.05, random=False):
    if random:
        r = torch.zeros(t.size(dim), dtype=t.dtype, device=t.device).uniform_() > thresh
        r[0] = 1
        r = r.expand(mask.shape)
        mask = mask & r
    t = t.masked_fill(~mask[:, :, None], 0.0)
    return t.sum(dim=1) / mask.sum(dim=1)[..., None]


def crop(x, xl, size=30, seq=1, random=False):
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


class ContrastiveTransducer(Transducer):
    def __init__(
        self,
        *args,
        hidden_sz=1024,
        cache_sz=128,
        modalities=2,
        **kwargs,
    ):
        a, b = hidden_sz, hidden_sz
        self.latents = nn.ModuleList([nn.Linear(a, b) for _ in range(modalities)])
        temps = 1 if modalities == 2 else 3
        self.temperature = nn.Parameter(torch.tensor([1.0 for _ in range(temps)]))
        self.cache_sz = cache_sz
        self.modalities = modalities
        self.cache = [[] for _ in range(modalities)]
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_config(conf, lang, lm=None, cls=None):
        cls = ContrastiveTransducer
        return Transducer.from_config(conf, lang, lm, cls)

    def param_groups(self):
        return [
            [
                self.temperature,
                *[l.weight for l in self.latents],
                *self.preprocessor.param_groups(),
            ],
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

    def forward(self, tpl, return_logits=False):
        """
        (x, y)
        x: N tuples (audios of shape [N, n_chans, seq_len, H], x_lens)
        y: N tuples (y_padded, y_lens)
        """

        # unpack
        x, y, xl, yl = tpl
        if self.mp:
            x = x.half()

        # preprocess
        x, xl = self.preprocessor(x, xl)

        # [N, T, H, W] -> [N, T, H]
        x = x.reshape(x.size(0), x.size(1), -1)

        if self.modalities == 2:

            # encoder
            encoder_out = self.encoder(x, lengths=xl)

            # N: batch size
            # T: n frames (time)
            # H: hidden features
            N, T, H = encoder_out.size()

            # predictor
            # concat first bos (yconcat is y shifted right by 1)
            bos = self.grab_bos(y, yl, bs=N, device=encoder_out.device)
            yconcat = torch.cat((bos, y), dim=1)
            # yl here because we want to omit the last label
            # in the resulting state (we had (yl + 1))
            predictor_out, _ = self.predictor(yconcat, lengths=yl + 1)
            U = predictor_out.size(1)

            ###
            # following lucidrains CLIP model:
            #  https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/dalle_pytorch.py
            ###

            # create masks
            emask = (
                torch.arange(T, dtype=xl.dtype, device=xl.device)[None, :] < xl[:, None]
            )
            pmask = (
                torch.arange(U, dtype=yl.dtype, device=yl.device)[None, :] < yl[:, None]
            )

            # reduce
            e = masked_mean(encoder_out, emask, dim=1, random=self.training)
            p = masked_mean(predictor_out, pmask, dim=1, random=self.training)

            # project
            e = self.latents[0](e)
            p = self.latents[1](p)

            # normalize
            e, p = map(lambda t: F.normalize(t, p=2, dim=-1), (e, p))

            # cache for later
            # and extend batch
            e, p = self.cache_and_extend(e, p)
            N = e.size(0)

            # loss
            temp = self.temperature.exp()
            sim = einsum("i d, j d -> i j", e, p) * temp
            labels = torch.arange(N, device=x.device)
            if return_logits:
                return e, p, sim, labels

            # calculate losses
            l1 = F.cross_entropy(sim, labels)
            l2 = F.cross_entropy(sim.T, labels)
            return (l1 + l2) / 2.0

        # N modalities
        else:

            # compute two crops
            x1, xl1 = crop(x, xl, seq=1, random=self.training)
            x2, xl2 = crop(x, xl, seq=2, random=self.training)

            # encode
            a = self.encoder(x1, lengths=xl1)
            b = self.encoder(x2, lengths=xl2)

            # N: batch size
            # T: n frames (time)
            # H: hidden features
            N, T, H = a.size()

            # predictor
            # concat first bos (yconcat is y shifted right by 1)
            bos = self.grab_bos(y, yl, bs=N, device=a.device)
            yconcat = torch.cat((bos, y), dim=1)
            # yl here because we want to omit the last label
            # in the resulting state (we had (yl + 1))
            c, _ = self.predictor(yconcat, lengths=yl + 1)
            U = c.size(1)

            ###
            # following lucidrains CLIP model:
            #  https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/dalle_pytorch.py
            ###

            # create masks
            amask = (
                torch.arange(T, dtype=xl.dtype, device=xl.device)[None, :]
                < xl1[:, None]
            )
            bmask = (
                torch.arange(T, dtype=xl.dtype, device=xl.device)[None, :]
                < xl2[:, None]
            )
            cmask = (
                torch.arange(U, dtype=yl.dtype, device=yl.device)[None, :] < yl[:, None]
            )

            # reduce
            a = masked_mean(a, amask, dim=1, random=self.training)
            b = masked_mean(b, bmask, dim=1, random=self.training)
            c = masked_mean(c, cmask, dim=1, random=self.training)

            # project
            a = self.latents[0](a)
            b = self.latents[1](b)
            c = self.latents[2](c)

            # normalize
            a, b, c = map(lambda t: F.normalize(t, p=2, dim=-1), (a, b, c))

            # cache for later
            # and extend batch
            a, b, c = self.cache_and_extend(a, b, c)
            N = a.size(0)

            # similarity scores
            temp1 = self.temperature[0].exp()
            temp2 = self.temperature[1].exp()
            temp3 = self.temperature[2].exp()
            sim1 = einsum("i d, j d -> i j", a, b) * temp1
            sim2 = einsum("i d, j d -> i j", b, c) * temp2
            sim3 = einsum("i d, j d -> i j", c, a) * temp3

            # create labels
            labels = torch.arange(N, device=x.device).long()
            if return_logits:
                return (a, b, c), (sim1, sim2, sim3), labels

            # calculate losses
            loss = torch.zeros((1,), device=x.device)
            count = 0.0
            for sim in (sim1, sim2, sim3):
                loss += F.cross_entropy(sim, labels)
                loss += F.cross_entropy(sim.T, labels)
                count += 2
            return loss / count


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
