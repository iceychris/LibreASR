import operator
import time
import random
from queue import PriorityQueue

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from fastai2.vision.models.xresnet import xresnet18
from fastai2.layers import Debugger, ResBlock
from fastai2.torch_core import Module
from fastai2.learner import CancelBatchException

from IPython.core.debugger import set_trace

from libreasr.lib.utils import *
from libreasr.lib.layers import *
from libreasr.lib.lm import LMFuser


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
        self.layer_norm = nn.LayerNorm(hidden_sz)

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
        x = self.layer_norm(x)

        # down projection
        x = self.down(x)
        x = self.activation(x)

        # up projection
        x = self.up(x)

        # residual connection
        return x + inp


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
        layer_norm=False,
        rnn_type="LSTM",
        use_tmp_state_pcent=0.9,
        **kwargs,
    ):
        self.num_layers = num_layers
        self.input_norm = nn.LayerNorm(feature_sz)
        self.rnn_stack = CustomCPURNN(
            feature_sz,
            hidden_sz,
            num_layers,
            rnn_type=rnn_type,
            reduction_indices=[],  # 1
            reduction_factors=[],  # 2
            layer_norm=layer_norm,
            rezero=False,
            utsp=use_tmp_state_pcent,
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
        layer_norm=False,
        rnn_type="NBRC",
        use_tmp_state_pcent=0.9,
    ):
        self.vocab_sz = vocab_sz
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_sz, embed_sz, padding_idx=blank)
        if not embed_sz == hidden_sz:
            self.ffn = nn.Linear(embed_sz, hidden_sz)
        else:
            self.ffn = nn.Sequential()
        self.rnn_stack = CustomCPURNN(
            hidden_sz,
            hidden_sz,
            num_layers,
            rnn_type=rnn_type,
            layer_norm=layer_norm,
            utsp=use_tmp_state_pcent,
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


class Joint(Module):
    def __init__(self, out_sz, joint_sz, vocab_sz, joint_method):
        self.joint_method = joint_method
        if joint_method == "add":
            input_sz = out_sz
        elif joint_method == "concat":
            input_sz = 2 * out_sz
        else:
            raise Exception("No such joint_method")
        self.joint = nn.Sequential(
            nn.Linear(input_sz, joint_sz),
            nn.Tanh(),
            nn.Linear(joint_sz, vocab_sz),
        )

    def param_groups(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, h_pred, h_enc, softmax=True):
        if self.joint_method == "add":
            x = h_pred + h_enc
        elif self.joint_method == "concat":
            x = torch.cat((h_pred, h_enc), dim=-1)
        else:
            raise Exception("No such joint_method")
        # x = torch.utils.checkpoint.checkpoint_sequential(self.joint, 2, x)
        x = self.joint(x)
        if softmax:
            x = F.log_softmax(x, dim=-1)
        return x


def get_model(conf, *args, **kwargs):
    if conf["training"]["noisystudent"]:
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
        **kwargs,
    ):
        self.encoder = Encoder(
            feature_sz, hidden_sz=hidden_sz, out_sz=out_sz, **encoder_kwargs,
        )
        self.predictor = Predictor(
            vocab_sz,
            embed_sz=embed_sz,
            hidden_sz=hidden_sz,
            out_sz=out_sz,
            **predictor_kwargs,
        )
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
        self.lm = None

    @staticmethod
    def from_config(conf, lang, lm=None, cls=None):
        if cls is None: cls = Transducer
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
            bs=conf["bs"],
            raw_audio=False,
            use_tmp_bos=conf["model"]["use_tmp_bos"],
            use_tmp_bos_pcent=conf["model"]["use_tmp_bos_pcent"],
            encoder_kwargs=conf["model"]["encoder"],
            predictor_kwargs=conf["model"]["predictor"],
        ).to(conf["cuda"]["device"])
        m.mp = conf["mp"]
        return m

    def param_groups(self):
        return [
            self.encoder.param_groups(),
            self.predictor.param_groups(),
            self.joint.param_groups(),
        ]

    def convert_to_cpu(self):
        self.encoder.rnn_stack = self.encoder.rnn_stack.convert_to_cpu()
        self.predictor.rnn_stack = self.predictor.rnn_stack.convert_to_cpu()
        return self

    def convert_to_gpu(self):
        self.encoder.rnn_stack = self.encoder.rnn_stack.convert_to_gpu()
        self.predictor.rnn_stack = self.predictor.rnn_stack.convert_to_gpu()
        return self

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
        joint_out = self.joint(predictor_out, encoder_out, softmax=softmax)

        return joint_out

    def transcribe_batch(self, tpl, max_iters=3, lm=False):

        # unpack
        if (isinstance(tpl, tuple) or isinstance(tpl, list)) and len(tpl) == 4:
            x, _, _, _ = tpl
        else:
            x = tpl
        if self.mp:
            x = x.half()

        # encoder
        x = x.reshape(x.size(0), x.size(1), -1)
        encoder_out = self.encoder(x)

        # N: batch size
        # T: n frames (time)
        # U: n label tokens
        # V: vocab size
        # H: hidden features
        N, T, H = encoder_out.size()
        V = self.vocab_sz
        dtype, device = x.dtype, x.device

        # frist
        pred_input = torch.zeros((N, 1), device=device).long()
        pred_input.fill_(self.bos)
        pred_output, pred_state = self.predictor(pred_input)

        # prepare vars
        outs = []
        predictions = []
        strs = []

        # check if sth is None
        def ok(a):
            if isinstance(a, list): return all([ok(x) for x in a])
            return a is not None

        # check if two things are the same
        def eq(a, b):
            if (a is None or b is None): return False
            if isinstance(a, list):
                return all([(x == y).all() for x, y in zip(a, b)])
            return (a == b).all()

        # memoize joint
        mp, me = None, None
        rj = None
        def memo_joint(p, e):
            nonlocal mp, me, rj
            if ok([mp, me]) and ok(rj) and eq(mp, p) and eq(me, e):
                return rj, False
            rj = self.joint(p, e)
            mp, me, = p, e
            return rj, True

        # memoize predictor
        mpo, mps = None, None
        rpo, rps = None, None
        def memo_predictor(po, ps):
            nonlocal mpo, mps, rpo, rps
            if ok([mpo, mps, rpo, rps]) and eq(mpo, po) and eq(mps, ps):
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
                joint_out, _ = memo_joint(pred_output.unsqueeze(1), encoder_out[:, t, None, None])

                # decode one character
                prob, pred = joint_out.max(-1)

                # set as next input
                pred_input = pred[:, :, 0]

                # bail if all blank
                if (pred == self.blank).all():
                    break

                # advance predictor (output & state)
                # issue: only advance predictor state
                #        when a non-blank has been decoded
                #        (use gather and where)...
                new_pred_output, new_pred_state, changed = memo_predictor(
                    pred_input, pred_state
                )

                if changed:
                    # select non-blanks for output & state
                    pred_output = torch.where(pred == self.blank, pred_output, new_pred_output)
                    qpred = pred[None, :, 0]
                    listify(pred_state, cls=list)
                    for i, (ps, nps) in enumerate(zip(pred_state, new_pred_state)):
                        for j, (psi, npsi) in enumerate(zip(ps, nps)):
                            pred_state[i][j] = torch.where(qpred == self.blank, psi, npsi)
                    listify(pred_state, cls=tuple)

                # store prediction
                predictions.append(pred[:, 0, 0])

        # denumericalize
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

    def decode_greedy(self, x, max_iters=3, alpha=0.005, theta=1.0):
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

        # encode full spectrogram (all timesteps)
        encoder_out = self.encoder(x)[0]

        # predictor: BOS goes first
        y_one_char = torch.LongTensor([[self.bos]]).to(encoder_out.device)
        h_t_pred, pred_state = self.predictor(y_one_char)

        # lm
        fuser = LMFuser(self.lm)

        # iterate through all timesteps
        y_seq, log_p = [], 0.0
        for h_t_enc in encoder_out:

            iters = 0
            while iters < max_iters:
                iters += 1

                # h_t_enc is of shape [H]
                # go through the joint network
                _h_t_pred = h_t_pred[None]
                _h_t_enc = h_t_enc[None, None, None]
                joint_out = self.joint(_h_t_pred, _h_t_enc)

                # decode one character
                extra["outs"].append(joint_out.clone())
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
                    joint_out, prob, pred = fuser.fuse(joint_out, prob, pred)

                    # print(iters)
                    y_seq.append(pred)
                    y_one_char[0][0] = pred

                    # advance predictor
                    h_t_pred, pred_state = self.predictor(y_one_char, state=pred_state)

                    # advance lm
                    fuser.advance(y_one_char)

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
        self, stream, denumericalizer, max_iters=10, alpha=0.3, theta=1.0
    ):
        """
        stream is expected to yield chunks of shape (NCHANS, CHUNKSIZE)
        """
        # put model into evaluation mode
        self.eval()

        # state to hold while transcribing
        encoder_state = None
        predictor_state = None

        # current token
        y_one_char = torch.LongTensor([[self.bos]])
        h_t_pred = None

        # sequence of the hole stream
        y = []

        # lm
        fuser = LMFuser(self.lm)

        def reset_encoder():
            nonlocal encoder_state
            encoder_state = None

        def reset_predictor():
            nonlocal y_one_char, h_t_pred, predictor_state
            # initialize predictor
            #  blank goes first
            y_one_char = torch.LongTensor([[self.bos]])
            h_t_pred, predictor_state = self.predictor(y_one_char)

        def reset_lm():
            fuser.reset()

        def reset():
            reset_encoder()
            reset_predictor()
            reset_lm()

        # reset at start
        reset()

        # iterate through time
        # T > 1 is possible
        blanks = 0
        nonblanks = 0
        for chunk in stream:

            # in case we get a None, just continue
            if chunk is None:
                continue

            # -> [1, T, H, W]
            chunk = chunk[None]

            # forward pass encoder
            if encoder_state is None:
                encoder_out, encoder_state = self.encoder(chunk, return_state=True)
            else:
                encoder_out, encoder_state = self.encoder(
                    chunk, state=encoder_state, return_state=True
                )
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
                    joint_out = self.joint(_h_t_pred, _h_t_enc)

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
                        joint_out, prob, pred = fuser.fuse(joint_out, prob, pred)

                        y_seq.append(pred)
                        y_one_char[0][0] = pred

                        # advance predictor
                        h_t_pred, predictor_state = self.predictor(
                            y_one_char, state=predictor_state
                        )

                        # advance lm
                        fuser.advance(y_one_char)

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


class CTCModel(Module):
    def __init__(self):
        layer = nn.TransformerEncoderLayer(128, 8)
        self.encoder = nn.TransformerEncoder(layer, 8)
        self.linear = nn.Linear(128, 2048)

    def convert_to_gpu(self):
        pass

    def param_groups(self):
        return [p for p in self.parameters() if p.requires_grad]

    @staticmethod
    def from_config(conf, lang):
        return CTCModel()

    def forward(self, tpl):
        x, y, xl, yl = tpl
        x = x.view(x.size(1), x.size(0), -1).contiguous()
        x = self.encoder(x)
        x = self.linear(x)
        x = F.log_softmax(x, -1)
        return x
