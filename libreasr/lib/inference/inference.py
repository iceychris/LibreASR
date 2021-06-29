from functools import partial

import torch

from libreasr.lib.models import Transducer
from libreasr.lib.defaults import (
    LM_ALPHA,
    LM_TEMP,
    MODEL_TEMP,
    DEFAULT_MAX_ITERS,
    DEFAULT_BEAM_SEARCH_OPTS,
)
from libreasr.lib.lm import LMFuser
from libreasr.lib.utils import defaults, nearest_multiple, TensorRingBuffer
from libreasr.lib.inference.beamsearch import start_rnnt_beam_search
from libreasr.lib.inference.events import *


def infer_stream(
    self: Transducer,
    generator,
    max_iters=DEFAULT_MAX_ITERS,
    alpha=LM_ALPHA,
    temp_lm=LM_TEMP,
    temp_model=MODEL_TEMP,
    enc_rb_sz=0,
    enc_rb_trim=0,
    pre_rb_sz=0,
    pre_rb_trim=0,
    beam_search_opts={},
    device=None,
):
    """
    generator is expected to yield chunks of shape (NCHANS, CHUNKSIZE)
    """
    # put model into evaluation mode
    self.eval()

    # state to hold while transcribing
    encoder_state = None
    predictor_state = None

    # variables
    dev = device or torch.device(self.device)
    y_first = torch.LongTensor([[self.bos]]).to(dev)
    y_one_char = torch.LongTensor([[self.bos]]).to(dev)
    h_t_pred = None

    # sequence of the whole stream
    y = []

    # lm
    fuser = LMFuser(self.lm)

    # functions
    etrb = TensorRingBuffer(enc_rb_sz, (1, 0, self.feature_sz), dim=1, device=dev)

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
    # TODO: pass these as arguments
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

    # signal start
    yield StartEvent(beamer)

    # iterate through time
    # T > 1 is possible
    blanks = 0
    nonblanks = 0
    output_found = False
    for chunk in generator:

        # in case we get a None, just continue
        if chunk is None:
            yield IdleEvent()
            continue

        # to correct device
        chunk = chunk.to(dev)

        # preprocessor
        chunk = self.preprocessor(chunk, inference=True)

        # adjust for correct stft window
        #  and stack_downsample size
        chunk, skip = fix_tiling(chunk)
        if skip:
            yield IdleEvent()
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
            hyps = beamer(h_enc)
            yield HypothesisEvent(hyps)

    # in case we haven't found any hypothesis
    yield StopEvent()