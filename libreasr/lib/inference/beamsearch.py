import time
from functools import partial, lru_cache
import itertools
import math
from dataclasses import dataclass, field
from typing import List, Callable

import torch
from torch import nn

import numpy as np

from libreasr.lib.memo import mk_memo


# beam search length normalization
#  alpha parameter.
#  See https://www.youtube.com/watch?v=ZGUZwk7xIwk
BEAM_SEARCH_SCORE_ALPHA = 1.0  # 0.0
BEAM_SEARCH_INITIAL_MULTIPLIER = 1.0


def BeamStateBuilder(predictor_fn, score_cache_sz):
    @dataclass
    class BeamState:
        enc: torch.Tensor = None
        pred: torch.Tensor = None
        tokens: List[int] = field(default_factory=list)
        probs: List[float] = field(default_factory=list)
        multiplier: float = BEAM_SEARCH_INITIAL_MULTIPLIER

        def __eq__(self, other):
            return self.tokens == other.tokens

        def __lt__(self, other):
            return self.score < other.score

        def __hash__(self):
            return (
                hash(tuple(self.tokens))
                ^ hash(tuple(self.probs))
                ^ hash(self.multiplier)
            )

        @property
        @lru_cache(maxsize=score_cache_sz)
        def score(self):
            # probs are expected to be elem [0.0, 1.0[
            tokens, probs = self.tokens, self.probs
            alpha = BEAM_SEARCH_SCORE_ALPHA
            probs = np.array([*probs, self.multiplier])
            factor = 1 / len(tokens) ** alpha
            score = factor * np.sum(np.log(probs))
            return score

        def expand(self, token, prob):
            toks = [*self.tokens, token]
            probs = [*self.probs, prob]
            return BeamState(
                self.enc,
                self.pred,
                toks,
                probs,
                self.multiplier,
            )

        def commit(self):
            return BeamState(
                self.enc,
                predictor_fn(self.tokens),
                self.tokens,
                self.probs,
                self.multiplier,
            )

        def with_enc(self, enc):
            return BeamState(
                enc,
                self.pred,
                self.tokens,
                self.probs,
                self.multiplier,
            )

        def with_probs(self, probs):
            return BeamState(
                self.enc,
                self.pred,
                self.tokens,
                probs,
                self.multiplier,
            )

        def with_multiplier(self, multiplier: float):
            return BeamState(
                self.enc,
                self.pred,
                self.tokens,
                self.probs,
                multiplier,
            )

        def str(self, raw=False):
            toks = self.tokens[1:]
            return str(toks)

    return BeamState


def generate_hyps(state, joint: Callable, t: int, k=4, blank=False, blank_token_idx=0):
    """
    a function to get next tokens
    starting from a given BeamState `state`
    """

    # fix encoder shape
    assert state.enc is not None
    _h_t_enc = state.enc[None, None, None]

    # fix predictor shape
    assert state.pred is not None
    _h_t_pred = state.pred[None]

    # we need the already tokens for hashing
    tokens = tuple(state.tokens)

    # go through the joint network
    #  and grab topk
    joint_out, _ = joint(key=(t, tokens), inp=(_h_t_pred, _h_t_enc))
    j = joint_out[0, 0, 0, ...]
    sz = j.size(-1)
    assert blank_token_idx == 0
    if blank:
        topk = j[:1].topk(1)
        indices = topk.indices.tolist()
    else:
        topk = j[1:].topk(min(k, sz - 1))
        indices = (topk.indices + 1).tolist()
    probs = topk.values.tolist()
    states = []
    for i, p in zip(indices, probs):
        states.append(state.expand(i, p))
    return states


class Beamer(nn.Module):
    def __init__(
        self,
        initial,
        joint_fn,
        beam_width=1,
        topk_next=1,
        blank_token_idx=0,
        max_iters=3,
        debug=False,
    ):
        super().__init__()
        self.initial = initial
        self.joint_fn = joint_fn
        self.beam = [initial]
        self.beam_width = beam_width
        self.topk_next = topk_next
        self.blank_token_idx = blank_token_idx
        self.max_iters = max_iters
        self.debug = debug
        self.t = 0
        if debug:
            print(f"[beamsearch] bw={beam_width}, topk={topk_next}, mi={max_iters}")

    def forward(self, enc):
        bw = self.beam_width
        candidates = self.beam
        beam = []

        # merge similar hypothesises
        key_lambda = lambda s: tuple([x for x in s.tokens if x != self.blank_token_idx])
        grouped = itertools.groupby(candidates, key=key_lambda)
        candidates = []
        for k, g in grouped:
            g = list(g)

            # no merges needed
            if len(g) == 1:
                candidates.append(g[0])
                continue

            # option 1
            #  just keep the best hypothesis :-)
            # candidates.append(max(g))

            # option 2
            #  naively increase the multiplier
            s = g[0]  # max(g)
            keep = s.with_multiplier(s.multiplier * len(g))
            candidates.append(keep)

            # debug
            # for s in g:
            #     print(s.score, s.multiplier, s.tokens)
            # print("kept", keep.score, keep.multiplier, keep.tokens)
            # print()

        # expand
        l = 0
        i = 0
        while len(candidates) > 0:
            # find best candidate
            scores = np.array([c.score for c in candidates])
            idx = scores.argmax()
            best = candidates[idx]
            candidates.remove(best)

            # bail if bw reached
            better_than_best = [b for b in beam if b > best]
            if len(better_than_best) >= bw:
                break

            # add encoder state and
            #  commit best as we need new outputs
            best = best.with_enc(enc).commit()
            if self.debug:
                print(f"extending: {best.str()}")

            # append blank and add to beam
            t = self.t
            beam.extend(generate_hyps(best, self.joint_fn, t, blank=True))

            # append non-blank and add to candidates
            if i <= self.max_iters:
                candidates.extend(
                    generate_hyps(best, self.joint_fn, t, k=self.topk_next, blank=False)
                )

            # increment u counter
            i += 1

        # order and select k best
        ordered = sorted(beam, reverse=True)
        self.beam = ordered[:bw]

        # increment frame counter
        self.t += 1

        # return current best
        return self.best

    @property
    def all(self):
        return self.beam

    @property
    def best(self):
        """
        Return the best hypothesis
        """
        return max(self.beam)


def print_beam_results(results, denumericalize_fn, top=4, blank=0):
    results = list(
        map(lambda s: (denumericalize_fn(s.tokens[1:]), s.tokens, s.score), results)
    )[:top]
    for q, r in enumerate(results):
        print(f"#{q}:")
        for part in r:
            print(" -", part)

        # split bpe
        print(" -", " ".join([denumericalize_fn(x) for x in r[1]]))

        # accumulated tokens
        blank_here = False
        blank_num = 0
        seq = []
        for tok in r[1]:
            if tok == blank:
                blank_here = True
                blank_num += 1
            else:
                if blank_here or tok == r[1][-1]:
                    seq.append(f"0#{blank_num}")
                seq.append(tok)
                blank_here = False
                blank_num = 0
        print(" -", seq)


def start_rnnt_beam_search(
    beam_search_opts,
    blank,
    bos,
    lang,
    predictor,
    joint,
    predictor_output,
    predictor_state,
    max_iters,
):
    # beam search params
    bsopts = beam_search_opts
    beam_width = bsopts.get("beam_width")
    topk_next = bsopts.get("topk_next")
    predictor_cache_sz = bsopts.get("predictor_cache_sz")
    joint_cache_sz = bsopts.get("joint_cache_sz")
    score_cache_sz = bsopts.get("score_cache_sz")
    debug = bsopts.get("debug")

    def _memo_predictor(prev_tokens, cache):
        # transforms to pred_tokens
        pred_tokens = tuple(filter(lambda x: x != blank, prev_tokens))

        # retrieve from cache
        key = tuple(pred_tokens)
        if key in cache:
            return cache[key][0]

        # backtrack until known
        remaining = tuple()
        state = list(cache[(bos,)])
        for i in range(len(pred_tokens)):
            toks = pred_tokens[:i]
            if toks in cache:
                remaining = pred_tokens[i:]
                state = list(cache[toks][1])

        # otherwise compute & insert
        t = torch.LongTensor([list(remaining)])
        output, state = predictor(t, state=state)
        cache[pred_tokens] = (output, list(state))
        return output

    def _memo_joint(key, inp, cache):
        if key in cache:
            return cache[key], False
        rj = joint(*inp)
        cache[key] = rj
        return rj, True

    # high-level memoization for predictor
    #  (prev_tokens) -> (output, state)
    memo_predictor = mk_memo(
        _memo_predictor,
        sz=predictor_cache_sz,
        initial={
            (bos,): (predictor_output, tuple(predictor_state)),
        },
    )

    # high-level memoization for joint
    #  (prev_tokens) -> (output, state)
    memo_joint = mk_memo(_memo_joint, sz=joint_cache_sz)

    # create initial stuff for beam search
    BeamState = BeamStateBuilder(memo_predictor, score_cache_sz)
    initial = BeamState(None, predictor_output, [bos], [1.0])
    beamer = Beamer(
        initial,
        joint_fn=memo_joint,
        beam_width=beam_width,
        topk_next=topk_next,
        max_iters=max_iters,
        debug=debug,
    )
    return beamer


if __name__ == "__main__":

    # BeamState tests
    a = BeamState(None, None, [1, 1, 1], [1.0, 0.5, 0.6, 0.2])
    b = BeamState(None, None, [1, 1, 1], [1.0, 0.2, 0.1, 0.2])
    assert a > b
    assert max(a, b) == a
