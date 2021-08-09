from functools import partial


class Beamsearch:
    def __init__(
        self,
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
        lm=None,
        lm_weight=0.0,
    ):
        self.impl = impl
        self.lang = lang
        if lm is None:
            lm_weight = 0.0
        if impl.lower() == "libreasr":
            from libreasr.lib.inference.beamsearch.libreasr import (
                start_rnnt_beam_search,
            )

            self.beamer = start_rnnt_beam_search(
                beam_search_opts, blank, bos, lang, p, j, po, ps, mi
            )
        else:
            from libreasr.lib.inference.beamsearch.speechbrain import (
                TransducerBeamSearcher,
            )

            self.beamer = TransducerBeamSearcher(
                decode_network_lst=[p],
                tjoint=partial(j, softmax=False),
                classifier_network=[],
                blank_id=blank,
                bos_id=bos,
                beam_size=beam_search_opts.pop("beam_width"),
                nbest=2,
                lm_module=lm,
                lm_weight=lm_weight,
                state_beam=beam_search_opts.pop("state_beam"),
                expand_beam=beam_search_opts.pop("expand_beam"),
            )
            self.beamer.forward_init(bs=1, device=dev)

    def _mk_event(self, hyps):
        from libreasr.lib.inference.events import (
            HypothesisEvent,
            TranscriptEvent,
        )

        if self.impl.lower() == "libreasr":
            return HypothesisEvent(hyps)
        else:
            return TranscriptEvent(self.lang.denumericalize(hyps[0]))

    def _dispatch(self, h_enc):
        if self.impl.lower() == "libreasr":
            hyps = self.beamer(h_enc)
            return hyps
        else:
            hyps, scores, _, _ = self.beamer.forward_step(h_enc[None, None])
            return hyps

    def __call__(self, h_enc, return_event=False):
        def ret_fn(hyps):
            if return_event:
                return self._mk_event(hyps)
            return hyps

        hyps = self._dispatch(h_enc)
        return ret_fn(hyps)
