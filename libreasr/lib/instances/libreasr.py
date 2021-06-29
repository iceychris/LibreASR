import collections
import inspect
import types
import queue
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from libreasr.lib.defaults import DEFAULT_SR, DEFAULT_BATCH_SIZE
from libreasr.lib.inference.events import EventTag
from libreasr.lib.instances import BaseInstance


class LibreASRInstance(BaseInstance):
    def __init__(self, lang, config_path=None, **kwargs):
        super().__init__()
        self.config_path = config_path
        from libreasr.lib.inference.main import load_stuff

        self.conf, self.lang, self.model, self.x_tfm, self.x_tfm_stream = load_stuff(
            lang, config_path=config_path, **kwargs
        )

    def transcribe(self, sth, batch=True, **kwargs):
        # update with tuned values if available
        kwargs.update(self.conf.get("hypers", {}).get("tuned", {}))

        # transcribe
        only_one = False
        if not isinstance(sth, (list, tuple)):
            sth = [sth]
            only_one = True
        transcripts = []
        if batch:
            res = self._transcribe_batches(sth, **kwargs)
            transcripts = [*res]
        else:
            for obj in sth:
                res = self._transcribe_one(obj, **kwargs)
                transcripts.append(res)
        if only_one:
            return transcripts[0]
        return transcripts

    def stream(self, sth, batch=False, **kwargs):
        assert not batch, "Batched streaming is not supported"

        # update with tuned values if available
        kwargs.update(self.conf.get("hypers", {}).get("tuned", {}))

        # go!
        if inspect.isgeneratorfunction(sth):
            sth = sth()
        if isinstance(
            sth,
            (
                types.GeneratorType,
                str,
                list,
                tuple,
                map,
                filter,
                collections.abc.Iterator,
                pathlib.Path,
            ),
        ):
            return self._transcribe_stream_generator(sth, **kwargs)
        elif isinstance(sth, queue.Queue):
            return self._transcribe_stream_queue(sth, **kwargs)
        else:
            raise NotImplementedError(f"Streaming type {type(sth)} is not supported")

    def _transcribe_one(self, obj, **kwargs):
        # load & convert to tensor
        obj, sr = _to_tensor(obj)

        # transcribe
        return _transcribe_tensor(self.model, self.x_tfm, obj, sr, **kwargs)

    def _transcribe_batches(self, obj, from_dataloader=False, **kwargs):
        # shortcut
        if from_dataloader:
            return _infer(self.model, obj, None, batched=True, **kwargs)

        # convert to tensors
        tensors = [_to_tensor(x) for x in obj]

        # apply transforms
        tensors = [_apply_transforms(self.x_tfm, *x) for x in tensors]

        # batch tensors
        batches, lens = _batch_tensors(tensors)

        # transcribe
        return _infer(self.model, batches, lens, batched=True, **kwargs)

    def _transcribe_stream_generator(self, sth, **kwargs):
        from libreasr.lib.inference.stream import (
            transcribe_stream,
            path_to_audio_generator,
        )

        def stream(sth):
            return transcribe_stream(
                sth, self.model, self.x_tfm_stream, self.lang, **kwargs
            )

        def get_result(stream_output) -> str:
            # grab last output
            l = list(stream_output)
            filtered = list(filter(lambda x: x.tag == EventTag.SENTENCE, l))
            last = filtered[-1]
            return last.transcript

        transcripts = None
        if isinstance(sth, (str, pathlib.Path)):
            # turn into generator
            sth = path_to_audio_generator(sth)
            transcripts = get_result(stream(sth))
        elif isinstance(sth, list) and len(sth) > 0 and isinstance(sth[0], str):
            transcripts = []
            for s in sth:
                s = path_to_audio_generator(s)
                transcripts.append(get_result(stream(s)))
        else:
            transcripts = stream(sth)
        return transcripts

    def _transcribe_stream_queue(self, sth, **kwargs):
        from libreasr.lib.inference.stream import transcribe_stream

        def yielder():
            while True:
                try:
                    item = sth.get_nowait()
                    yield item
                except:
                    pass

        return transcribe_stream(
            yielder(), self.model, self.x_tfm_stream, self.lang, **kwargs
        )


class LibreASRTraining(LibreASRInstance):
    def __init__(self, lang, config_path, **kwargs):
        super().__init__()
        self.config_path = config_path
        from libreasr.lib.imports import parse_and_apply_config

        (
            self.conf,
            self.lang,
            self.builder_train,
            self.builder_valid,
            self.db,
            self.model,
            self.learn,
        ) = parse_and_apply_config(path=config_path, lang=lang, **kwargs)


class LibreASRInference(LibreASRInstance):
    def __init__(self, lang, config_path, **kwargs):
        self.config_path = config_path
        from libreasr.lib.inference.main import load_stuff

        self.conf, self.lang, self.model, self.x_tfm, self.x_tfm_stream = load_stuff(
            lang, config_path=config_path, **kwargs
        )


def _flatten(l):
    return [item for sublist in l for item in sublist]


def _to_tensor(obj):
    # already tensor
    if torch.is_tensor(obj):
        return obj, DEFAULT_SR

    # filepath (load)
    elif isinstance(obj, str):
        import torchaudio

        aud, sr = torchaudio.load(obj)
        return aud, sr

    # not supported
    else:
        raise Exception(f"Can't convert type {type(obj)} to tensor.")


def _apply_transforms(x_tfm, t, sr):
    from libreasr.lib.inference.main import AudioTensor

    aud = AudioTensor(t, sr)
    aud = x_tfm(aud)[0]
    return aud


def _batch_tensors(tensors, bs=DEFAULT_BATCH_SIZE):
    l = len(tensors)
    batches, lens = [], []
    back = tensors[0].shape[1:]
    t_idx = 0
    for i in range(0, l, bs):
        rng = tensors[i : i + bs]
        ls = [x.size(t_idx) for x in rng]
        lmax = max(ls)
        shp = (len(rng), lmax, *back)
        b = torch.zeros(shp).float()
        for j, t in enumerate(rng):
            b[j][: t.size(t_idx)] = t
        batches.append(b)
        lens.append(torch.LongTensor(ls))
    return batches, lens


def _transcribe_tensor(model, x_tfm, t, sr, **kwargs):
    aud = _apply_transforms(x_tfm, t, sr)
    return _infer(model, aud, batched=False, **kwargs)


def _infer(model, smpl, lens=None, batched=True, **kwargs):
    with torch.no_grad():
        if batched:
            if lens is None:
                return model.transcribe_batch(smpl, **kwargs)
            return _flatten(
                [model.transcribe_batch(x, **kwargs) for x in zip(smpl, lens)]
            )
        else:
            return model.transcribe(smpl, **kwargs)[0]
