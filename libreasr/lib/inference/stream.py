import inspect
import types
import queue
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from libreasr.lib.inference.imports import *
from libreasr.lib.inference.processors import *
from libreasr.lib.inference.inference import infer_stream
from libreasr.lib.inference.events import SentenceEvent
from libreasr.lib.defaults import DEFAULT_STREAM_OPTS, DEFAULT_STREAM_CHUNK_SZ


def transcribe_stream(
    gen, model, x_tfm_stream, lang, sr=16000, stream_opts=DEFAULT_STREAM_OPTS, **kwargs
):
    """
    Transcribe an audio stream.
    :gen: is supposed to be a generator yielding
    20ms chunks of raw audio data.
    `buffer_n_frames` are then concatenated and transcribed.
    """

    ###
    # setup
    ###

    stream_opts = update(DEFAULT_STREAM_OPTS, stream_opts)
    assis, assis_kw, dbg_proc = stream_opts["assistant"], stream_opts["assistant_keywords"], stream_opts["debug"]
    if assis:
        processors = [
            VADProcessor(sr=sr, debug=False),
            WakewordProcessor(sr=sr, debug=False, keywords=assis_kw),
            EOSProcessor(debug=False),
            TranscriptProcessor(lang.denumericalize),
            SentenceProcessor(),
            AssistantProcessor(),
            TTSProcessor(),
        ]
    else:
        processors = [
            EOSProcessor(),
            TranscriptProcessor(lang.denumericalize),
            SentenceProcessor(),
        ]
    if dbg_proc:
        processors.append(EventDebugProcessor())


    ###
    # input processing
    ###
    
    # tensorize
    gen = map(tensorize, gen)

    # buffer n frames
    def buffer(gen, n_frames=stream_opts["buffer_n_frames"]):
        frames = []
        for frame in gen:
            frames.append(frame)
            if len(frames) == n_frames:
                catted = torch.cat(frames, dim=1)
                frames.clear()
                yield catted 
    gen = buffer(gen)

    # convert to AudioTensor
    gen = map(partial(AudioTensor, sr=sr), gen)

    # apply transforms
    gen = map(x_tfm_stream, gen)

    # go through processors
    #  and flatten list of segments in between
    def flatten(gen):
        for segments in gen:
            for seg in segments:
                yield seg
    procs = MultiInferenceProcessor(processors, flatten_fn=flatten)
    gen = procs(gen)

    # make ready for model
    def add_dims(segment):
        if torch.is_tensor(segment):
            return segment[:, :, None, None]
        else: return segment
    gen = map(add_dims, gen)

    ###
    # output processing
    ###

    # inference
    output_found = False
    with torch.no_grad():

        # start inference
        events = infer_stream(model, gen, **kwargs)

        # output processing
        last = ""
        for i, event in enumerate(events):
            procs.emit_model_event(event)
            events = procs.get_output_events()
            for output_event in events:
                output_found = True
                yield output_event
            procs.clear_output_events()

    # if we haven't found any hypothesis
    if not output_found:
        yield SentenceEvent("")


def path_to_audio_generator(
    path: str, secs=DEFAULT_STREAM_CHUNK_SZ, to_sr=16000, start_frames=1, end_frames=2
):
    """
    Load audio from a path `path` via torchaudio,
    resample it to `to_sr` and return chunks
    of size `secs`.
    """
    data, sr = torchaudio.load(path)
    data = data[0][None]
    data = torchaudio.transforms.Resample(sr, to_sr)(data)
    data = data.numpy().astype(np.float32).tobytes()
    sr = to_sr
    slice_sz = int(secs * sr) * 4
    l = len(data) // slice_sz

    # [start] zero
    for _ in range(start_frames):
        yield bytes([0] * slice_sz)

    # [mid] transmit audio
    for i in range(l):
        chunk = data[i * slice_sz : (i + 1) * slice_sz]
        # pad with zeros
        chunk = chunk + bytes([0] * (slice_sz - len(chunk)))
        assert len(chunk) % 4 == 0
        yield chunk

    # [end] zero frames mark end
    for _ in range(end_frames):
        yield bytes([0] * slice_sz)
