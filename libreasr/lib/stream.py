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
from libreasr.lib.defaults import DEFAULT_STREAM_OPTS, DEFAULT_STREAM_CHUNK_SZ


def transcribe_stream(
    sth, model, x_tfm_stream, lang, sr=16000, stream_opts=DEFAULT_STREAM_OPTS, **kwargs
):
    """
    Transcribe an audio stream.
    :sth: is supposed to be a generator yielding
    20ms chunks of raw audio data.
    `buffer_n_frames` are then concatenated and transcribed.
    """
    stream_opts = update(DEFAULT_STREAM_OPTS, stream_opts)

    def stream_fn():
        started = False
        frames = []
        printed = False
        for i, frame in enumerate(sth):
            # fill up frames
            t = tensorize(frame)
            frames.append(t)

            # may continue?
            if not len(frames) == stream_opts["buffer_n_frames"]:
                continue

            # cat all frames
            aud = torch.cat(frames, dim=1)

            # clear first
            del frames[0]

            # convert to AudioTensor
            aud = AudioTensor(aud, sr)

            # apply transforms
            aud = x_tfm_stream(aud)

            yield aud

    # inference
    outputs = model.transcribe_stream(stream_fn(), lang.denumericalize, **kwargs)
    last = ""
    for i, (y, reset_fn) in enumerate(outputs):
        y = lang.denumericalize(y)
        if y != last:
            last = y
            yield y


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
