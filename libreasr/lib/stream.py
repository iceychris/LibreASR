import inspect
import types
import queue
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from libreasr.lib.inference.imports import *
from libreasr.lib.defaults import DEFAULT_STREAM_OPTS


def should_reset(steps, downsample, n_buffer, thresh):
    # one step length
    steps = int(10.0 * downsample * n_buffer * steps)
    if steps >= thresh:
        # print("reset")
        return True
    return False


def transcribe_stream(
    sth, model, x_tfm_stream, lang, sr=16000, stream_opts=DEFAULT_STREAM_OPTS, **kwargs
):
    """
    Transcribe an audio stream.
    :sth: is supposed to be a generator yielding
    80ms chunks of raw audio data.
    """
    stream_opts = update(DEFAULT_STREAM_OPTS, stream_opts)

    def stream_fn():
        started = False
        frames = []
        counter = 0
        printed = False
        for i, frame in enumerate(sth):
            # fill up frames
            # TODO check for type
            t = tensorize(frame)
            frames.append(t)
            counter += 1

            # may continue?
            if not len(frames) == stream_opts["buffer_n_frames"]:
                continue

            # cat all frames
            aud = torch.cat(frames, dim=1)

            # clear first
            del frames[0]

            # convert to AudioTensor
            aud = AudioTensor(aud, sr)

            # print
            # if not printed:
            #     print(
            #         f"TranscribeStream(sr={sr}, shape={aud.shape})"
            #     )
            #     printed = True

            aud = x_tfm_stream(aud)
            yield aud

    # inference
    outputs = model.transcribe_stream(stream_fn(), lang.denumericalize, **kwargs)
    last = ""
    last_diff = ""
    steps = 0
    for i, (y, y_one, reset_fn) in enumerate(outputs):
        steps += 1
        if y_one != "":
            now = lang.denumericalize(y)
            diff = "".join(y for x, y in itertools.zip_longest(last, now) if x != y)
            last = now
            # bail if we just output the same thing twice
            if diff == last_diff:
                continue
            last_diff = diff
            yield (diff, now)
        elif should_reset(
            steps,
            stream_opts["downsample"],
            stream_opts["n_buffer"],
            stream_opts["reset_thresh"],
        ):
            reset_fn()
            steps = 0
