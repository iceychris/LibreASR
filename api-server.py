from concurrent import futures
import time
import math
import logging
from pathlib import Path
import itertools as it

import grpc

import interfaces.libreasr_pb2 as ap
import interfaces.libreasr_pb2_grpc as apg
from libreasr.lib.inference import *
from libreasr.lib.utils import tensorize


WORKERS = 4
PORTS = {
    "en": "[::]:50051",
    "de": "[::]:50052",
    "fr": "[::]:50053",
}

# streaming
# reset threshold
THRESH = 4000
BUFFER_N_FRAMES = 3


def log_print(*args, **kwargs):
    print("[api-server]", *args, **kwargs)


def get_settings(conf):
    downsample = None
    n_buffer = None
    for tfm in conf["transforms"]["stream"]:
        if tfm["name"] == "StackDownsample":
            downsample = tfm["args"]["downsample"]
        if tfm["name"] == "Buffer":
            n_buffer = tfm["args"]["n_buffer"]
    return downsample, n_buffer


def should_reset(steps, downsample, n_buffer):
    # one step length
    steps = int(10.0 * downsample * n_buffer * steps)
    if steps >= THRESH:
        log_print("reset")
        return True
    return False


class ASRServicer(apg.ASRServicer):
    def __init__(self, lang):
        self.lang_name = lang
        conf, lang, m, x_tfm, x_tfm_stream = load_stuff(lang)
        self.conf = conf
        self.downsample, self.n_buffer = get_settings(conf)
        self.lang = lang
        self.model = m
        self.x_tfm = x_tfm
        self.x_tfm_stream = x_tfm_stream

    def Transcribe(self, request, context):

        # tensorize
        aud, sr = request.data, request.sr
        aud = tensorize(aud)

        # print
        log_print(f"Transcribe(lang={self.lang_name}, sr={sr}, shape={aud.shape})")

        # tfms
        aud = AudioTensor(aud, sr)
        aud = self.x_tfm(aud)[0]

        # inference
        out = self.model.transcribe(aud)

        return ap.Transcript(data=out[0])

    def TranscribeStream(self, request_iterator, context):
        def stream():
            started = False
            frames = []
            counter = 0
            printed = False
            for i, frame in enumerate(request_iterator):
                # fill up frames
                t = tensorize(frame.data)
                frames.append(t)
                counter += 1

                # may continue?
                if not len(frames) == BUFFER_N_FRAMES:
                    continue

                # cat all frames
                aud = torch.cat(frames, dim=1)

                # clear first
                del frames[0]

                # convert to AudioTensor
                aud = AudioTensor(aud, frame.sr)

                # print
                if not printed:
                    log_print(
                        f"TranscribeStream(lang={self.lang_name}, sr={frame.sr}, shape={aud.shape})"
                    )
                    printed = True

                aud = self.x_tfm_stream(aud)
                yield aud

        # inference
        outputs = self.model.transcribe_stream(stream(), self.lang.denumericalize)
        last = ""
        last_diff = ""
        steps = 0
        for i, (y, y_one, reset_fn) in enumerate(outputs):
            steps += 1
            if y_one != "":
                now = self.lang.denumericalize(y)
                diff = "".join(y for x, y in it.zip_longest(last, now) if x != y)
                last = now
                # bail if we just output the same thing twice
                if diff == last_diff:
                    continue
                last_diff = diff
                yield ap.Transcript(data=diff)
            elif should_reset(steps, self.downsample, self.n_buffer):
                reset_fn()
                steps = 0


def serve(lang):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=WORKERS))
    apg.add_ASRServicer_to_server(ASRServicer(lang), server)
    port = PORTS[lang]
    server.add_insecure_port(port)
    server.start()
    log_print("gRPC server running on", port, "language", lang)
    server.wait_for_termination()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("lang", help="language to serve")
    args = parser.parse_args()
    logging.basicConfig()
    serve(args.lang)
