import argparse
from concurrent import futures
import time
import math
import logging
from pathlib import Path
import itertools
from multiprocessing import Process

import grpc

import libreasr.api.interfaces.libreasr_pb2 as ap
import libreasr.api.interfaces.libreasr_pb2_grpc as apg
from libreasr.lib.inference.imports import *
from libreasr.lib.inference.events import *
from libreasr.lib.inference.main import load_stuff
from libreasr.lib.inference.utils import load_config
from libreasr.lib.defaults import ALIASES


# gRPC thread pool
WORKERS = 4


def log_print(*args, **kwargs):
    print("[api-server]", *args, **kwargs)


class LibreASRServicer(apg.LibreASRServicer):
    def __init__(self, libreasr, lang):
        self.lang_name = lang
        self.l = libreasr

    def Transcribe(self, request, context):

        # tensorize
        aud, sr = request.data, request.sr
        aud = tensorize(aud)

        # print
        log_print(f"Transcribe(lang={self.lang_name}, sr={sr}, shape={aud.shape})")

        # inference
        aud = AudioTensor(aud, sr)
        transcript = self.l.transcribe(aud)

        return ap.Transcript(data=transcript)

    def TranscribeStream(self, request_iterator, context):
        # peek at the first frame
        #  (for getting sr)
        frame = request_iterator.next()
        sr = frame.sr
        unpeeked = itertools.chain([frame], request_iterator)

        # print
        log_print(
            f"TranscribeStream(lang={self.lang_name}, sr={sr}, shape=({len(frame.data)}))"
        )

        # inference
        kwargs = {
            "sr": sr,
            "stream_opts": {
                "assistant": True
            }
        }
        for event in self.l.stream(iter(unpeeked), **kwargs):
            # convert to protobuf
            #  and reply in a streaming fashion
            yield event.to_protobuf()
        log_print(f"... done.")


def serve(lang, config_path=None):

    # load model
    from libreasr import LibreASR

    def hook(conf):
        conf["model"]["loss"] = False
        conf["cuda"]["enable"] = False
        conf["cuda"]["device"] = "cpu"
        conf["model"]["load"] = True
        conf["model"]["path"] = {"n": "./models/de-4096.pth"}

    libreasr = LibreASR(lang, config_path=config_path, config_hook=hook)
    libreasr.load_inference()

    # bring up server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=WORKERS))
    apg.add_LibreASRServicer_to_server(LibreASRServicer(libreasr, lang), server)

    # start gRPC server
    port = f"[::]:{libreasr.get_grpc_port()}"
    log_print("gRPC server starting on", port, "language", lang)
    server.add_insecure_port(port)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang", default="all", help="Languages to serve ('all', 'en,de', ...)"
    )
    parser.add_argument(
        "--conf",
        "--config",
        default=None,
        help="Path to LibreASR configuration file",
    )
    args = parser.parse_args()
    lang = args.lang.lower()
    logging.basicConfig()

    # spawn one process for each language
    #  or just the desired one
    if args.lang.lower() == "all":
        ls = list(ALIASES.keys())
    else:
        ls = args.lang.lower().split(",")
    ps = []
    for l in ls:
        p = Process(target=serve, args=(l, args.conf))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()
