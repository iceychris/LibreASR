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
from libreasr.lib.defaults import LANGUAGES
from libreasr.lib.inference.events import EventTag
from libreasr import LibreASR


# gRPC settings
EXECUTER = futures.ThreadPoolExecutor


def log_print(*args, **kwargs):
    print("[api-server]", *args, **kwargs)


def get_cached_libreasr_instance(cache, lang):
    if lang in cache:
        return cache[lang]
    l = LibreASR(lang)
    cache[lang] = l
    return l


class LibreASRServicer(apg.LibreASRServicer):
    def __init__(self, languages):
        self.cache = {}
        for l in languages:
            get_cached_libreasr_instance(self.cache, l)

    def Transcribe(self, request, context):

        # tensorize
        aud, sr, lang = request.data, request.sr, request.lang
        aud = tensorize(aud)

        # print
        log_print(f"Transcribe(lang={lang}, sr={sr}, shape={aud.shape})")

        # grab instance
        l = get_cached_libreasr_instance(self.cache, lang)

        # inference
        aud = AudioTensor(aud, sr)
        transcript = l.transcribe(aud)

        return ap.Transcript(data=transcript)

    def TranscribeStream(self, request_iterator, context):
        # peek at the first frame
        #  (for getting sr)
        frame = request_iterator.next()
        sr = frame.sr
        lang = frame.lang
        unpeeked = itertools.chain([frame], request_iterator)

        # print
        log_print(f"TranscribeStream(lang={lang}, sr={sr}, shape=({len(frame.data)}))")

        # inference
        kwargs = {"sr": sr, "stream_opts": {"assistant": False, "debug": False}}

        # grab instance
        l = get_cached_libreasr_instance(self.cache, lang)

        for event in l.stream(iter(unpeeked), **kwargs):
            # convert to protobuf
            #  and reply in a streaming fashion
            if event.tag in (EventTag.TRANSCRIPT, EventTag.SENTENCE):
                yield event.to_protobuf()
        log_print(f"... done.")


def serve(languages, port="[::]:50051", workers=4):

    # bring up server
    server = grpc.server(EXECUTER(max_workers=workers))
    apg.add_LibreASRServicer_to_server(LibreASRServicer(languages), server)

    # start gRPC server
    log_print("gRPC server starting on", port)
    server.add_insecure_port(port)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        default="en",
        help="Language-codes of models to pre-load in the cache with ('en', 'en,de', ...)",
    )
    parser.add_argument(
        "--grpc-workers",
        default=4,
        type=int,
        help="Number of max_workers argument for ThreadPoolExecuter",
    )
    parser.add_argument(
        "--grpc-port",
        default="[::]:50051",
        help="gRPC port to listen on",
    )
    args = parser.parse_args()
    cache = args.cache.lower()
    logging.basicConfig()

    # spawn one process for each language
    #  or just the desired one
    if args.cache.lower() == "all":
        ls = LANGUAGES
    else:
        ls = args.cache.lower().split(",")
    serve(ls, port=args.grpc_port, workers=args.grpc_workers)
