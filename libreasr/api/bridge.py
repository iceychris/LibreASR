#!/usr/bin/env python

import os
import time
import sys
import logging
import struct
import yaml
import json
from functools import partial
import multiprocessing as mp
from multiprocessing import Process, Queue

import grpc
import tornado.web
import tornado.websocket
from tornado.ioloop import PeriodicCallback
from google.protobuf import json_format

import libreasr.api.interfaces.libreasr_pb2 as ap
import libreasr.api.interfaces.libreasr_pb2_grpc as apg


DUMP_AUDIO = False
DEBUG = False
GRPC_TIMEOUT = 5.0
TIMER_MILLIS = 100


def log_print(*args, **kwargs):
    print("[api-bridge]", *args, **kwargs)


def with_grpc_api(chan="localhost:50051"):
    def wrap(f):
        def inner(*args, **kwargs):
            log_print(f"gRPC connecting to {chan}...")
            conn = grpc.insecure_channel(chan)
            with conn as channel:
                stub = apg.LibreASRStub(channel)
                log_print("gRPC connected.")
                return f(stub, *args, **kwargs)

        return inner

    return wrap


def grpc_transcribe_stream(stub, lang, q_recv, q_send):
    log_print(f"grpc_transcribe_stream (lang {lang})")

    def yielder():
        while True:
            try:
                itm = q_recv.get(timeout=GRPC_TIMEOUT)

                # build grpc representation
                data, sr, lang = itm
                itm = ap.Audio(data=data, sr=sr, lang=lang)

                yield itm
            except:
                return

    # inference
    for event in stub.TranscribeStream(yielder()):
        log_print(event)

        # convert to json
        event = json_format.MessageToDict(event)
        event = json.dumps(event)

        # send back
        q_send.put(event)
    log_print("gRPC stopped")


def grpc_translate(stub, src, tgt, text):
    result = stub.Translate(ap.Text(src=src, tgt=tgt, text=text))
    src, tgt, text = result.src, result.tgt, result.text
    return src, tgt, text


class APIHandler(tornado.web.RequestHandler):
    def initialize(self, chan):
        from libreasr.lib.defaults import LANGUAGES

        info = {l: {"code": l, "name": l, "enable": True} for l in LANGUAGES}
        self.langs = {"languages": list(info.values())}
        self.chan = chan

    def set_default_headers(self):
        self.set_header("Content-Type", "application/json")
        self.set_header("Access-Control-Allow-Origin", "*")

    async def get(self, path, *args, **kwargs):

        # info
        if path == "":
            raw = json.dumps(self.langs)
            self.write(raw)

        else:
            self.set_status(404)
            self.finish("{}")

    async def post(self, path, *args, **kwargs):

        # transcription API
        if path == "/transcribe":
            self.set_status(404)
            self.finish("{}")

        # translation API
        elif path == "/translate":
            src, tgt, text = map(
                lambda x: self.get_argument(x, ""), ["src", "tgt", "text"]
            )
            if src != "" and tgt != "" and text != "":
                src, tgt, text = with_grpc_api(chan=self.chan)(grpc_translate)(src, tgt, text)
            else:
                print("/translate empty")
            res = {
                "src": src,
                "tgt": tgt,
                "text": text,
            }
            self.write(res)

        else:
            self.set_status(404)
            self.finish("{}")


class WebSocket(tornado.websocket.WebSocketHandler):
    def initialize(self, chan):
        self.chan = chan
        self.ready = lambda: False
        self.timer = None
        self.closed = False
        self.info = {}

    def _timer_callback(self):
        # check results
        while self.q_send.qsize() > 0:
            try:
                event = self.q_send.get_nowait()
            except:
                continue

            # reply to websocket peer
            self.write_message(event)

        # if the process is gone, stop the timer
        #  and disconnect
        if not self.ready():
            self.stop_timer()

    def start_timer(self):
        self.timer = PeriodicCallback(self._timer_callback, TIMER_MILLIS)
        self.timer.start()

    def stop_timer(self):
        if self.timer is not None and self.timer.is_running():
            self.timer.stop()
            self.timer = None
        if not self.closed:
            self.close()

    def start_grpc_process(self, lang):
        # start grpc process
        q_recv, q_send = Queue(), Queue()
        f = with_grpc_api(chan=self.chan)(grpc_transcribe_stream)
        s = Process(target=f, args=(lang, q_recv, q_send))
        s.start()
        self.q_recv = q_recv
        self.q_send = q_send
        self.ready = lambda: s.is_alive()
        self.start_timer()
        log_print("gRPC process started")

    def check_origin(self, origin):
        return True

    def set_default_headers(self):
        self.set_header("Content-Type", "application/json")
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "content-type")
        self.set_header(
            "Access-Control-Allow-Methods", "POST, GET, OPTIONS, PATCH, PUT"
        )

    def open(self, *args, **kwargs):
        log_print("ws open")

    def on_close(self):
        log_print("ws close")
        self.closed = True

    def _handle_json(self, message, payload):
        decoded = json.loads(payload)
        self.info = decoded

    def _handle_data(self, message, payload):
        # grab info
        lang = self.info.get("modelId", "en")
        sr = self.info.get("sr", 16000)

        # fix websocket client sending strings
        #  instead of binary data...
        if isinstance(payload, str):
            payload = payload.split(",")
            payload = bytes(map(lambda x: int(x), payload))

        # dump to stdout
        if DUMP_AUDIO:
            sys.stdout.buffer.write(payload)
            sys.stdout.flush()

        # make sure we're ready
        if not self.ready():
            self.start_grpc_process(lang=lang)

        # put on queue
        #  to be consumed by grpc process
        q_recv = self.q_recv
        q_recv.put_nowait((payload, sr, lang))

    def on_message(self, message):

        # print
        if DEBUG:
            sys.stderr.write(f"recv: {len(payload)} bytes\n")
            sys.stderr.flush()

        # check message type
        #  by websocket data type
        payload = message
        if isinstance(message, str) and message[0] == "{":
            self._handle_json(message, payload)
        else:
            self._handle_data(message, payload)


if __name__ == "__main__":

    # use fork
    mp.set_start_method("fork")

    # parse args
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        default=8080,
        help="HTTP port to listen on",
    )
    parser.add_argument(
        "--web-path",
        default="./examples/web/build",
        help="Path to LibreASR web app build directory",
    )
    parser.add_argument(
        "--grpc-host",
        default="localhost:50051",
        help="URL of LibreASR server to connect to",
    )
    args = parser.parse_args()

    # start http/websocket server
    opts = {"chan": args.grpc_host}
    handlers = [
        (r"/api(.*)", APIHandler, opts),
        (r"/websocket", WebSocket, opts),
        (
            r"/(.*)",
            tornado.web.StaticFileHandler,
            {"path": args.web_path, "default_filename": "index.html"},
        ),
    ]
    application = tornado.web.Application(handlers, debug=True)
    application.listen(args.port)
    log_print(f"running on :{args.port} in", os.getcwd())
    tornado.ioloop.IOLoop.instance().start()
