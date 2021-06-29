#!/usr/bin/env python

import os
import time
import sys
import queue
import logging
import struct
import yaml
import json
from threading import Thread
from functools import partial

import grpc
import tornado.web
import tornado.websocket
from tornado.ioloop import PeriodicCallback
from google.protobuf import json_format

import libreasr.api.interfaces.libreasr_pb2 as ap
import libreasr.api.interfaces.libreasr_pb2_grpc as apg
from libreasr.lib.inference.utils import load_config


DUMP_AUDIO = False
DEBUG = False
GRPC_TIMEOUT = 2.0


def log_print(*args, **kwargs):
    print("[api-bridge]", *args, **kwargs)


def choose_channel(conf, lang):
    p = conf.get("overrides", {}).get("languages", [])
    if lang in p:
        p = p[lang].get("grpc_port", 50051)
    else:
        p = 50051
    return f"localhost:{p}"


def grpc_thread_func(conf, lang, q_recv, q_send):
    # choose channel & connect
    chan = choose_channel(conf, lang)
    log_print(f"gRPC connecting to {chan}")
    with grpc.insecure_channel(chan) as channel:
        log_print("gRPC connected")
        stub = apg.LibreASRStub(channel)

        def yielder():
            while True:
                try:
                    itm = q_recv.get(timeout=GRPC_TIMEOUT)
                    yield itm
                except:
                    return

        # inference
        for event in stub.TranscribeStream(yielder()):
            log_print(event)
            q_send.put(event)
        log_print("gRPC stopped")


class APIHandler(tornado.web.RequestHandler):
    def initialize(self, conf):
        self.conf = conf
        info = {k: {"code": k, **v} for k, v in conf["overrides"]["languages"].items()}
        self.langs = {"languages": list(info.values())}

    def set_default_headers(self):
        self.set_header("Content-Type", "application/json")
        self.set_header("Access-Control-Allow-Origin", "*")

    def get(self):
        raw = json.dumps(self.langs)
        self.write(raw)


class WebSocket(tornado.websocket.WebSocketHandler):
    def initialize(self, conf):
        self.conf = conf
        self.ready = lambda: False

    def start_grpc_thread(self, lang):
        # start grpc thread
        q_recv, q_send = queue.SimpleQueue(), queue.SimpleQueue()
        t = Thread(target=grpc_thread_func, args=(self.conf, lang, q_recv, q_send))
        t.start()
        self.q_recv = q_recv
        self.q_send = q_send
        self.ready = lambda: t.is_alive()
        log_print("gRPC thread started")

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

    def on_message(self, message):
        payload = message

        # print
        if DEBUG:
            sys.stderr.write(f"recv: {len(payload)} bytes\n")
            sys.stderr.flush()

        ## decode
        # lang
        lang = payload[:4].decode("ascii").strip()
        payload = payload[4:]
        # sr
        unp = struct.unpack("f", payload[:4])
        sr = int(unp[0])
        payload = payload[4:]
        # data
        data = payload
        if DUMP_AUDIO:
            sys.stdout.buffer.write(data)
            sys.stdout.flush()

        # make sure we're ready
        if not self.ready():
            self.start_grpc_thread(lang=lang)

        # on queue
        q_recv = self.q_recv
        q_send = self.q_send
        q_recv.put_nowait(ap.Audio(data=data, sr=sr))

        # check results
        while q_send.qsize() > 0:
            res = q_send.get_nowait()

            # convert to json
            obj = json_format.MessageToDict(res)
            s = json.dumps(obj)

            # reply
            self.write_message(s)


if __name__ == "__main__":

    # parse args
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf",
        "--config",
        default="./config/deploy.yaml",
        help="Path to LibreASR configuration file",
    )
    parser.add_argument(
        "--web-path",
        default="./examples/web/build",
        help="Path to LibreASR web app build directory",
    )
    args = parser.parse_args()

    # load config
    conf = load_config(args.conf, None)

    # start websocket server
    opts = {"conf": conf}
    handlers = [
        (r"/api", APIHandler, opts),
        (r"/websocket", WebSocket, opts),
        (
            r"/(.*)",
            tornado.web.StaticFileHandler,
            {"path": args.web_path, "default_filename": "index.html"},
        ),
    ]
    application = tornado.web.Application(handlers, debug=True)
    application.listen(8080)
    log_print("running on :8080 in", os.getcwd())
    tornado.ioloop.IOLoop.instance().start()
