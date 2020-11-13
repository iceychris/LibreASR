#!/usr/bin/env python

import time
import sys
import queue
import logging
import struct
from threading import Thread
from functools import partial

import grpc
import tornado.web
import tornado.websocket
from tornado.ioloop import PeriodicCallback

import interfaces.libreasr_pb2 as ap
import interfaces.libreasr_pb2_grpc as apg


DUMP_AUDIO = False
DEBUG = False
GRPC_TIMEOUT = 2.0


def log_print(*args, **kwargs):
    print("[api-bridge]", *args, **kwargs)


def choose_channel(lang):
    return {"en": "localhost:50051", "de": "localhost:50052", "fr": "localhost:50053",}[
        lang
    ]


def grpc_thread_func(lang, q_recv, q_send):
    # choose channel & connect
    with grpc.insecure_channel(choose_channel(lang)) as channel:
        log_print("gRPC connected")
        stub = apg.ASRStub(channel)

        def yielder():
            while True:
                try:
                    itm = q_recv.get(timeout=GRPC_TIMEOUT)
                    yield itm
                except:
                    return

        # inference
        for transcript in stub.TranscribeStream(yielder()):
            log_print("Transcript:", transcript.data)
            q_send.put(transcript)
        log_print("gRPC stopped")


class WebSocket(tornado.websocket.WebSocketHandler):
    def initialize(self):
        self.ready = lambda: False

    def start_grpc_thread(self, lang=None):
        # start grpc thread
        q_recv, q_send = queue.SimpleQueue(), queue.SimpleQueue()
        t = Thread(target=grpc_thread_func, args=(lang, q_recv, q_send))
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
        try:
            while q_send.qsize() > 0:
                res = q_send.get_nowait()
                self.write_message(res.data)
        except:
            pass


if __name__ == "__main__":

    # start websocket server
    handlers = [
        (r"/asupersecretwebsocketpath345", WebSocket, {}),
        (
            r"/(.*)",
            tornado.web.StaticFileHandler,
            {"path": "./apps/web/build", "default_filename": "index.html"},
        ),
    ]
    application = tornado.web.Application(handlers, debug=True)
    application.listen(8080)
    log_print("running on :8080")
    tornado.ioloop.IOLoop.instance().start()
