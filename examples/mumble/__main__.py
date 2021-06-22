import argparse
from multiprocessing import Process
from functools import partial
import subprocess
import time
import queue
import pprint
import os
from subprocess import Popen

import numpy as np
from scipy.signal import resample_poly

import pymumble_py3
from pymumble_py3.callbacks import (
    PYMUMBLE_CLBK_SOUNDRECEIVED as PCS,
    PYMUMBLE_CLBK_CONNECTED as CONN,
)

from libreasr import LibreASR
from libreasr.lib.inference.events import *


class Chunker:
    def __init__(self, sr=48000, duration=0.08):
        self.last = None
        self.sr = sr
        self.duration = duration
        self.cut_at = int(sr * duration)

    def __call__(self, raw):
        # convert to numpy float32
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32_767.

        # add last
        if self.last is not None:
            arr = np.concatenate((self.last, arr), axis=0)

        # slice
        if len(arr) < self.cut_at:
            self.last = arr
            return None, False
        else:
            a, b = arr[: self.cut_at], arr[self.cut_at :]
            self.last = b
            return a, True


def check_user(name, whitelist):
    ltu = whitelist
    if ltu is None or (isinstance(ltu, str) and ltu == "all"):
        return True
    return name in whitelist


def on_connected(channel, *args, **kwargs):
    # move into specified channel
    chans = mumble.channels
    print("[Mumble] available channels:")
    pprint.pprint(chans)
    chans[get_channel_id(mumble, channel)].move_in()


def on_sound_received(q_recv, q_send, chunker, whitelist, user, soundchunk):
    chunk = soundchunk.pcm
    dur = soundchunk.duration

    # extract username
    name = user["name"]
    if check_user(name, whitelist):

        # transcribe
        chunk, ready = chunker(chunk)
        if ready:
            q_recv.put(chunk, block=False)

def libreasr_process(lang, config_path, q_recv, q_send, sr=48000):

    def hook(conf):
        conf["model"]["loss"] = False
        conf["cuda"]["enable"] = False
        conf["cuda"]["device"] = "cpu"
        conf["model"]["load"] = True
        conf["model"]["path"] = {"n": "./models/de-4096.pth"}

    libreasr = LibreASR(lang, config_path=config_path, config_hook=hook)
    libreasr.load_inference()

    # main loop
    gen = libreasr.stream(q_recv, sr=sr)
    for event in gen:
        if event.tag == EventTag.SENTENCE:
            text = event.transcript
            q_send.put(text, block=False)
        if event.tag == EventTag.TTS:
            audio = event.audio
            q_send.put(audio, block=False)

def get_channel_id(mumble, channel):
    for cid, vals in mumble.channels.items():
        if vals["name"] == channel:
            return cid


def chat(mumble, channel, text):
    chans = mumble.channels
    chans[get_channel_id(mumble, channel)].send_text_message(text)


def resample(audio, from_sr=22050, to_sr=48000):
    audio = audio.astype(np.float32)
    audio = resample_poly(audio, to_sr, from_sr)#  / 32_767.
    audio = audio.astype(np.int16)
    audio = audio.tobytes()
    return audio


def voice(mumble, audio, do_resample=True):
    if do_resample:
        audio = resample(audio)
    mumble.sound_output.add_sound(audio)


if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("server", help="URL of Mumble server to use")
    parser.add_argument(
        "--nickname", default="LibreASR", help="Bot Nickname of Mumble server to use"
    )
    parser.add_argument("--password", default="", help="Mumble server password")
    parser.add_argument("--channel", default=0, help="Mumble channel name to join")
    parser.add_argument(
        "--users",
        default="all",
        help="Users LibreASR should listen to ('all', 'joe,mike')",
    )
    parser.add_argument(
        "--dry-run", default=False, help="Don't connect to Mumble, just load LibreASR"
    )
    parser.add_argument(
        "--lang", default="de", help="Language to use ('en', 'de', ...)"
    )
    parser.add_argument(
        "--conf",
        "--config",
        default="./config/base.yaml",
        help="Path to LibreASR configuration file",
    )
    args = parser.parse_args()
    server = args.server
    nick = args.nickname
    password = args.password
    users = args.users
    if not users == "all":
        users = users.split(",")
    channel = args.channel
    lang = args.lang.lower()
    config_path = args.conf
    dry_run = args.dry_run

    # prepare queues
    q_recv, q_send = queue.Queue(), queue.Queue()

    ###
    # initialize mumble bot
    ###

    # preload with stuff
    on_sound_received = partial(on_sound_received, q_recv, q_send, Chunker(), users)
    on_connected = partial(on_connected, channel)

    # start mumble
    if not dry_run:
        mumble = pymumble_py3.Mumble(server, nick, password=password)
        mumble.callbacks.set_callback(CONN, on_connected)
        mumble.callbacks.set_callback(PCS, on_sound_received)
        mumble.set_receive_sound(1)  # we want to receive sound
    else:
        mumble = None

    ###
    # initialize LibreASR subprocess
    ###

    # start LibreASR (separate thread)
    from threading import Thread

    t = Thread(target=libreasr_process, args=(lang, config_path, q_recv, q_send))
    t.start()

    # start mumble (this thread)
    if not dry_run:
        mumble.start()

    # loop forever
    while 1:
        # continuously receive & output transcripts
        if not dry_run:
            result = None
            while q_send.qsize() > 0:
                result = q_send.get()
                if result is not None:
                    if isinstance(result, str):
                        chat(mumble, channel, result)
                    elif isinstance(result, np.ndarray):
                        voice(mumble, result)
        time.sleep(0.5)
