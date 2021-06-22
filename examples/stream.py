import argparse
import os
from threading import Thread
from queue import Queue
import sys
import itertools
import multiprocessing

import numpy as np
import pyaudio


FORMAT = pyaudio.paInt16
CHANNELS = 1


def recorder_task(q, sr, gain):
    """
    Producer. Run this as a separate thread.
    """
    audio = pyaudio.PyAudio()

    def callback(byts, frame_count, time_info, status):
        arr = np.frombuffer(byts, dtype=np.int16)
        arr = arr * gain
        q.put_nowait(arr)
        print("feed")
        return (None, pyaudio.paContinue)

    chunk = int(sr * 0.08)
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=sr,
        input=True,
        frames_per_buffer=chunk,
        stream_callback=callback,
    )


def record(sr, gain, timeout=None):
    """
    Use PyAudio for live recording from the commandline.
    Converts callback-based API to queues to generators.
    """
    q = Queue()
    job_done = object()

    # Start in a new thread
    # proc = multiprocessing.Process(target=recorder_task, args=(q, sr, gain))
    # proc.start()
    thread = Thread(target=recorder_task, args=(q, sr, gain))
    thread.start()

    # Consumer
    block = False
    timeout = timeout or 0.01
    print("Recording...")
    while True:
        try:
            aud = q.get(block, timeout)
            print("got")
            if aud is job_done:
                break
            yield aud
        except: pass


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(
        description="Combine LibreASR with PyAudio for streaming ASR from the commandline."
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=44100,
        help="Sample Rate of your microphone (usually 44100 or 48000)",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "chunks"],
        default="all",
        type=str,
        nargs="+",
        help="transcription mode (stream implies chunks)",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=1e-4,
        help="Adjust gain / volume if it's not working",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to LibreASR config (optional)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="de",
        help="Change LibreASR language / model",
    )
    args = parser.parse_args()

    # fire up LibreASR
    os.chdir("/workspace")
    sys.path.append("/workspace")
    from libreasr import LibreASR

    def hook(conf):
        conf["model"]["loss"] = False
        conf["cuda"]["enable"] = False
        conf["cuda"]["device"] = "cpu"
        conf["model"]["load"] = True

        # load german model
        conf["model"]["path"] = {"n": "./models/de-4096.pth"}
        conf["tokenizer"]["model_file"] = "./tmp/tokenizer-de-4096.yttm-model"

    l = LibreASR(args.lang, config_path=args.config_path, config_hook=hook)
    l.load_inference()

    # start threads
    record(sr=args.sr, gain=args.gain)

    # retrieve & display transcripts
    generator = record(args.sr, args.gain)
    for i, transcript in enumerate(l.stream(generator, sr=args.sr)):
        print(transcript)
