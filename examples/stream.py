import argparse
import os
from threading import Thread
from queue import Queue
import sys
import itertools

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
        q.put(arr)
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
    thread = Thread(target=recorder_task, args=(q, sr, gain))
    thread.start()

    # Consumer
    print("Recording...")
    while True:
        aud = q.get(True, timeout)
        if aud is job_done:
            break
        yield aud


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

    l = LibreASR(args.lang, config_path=args.config_path)
    l.load_inference()

    # start threads
    record(sr=args.sr, gain=args.gain)

    # retrieve & display transcripts
    generator = record(args.sr, args.gain)
    for i, (diff, y_all) in enumerate(l.stream(generator, sr=args.sr)):
        print(diff, end="")
        if (i + 1) % 10 == 0:
            print()
        sys.stdout.flush()
