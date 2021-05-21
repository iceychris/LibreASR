import argparse
import random
import logging
import sys

import grpc
import torchaudio
import numpy as np

import libreasr.api.interfaces.libreasr_pb2 as ap
import libreasr.api.interfaces.libreasr_pb2_grpc as apg
from libreasr.lib.defaults import AUDIOS, DEFAULT_STREAM_CHUNK_SZ

PORT = 50051


def load_demo(f):
    data, sr = torchaudio.load(f)
    data = data[0][None]
    data = torchaudio.transforms.Resample(sr, 16000)(data)
    sr = 16000
    data = data.numpy().astype(np.float32).tobytes()
    return data, sr


def grab_audio(f):
    data, sr = load_demo(f)
    return ap.Audio(data=data, sr=sr)


def grab_audio_stream(f, secs):
    data, sr = load_demo(f)
    slice_sz = int(secs * sr) * 4
    l = len(data) // slice_sz
    # [start] zero
    yield ap.Audio(data=bytes([0] * slice_sz), sr=sr)
    for i in range(l):
        chunk = data[i * slice_sz : (i + 1) * slice_sz]
        # pad with zeros
        chunk = chunk + bytes([0] * (slice_sz - len(chunk)))
        assert len(chunk) % 4 == 0
        # [mid]
        yield ap.Audio(data=chunk, sr=sr)
    # [end] zero frames mark end
    for _ in range(10):
        yield ap.Audio(data=bytes([0] * slice_sz), sr=sr)


def test_asr(stub):
    for f, l in AUDIOS:
        print("Transcribe:")
        audio = grab_audio(f)
        print("-", stub.Transcribe(audio).data)

    for f, l in AUDIOS:
        print("TranscribeStream:\n- ", end="")
        audio_stream = grab_audio_stream(f, secs=DEFAULT_STREAM_CHUNK_SZ)
        for transcript in stub.TranscribeStream(audio_stream):
            print("-", transcript.data, end="\r")
            sys.stdout.flush()
        print()


def run(args):
    with grpc.insecure_channel(f"localhost:{PORT}") as channel:
        stub = apg.LibreASRStub(channel)
        test_asr(stub)


if __name__ == "__main__":
    logging.basicConfig()

    # parse args
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--source",
        choices=["file", "stream"],
        type=str,
        default="file",
        help="audio source",
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
        "--file",
        type=str,
        default="",
        help="if mode==file: what file to transcribe",
    )
    args = parser.parse_args()

    run(args)
