import argparse
import random
import logging
import sys

import grpc
import torchaudio
import numpy as np

import libreasr.api.interfaces.libreasr_pb2 as ap
import libreasr.api.interfaces.libreasr_pb2_grpc as apg

DEMO_AUDIO = "./demo/3729-6852-0035.flac"
CHUNK_DURATION = 0.08  # secs
PORT = 50052


def load_demo():
    data, sr = torchaudio.load(DEMO_AUDIO)
    data = data[0][None]
    data = torchaudio.transforms.Resample(sr, 16000)(data)
    sr = 16000
    data = data.numpy().astype(np.float32).tobytes()
    return data, sr


def grab_audio():
    data, sr = load_demo()
    return ap.Audio(data=data, sr=sr)


def grab_audio_stream(secs):
    data, sr = load_demo()
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
    print("Transcribe:")
    audio = grab_audio()
    print("-", stub.Transcribe(audio).data)

    print("TranscribeStream:\n- ", end="")
    audio_stream = grab_audio_stream(secs=CHUNK_DURATION)
    for transcript in stub.TranscribeStream(audio_stream):
        print(transcript.data, end="")
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
        default=DEMO_AUDIO,
        help="if mode==file: what file to transcribe",
    )
    args = parser.parse_args()

    run(args)
