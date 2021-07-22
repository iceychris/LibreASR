#!/usr/bin/env python
# coding: utf-8

import argparse
import copy
from pathlib import Path

import torch
from torch import nn
from torch.quantization import quantize_dynamic

from libreasr import LibreASR
from libreasr.jit import (
    jittify_preprocessor,
    jittify_encoder,
    jittify_predictor,
    jittify_joint,
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def quantize(m):
    return quantize_dynamic(m, {nn.LSTM, nn.Linear}, dtype=torch.qint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_id", type=str, help="Which model_id to load & convert to TorchScript."
    )
    parser.add_argument(
        "--quantize",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Quantize model?",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./cli/models",
        help="Where to save the resulting model files.",
    )
    args = parser.parse_args()

    # extract args
    base = args.path
    do_quantization = args.quantize

    # create instance
    l = LibreASR(args.model_id)

    # check if instance is working
    # l.stream("./assets/samples/common_voice_en_22738408.wav")

    # quantization
    m = l.model
    prepro, e, p, j = m.preprocessor, m.encoder, m.predictor, m.joint
    if do_quantization:
        m.preprocessor = quantize(prepro)
        m.encoder = quantize(e)
        m.predictor = quantize(p)
        m.joint = quantize(j)

    # jit
    prepro, e, p, j = m.preprocessor, m.encoder, m.predictor, m.joint

    ###
    # Preprocessor
    ###

    fname = str(Path(base) / "preprocessor.pth")
    preprocessor = copy.deepcopy(prepro)
    preprocessor_jit = jittify_preprocessor(m)

    x = torch.randn(1, 16000, 1, 1)
    xl = torch.LongTensor([16000])

    y1, yl1 = preprocessor(x, xl, inference=True)
    y2, yl2 = preprocessor_jit(x, xl, inference=True)

    # tests: outputs and states must be equal
    assert (y1 == y2).all().item()
    assert (yl1 == yl2).all().item()

    # save & load
    torch.jit.save(preprocessor_jit, fname)
    preprocessor2 = torch.jit.load(fname)

    ###
    # Encoder
    ###

    fname = str(Path(base) / "encoder.pth")
    enc = copy.deepcopy(e)
    enc_jit = jittify_encoder(m)

    x = torch.randn(1, 21, 768)
    xl = torch.LongTensor([21])
    s = m.encoder.initial_state()

    y1, so1 = enc(x, state=s, return_state=True)
    y2, so2 = enc_jit(x, xl, s)
    y3, _ = enc(x, return_state=True)
    y4, _ = enc_jit(x, xl, None)
    y5, _ = enc(x, state=so2, return_state=True)
    y6, _ = enc_jit(x, xl, so2)

    # tests: outputs and states must be equal
    assert (y1 == y2).all().item()
    for m1, m2 in zip(so1, so2):
        for l in range(len(m1)):
            assert ((m1[l] == m2[l]).all()).item()
    assert (y3 == y4).all().item()
    assert (y5 == y6).all().item()

    # save & load
    torch.jit.save(enc_jit, fname)
    enc2 = torch.jit.load(fname)

    ###
    # Predictor
    ###

    fname = str(Path(base) / "predictor.pth")
    pre = copy.deepcopy(p)
    pre_jit = jittify_predictor(m)

    x = torch.LongTensor([1, 2, 3, 4])[None]
    xl = torch.LongTensor([4])
    s = m.predictor.initial_state()

    y1, so1 = pre(x)
    y2, so2 = pre_jit(x, xl, None)
    y3, _ = pre(x, state=so2)
    y4, _ = pre_jit(x, xl, state=so1)

    # tests: outputs and states must be equal
    assert (y1 == y2).all().item()
    for m1, m2 in zip(so1, so2):
        for l in range(len(m1)):
            assert ((m1[l] == m2[l]).all()).item()
    assert (y3 == y4).all().item()

    # save & load
    torch.jit.save(pre_jit, fname)
    pre2 = torch.jit.load(fname)

    ###
    # Joint
    ###

    fname = str(Path(base) / "joint.pth")
    joint = copy.deepcopy(j)
    joint_jit = jittify_joint(m)

    a = torch.randn(1, 3, 384)
    b = torch.randn(1, 3, 1024)

    y1 = joint(a, b, softmax=True, log=True)
    y2 = joint(a, b, softmax=True, log=True)

    # tests: outputs and states must be equal
    assert (y1 == y2).all().item()

    # save & load
    torch.jit.save(joint_jit, fname)
    joint2 = torch.jit.load(fname)

    # done.
    print("> 4 TorchScript modules exported.")
