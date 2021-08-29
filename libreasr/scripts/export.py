#!/usr/bin/env python
# coding: utf-8

import argparse
import copy
from pathlib import Path
import os
import glob
import yaml
import shutil

import torch
from torch import nn
from torch.quantization import quantize_dynamic

from libreasr.lib.config import open_config, prepare_config
from libreasr.lib.utils import str2bool, update, dot


def select(l, what):
    l = list(l)

    print(f"Choose '{what}':")
    for i, x in enumerate(l):
        print(f" - [{i}] '{x}'")

    if len(l) == 0:
        chosen = None
    elif len(l) == 1:
        chosen = l[0]
    else:
        try:
            answer = input(f"<= {list(range(len(l))) + [None]}? ")
            if answer == "":
                chosen = None
            else:
                chosen = l[int(answer)]
        except:
            pass
    print(f"=> '{chosen}'\n")
    return chosen


def export(conf, tok, model, lm, i, o, model_prefix, model_strip_prefix, hook):
    print("Exporting...")

    # make sure dirs exist
    try:
        os.makedirs(o)
    except:
        pass

    # also store a meta file for bookkeeping
    meta = {
        "configs": {},
        "model": {},
        "lm": {},
    }

    # load config
    configs = [str(i / "base.yaml"), conf]
    conf = prepare_config(True, None, configs, hook)
    with open(o / "config.yaml", "w") as f:
        f.write(yaml.dump(conf))

    # note configs
    meta["configs"]["export"] = conf
    meta["configs"]["paths"] = [str(x) for x in configs]
    for k in configs:
        v = open_config(path=k)
        meta["configs"][k] = v

    # just copy tok
    t = o / "tokenizer.yttm-model"
    shutil.copy(tok, t)
    meta["tokenizer_path"] = str(t)

    # load state dict first
    exported = {}
    d = torch.load(model, map_location="cpu")
    keys = d.keys()

    # strip opt
    if "opt" in keys:
        del d["opt"]
        d = d["model"]

    # select with prefix
    for k, v in d.items():
        if k.startswith(model_prefix):
            if model_strip_prefix:
                k = k[len(model_prefix) :]
            exported[k] = v
    torch.save(exported, o / "model.pth")
    meta["model"]["state_dict_keys"] = list(exported.keys())

    # TODO do the same with lm
    meta["lm"]["state_dict_keys"] = []

    # save meta
    with open(o / "meta.yaml", "w") as f:
        f.write(yaml.dump(meta))


def detect_and_prompt_files(path):
    assert path.exists(), f"Path {path} does not exist..."

    # get all
    files = glob.glob(str(path / "*"))

    # select stuff
    conf = select(
        filter(lambda x: x.endswith(".yaml") and "base.yaml" not in x, files), "config"
    )
    tok = select(filter(lambda x: x.endswith(".yttm-model"), files), "tokenizer")
    model = select(filter(lambda x: x.endswith(".pth"), files), "model")
    lm = select(
        filter(lambda x: x.endswith(".pth") and "lm" in x, files), "language model"
    )
    return conf, tok, model, lm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=str,
        help="Path to files (like pretrained model and config)",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path to output files to",
    )
    parser.add_argument(
        "--model-prefix",
        type=str,
        default="student.",
        help="Only keep weights with this prefix in the PyTorch state dict",
    )
    parser.add_argument(
        "--model-strip-prefix",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Strip the prefix",
    )
    parser.add_argument(
        "--apply-default-hook",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Apply the default hook (NoisyStudent => Transducer) to config",
    )
    parser.add_argument(
        "--cpu",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Export for cpu execution",
    )
    args = parser.parse_args()

    # extract args
    i, o = Path(args.input), Path(args.output)

    # detect files
    items = detect_and_prompt_files(i)
    conf, tok, model, lm = items

    # adapt config
    def hook(conf):

        # training & noisy student
        conf["model"]["name"] = "Transducer"
        update(
            conf["model"], dot("training.noisystudent.overrides.student.model")(conf)
        )
        del conf["training"]

        # cpu execution
        if args.cpu:
            conf["model"]["loss"] = False
            conf["cuda"]["enable"] = False
            conf["cuda"]["device"] = "cpu"
            conf["model"]["load"] = True

        # paths
        conf["model"]["path"] = {"n": str(o / "model.pth")}
        conf["tokenizer"]["model_file"] = str(o / "tokenizer.yttm-model")

        # lm
        if lm is not None:
            conf["lm"]["enable"] = True
            ...

    if not args.apply_default_hook:
        hook = lambda x: None

    # export
    export(*items, i, o, args.model_prefix, args.model_strip_prefix, hook)
