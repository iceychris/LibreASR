import argparse
import copy
from pathlib import Path
from functools import partial
import os
import time
import sys

import numpy as np
import torchaudio
import pandas as pd
import ray
from ray import tune

import torch
from torch import nn

from libreasr import LibreASR
from libreasr.lib.metrics import cer, wer
from libreasr.lib.utils import sanitize_str, str2bool


def tune_everything(
    config,
    factory_fn,
    dataset,
    fraction=1.0,
    chdir="",
    do_print=False,
    sanity_check=False,
):

    # chdir (if necessary)
    if chdir != "":
        os.chdir(chdir)

    # load dataset
    df = pd.read_csv(dataset)
    if fraction < 1.0:
        df = df.sample(frac=fraction, random_state=42)
        print(f"Tuning LM on {len(df)} samples.")

    # load inference instance
    li = factory_fn()

    # maybe do sanity checks
    if sanity_check:

        # test beam search
        res = li.transcribe(LibreASR.example(), batch=False, **config)
        print("[sanity-check] [ASR] Transcribe:", res)

        # test lm generation
        res = li.model.lm.generate(
            "the other day when we visited san", lang=li.model.lang, steps=64
        )
        print("[sanity-check] [LM] Generation :", res)
        return

    # helper
    def fx(a):
        return int((a / 1000.0) * 16000)

    # run
    wers = []
    times = []
    for i, row in df.iterrows():
        f = row["file"]
        l = sanitize_str(row["label"])
        xstart = fx(row["xstart"])
        xlen = fx(row["xlen"])

        # load
        aud, sr = torchaudio.load(f, frame_offset=xstart, num_frames=xlen)
        assert sr == 16000

        # infer
        t1 = time.perf_counter()
        p = li.transcribe(aud, batch=False, **config)
        t2 = time.perf_counter()

        # metrics
        _cer = cer(p, l)
        _wer = wer(p, l)
        t = t2 - t1
        wers.append(_wer)
        times.append(t)

        if do_print:
            print("file :", f)
            print("sr   :", sr)
            print("dur  :", aud.size(1) / sr, "sec")
            print("label:", l)
            print("pred :", p)
            print("cer  :", _cer)
            print("wer  :", _wer)
            from IPython.display import Audio
            from IPython.core.display import display, HTML

            display(Audio(aud, rate=sr))
            print()

    # report
    wers = np.array(wers)
    times = np.array(times)
    return {
        "mean_wer": wers.mean(),
        "std_wer": wers.std(),
        "mean_t": times.mean(),
        "std_t": times.std(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", type=str, help="Which model_id to load.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="/data/stt/data/common-voice/en/asr-dataset-valid.csv",
        help="Dataset to use for tuning (csv).",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.05,
        help="Fraction of dataset to use (0.0 to 1.0).",
    )
    parser.add_argument(
        "--lm-path",
        type=str,
        default="./lms/lm-en-4096.pth",
        help="Path to language model.",
    )
    parser.add_argument(
        "--chdir",
        type=str,
        default="",
        help="Switch to this working directory.",
    )
    parser.add_argument(
        "--sanity-check",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Just do a sanity check of the model.",
    )
    args = parser.parse_args()

    def fn():
        def hook(c):
            c["lm"] = {
                "enable": True,
                "load": True,
                "path": {
                    "n": args.lm_path,
                },
                # just default values...
                "vocab_sz": c["model"]["vocab_sz"],
                "embed_sz": 768,
                "hidden_sz": 768,
                "num_layers": 4,
                "p": 0.3,
            }
            c["cuda"]["enable"] = True
            c["cuda"]["device"] = "cuda:0"

        l = LibreASR(args.model_id, config_hook=hook)
        return l

    # preload function
    fit = partial(
        tune_everything,
        factory_fn=fn,
        dataset=args.dataset,
        fraction=args.fraction,
        chdir=args.chdir,
    )

    # space to tune over
    cfg = {
        "alpha": tune.sample_from(lambda spec: np.random.uniform(0.0, 0.6)),
        "beam_search_opts": {
            "implementation": "speechbrain",
            "beam_width": tune.choice([2, 3, 4, 5, 6, 7, 8, 12]),
            "state_beam": tune.sample_from(lambda spec: np.random.uniform(0.0, 2.5)),
            "expand_beam": tune.sample_from(lambda spec: np.random.uniform(0.0, 2.5)),
        },
    }

    # maybe do sanity check
    if args.sanity_check:
        fit({}, sanity_check=True)
        sys.exit(0)

    # start ray
    cpus, gpus = 4, 1
    ray.init(num_cpus=cpus, ignore_reinit_error=True)

    analysis = tune.run(
        fit,
        name=f"libreasr-tune-{args.model_id}",
        num_samples=1000,
        metric="mean_wer",
        mode="min",
        stop={"training_iteration": 10},
        config=cfg,
        resources_per_trial={"cpu": cpus, "gpu": gpus},
        # scheduler=bohb_hyperband,
        # search_alg=bohb_search,
        fail_fast=False,
    )
    print("Best hyperparameters found were: ", analysis.best_config)
