import os
from functools import partial
from importlib import import_module
import collections.abc
from pathlib import Path

import torch
import yaml

from libreasr.lib.utils import n_params, what, wrap_transform, update
from libreasr.lib.language import get_language
from libreasr.lib.builder import ASRDatabunchBuilder
from libreasr.lib.data import ASRDatabunch
from libreasr.lib.models import get_model
from libreasr.lib.learner import LibreASRLearner
from libreasr.lib.lm import load_lm
from libreasr.lib.download import download_all
from libreasr.lib.utils import warn
from libreasr.lib.defaults import DEFAULT_STREAM_TRANSFORMS


def open_config(*args, path="./config/testing.yaml", **kwargs):
    # load
    objs = []
    conf = {}
    if not isinstance(path, (list, tuple)):
        path = [path]
    for p in path:
        p = Path(p)
        if not os.path.exists(p):
            p = ".." / p
        with open(p, "r") as stream:
            try:
                obj = yaml.safe_load(stream)
                objs.append(obj)
            except yaml.YAMLError as exc:
                print(exc)

    # merge
    for obj in objs:
        update(conf, obj)
    return conf


def parse_transforms(conf, inference):
    mod = import_module("libreasr.lib.transforms")
    tfms = []
    if inference:
        conf_tfms = [
            conf["transforms"]["x"],
            conf["transforms"]["stream"],
            conf["transforms"]["y"],
        ]
    else:
        conf_tfms = [conf["transforms"]["x"], conf["transforms"]["y"]]
    for i, conf_one_var in enumerate(conf_tfms):
        tfms_one_var = []
        for j, conf_one_tfm in enumerate(conf_one_var):
            args = conf_one_tfm.get("args", {})
            is_partial = conf_one_tfm.get("partial", False)
            is_wrap = conf_one_tfm.get("wrap", False)
            func = getattr(mod, conf_one_tfm["name"])
            if is_partial:
                func = partial(func, **args)
            if is_wrap:
                func = wrap_transform(func)
            tfms_one_var.append(func)
        tfms.append(tfms_one_var)
    return tfms


def apply_cuda_stuff(conf):
    ddp = conf.get("training", {}).get("ddp", {}).get("enable", False)

    # set correct gpu
    if conf["cuda"]["enable"]:
        if torch.cuda.is_available():
            if not ddp:
                torch.cuda.set_device(int(conf["cuda"]["device"].split(":")[1]))
            torch.backends.cudnn.benchmark = conf["cuda"]["benchmark"]
        else:
            warn("cuda enabled in config but not available...")
            warn("... switching to cpu execution")
            conf["cuda"]["enable"] = False
            conf["cuda"]["device"] = "cpu"


def check_vocab_sz(conf):
    a = conf["model"]["vocab_sz"]
    b = conf["wanted_vocab_sz"]
    if a != b:
        raise Exception(f"vocab sizes don't match: wanted={b}, current={a}")


def check_db(db):
    tpl = db.one_batch()
    print(tpl)
    X, Ym, _, _ = tpl[0]
    Y, Y_lens, X_lens = tpl[1]
    what(X), what(X_lens), what(Y), what(Y_lens)
    assert X_lens.size(0) == Y_lens.size(0)


def apply_overrides(conf, config_paths, silent=False):
    try:
        for cp in config_paths:
            p = conf
            for one in cp:
                p = p[one]
            update(conf, p)
    except Exception as e:
        if silent:
            return conf
        raise e
    return conf


def fix_config(conf):
    # choose transforms
    if "x" not in list(conf["transforms"].keys()):
        if conf["model"]["learnable_stft"]:
            conf["transforms"]["x"] = conf["transforms"]["x-stft"]
        else:
            conf["transforms"]["x"] = conf["transforms"]["x-no-stft"]


def fix_config_inference(conf, model_name, base="~/.cache/LibreASR"):
    # check if tokenizer path is correct
    p = f"{base}/{model_name}/tokenizer.yttm-model"
    if not "tokenizer" in conf:
        conf["tokenizer"] = {}
    conf["tokenizer"]["model_file"] = p

    # make sure the model path is correct
    conf["model"]["load"] = True
    p = f"{base}/{model_name}/model.pth"
    update(conf["model"], {"path": {"n": p}})


def fix_transforms(conf, inference=False):
    if inference:
        # stream transforms
        conf["transforms"]["x"] = conf["transforms"]["x"][1:]
        if "stream" not in list(conf["transforms"].keys()):
            warn("no stream transforms defined...")
            warn("... using defaults")
            conf["transforms"]["stream"] = DEFAULT_STREAM_TRANSFORMS

        # transcribe transforms
        l = conf["transforms"]["x"]
        for t in l:
            if t["name"] == "PadderCutter":
                # remove
                l.remove(t)


def parse_and_apply_config(
    *args, inference=False, lang="", path=None, config_hook=lambda x: None, **kwargs
):

    # download pretrained models
    if inference and path is None:
        lang, release, config_path = download_all(lang)
        if config_path is not None:
            conf = open_config(*args, path=config_path, **kwargs)
            fix_config_inference(conf, release)
    else:
        # open config
        conf = open_config(*args, path=path, **kwargs)

    # override config for inference + language
    overrides = []
    if inference:
        overrides.append(["overrides", "inference"])
    lang_name = lang
    if lang is not None and len(lang) > 0:
        overrides.append(["overrides", "languages", lang])
    conf = apply_overrides(conf, overrides, silent=True)

    # special config fixes...
    fix_config(conf)

    # fix transforms for inference
    fix_transforms(conf, inference=inference)

    # apply hook
    config_hook(conf)

    # torch-specific cuda settings
    apply_cuda_stuff(conf)

    # grab transforms
    tfms = parse_transforms(conf, inference=inference)

    if not inference:
        # grab builder
        builder_train = ASRDatabunchBuilder.from_config(conf, mode="train")
        builder_valid = ASRDatabunchBuilder.from_config(conf, mode="valid")

    # quantization settings
    if inference:
        torch.backends.quantized.engine = conf["quantization"]["engine"]

    # grab language + sanity check
    tok_path = os.path.expanduser(conf["tokenizer"]["model_file"])
    try:
        lang, _ = get_language(model_file=tok_path)
    except:
        builder_train.train_tokenizer(
            model_file=tok_path,
            vocab_sz=conf["model"]["vocab_sz"],
        )
        lang, _ = get_language(model_file=tok_path)
    check_vocab_sz(conf)

    if not inference:
        # grab databunch + sanity check
        db = ASRDatabunch.from_config(conf, lang, builder_train, builder_valid, tfms)
        # check_db(db)

    # grab params
    model_path_to_load = "n"
    model_path = conf.get("model", {}).get("path", {}).get(model_path_to_load, "")
    if isinstance(model_path, list):
        model_path = [os.path.expanduser(x) for x in model_path]
    else:
        model_path = os.path.expanduser(model_path)
    model_args = (model_path,)
    model_do_load = conf["model"].get("load", False)
    lm_path_to_load = "n"
    lm_path = os.path.expanduser(
        conf.get("lm", {}).get("path", {}).get(lm_path_to_load, "")
    )
    lm_args = (lm_path,)
    lm_enable = conf["lm"].get("enable", False)
    device = conf["cuda"]["device"]

    # post quantization
    do_quantization = conf.get("quantization", {}).get("enable", True)

    # load lm
    lm = None
    if lm_enable:
        try:
            lm = load_lm(
                conf.get("lm", {}),
                *lm_args,
                load=conf["lm"].get("load", False),
                device=device,
            )
        except Exception as e:
            print("[lm] Failed to load")
            print(e)

    # grab model instance
    m = get_model(conf, lang)

    if model_do_load:
        # load weights
        from libreasr.lib.model_utils import load_asr_model

        m = load_asr_model(
            m,
            lang_name,
            lang,
            *model_args,
            conf["cuda"]["device"],
            lm=lm,
        )
        m.lm = lm
        m.lang = lang

    if inference:
        # eval mode
        #  and move model to device
        m = m.to(device).eval()

        # quantize
        if do_quantization:
            from libreasr.jit.utils import quantize_model_safely

            quantize_model_safely(m)

        return conf, lang, m, tfms
    else:
        # grab learner
        learn = LibreASRLearner.from_config(conf, db, m)
        learn.lang = lang

        return conf, lang, builder_train, builder_valid, db, m, learn
