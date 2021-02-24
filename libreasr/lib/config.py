import os
from functools import partial
from importlib import import_module
import collections.abc
from pathlib import Path

import torch
import yaml

from libreasr.lib.utils import n_params, what, wrap_transform
from libreasr.lib.language import get_language
from libreasr.lib.builder import ASRDatabunchBuilder
from libreasr.lib.data import ASRDatabunch
from libreasr.lib.models import get_model
from libreasr.lib.learner import ASRLearner
from libreasr.lib.lm import load_lm
from libreasr.lib.download import download_all


def update(d, u):
    "from: https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth"
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def open_config(*args, path="./config/testing.yaml", **kwargs):
    path = Path(path)
    if not os.path.exists(path):
        path = ".." / path
    with open(path, "r") as stream:
        try:
            obj = yaml.safe_load(stream)
            return obj
        except yaml.YAMLError as exc:
            print(exc)


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
    if conf["cuda"]["enable"] and torch.cuda.is_available():
        if torch.cuda.is_available():
            torch.cuda.set_device(int(conf["cuda"]["device"].split(":")[1]))
            torch.backends.cudnn.benchmark = conf["cuda"]["benchmark"]
        else:
            raise Exception("cuda not available")


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


def apply_overrides(conf, config_paths):
    for cp in config_paths:
        p = conf
        for one in cp:
            p = p[one]
        update(conf, p)
    return conf


def parse_and_apply_config(
    *args, inference=False, lang="", path=None, config_hook=lambda x: None, **kwargs
):

    # download pretrained models
    if inference and path is None:
        lang, mcp = download_all(lang)
        if mcp is not None:
            conf = open_config(*args, path=mcp, **kwargs)
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
    conf = apply_overrides(conf, overrides)

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
    model_qpre = conf.get("quantization", {}).get("model", {}).get("pre", False)
    model_qpost = conf.get("quantization", {}).get("model", {}).get("post", False)
    model_path_to_load = "q" if model_qpre else "n"
    model_path = conf.get("model", {}).get("path", {}).get(model_path_to_load, "")
    if isinstance(model_path, list):
        model_path = [os.path.expanduser(x) for x in model_path]
    else:
        model_path = os.path.expanduser(model_path)
    model_args = (model_qpre, model_qpost, model_path)
    model_do_load = conf["model"].get("load", False)
    lm_qpre = conf.get("quantization", {}).get("lm", {}).get("pre", False)
    lm_qpost = conf.get("quantization", {}).get("lm", {}).get("post", False)
    lm_path_to_load = "q" if lm_qpre else "n"
    lm_path = os.path.expanduser(
        conf.get("lm", {}).get("path", {}).get(lm_path_to_load, "")
    )
    lm_args = (lm_qpre, lm_qpost, lm_path)
    lm_enable = conf["lm"].get("enable", False)

    # load lm
    lm = None
    dev = conf["cuda"]["device"]
    if lm_enable:
        try:
            lm = load_lm(
                conf.get("lm", {}),
                *lm_args,
                load=conf["lm"].get("load", False),
                device=dev,
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
        m.eval()

        return conf, lang, m, tfms
    else:
        # grab learner
        learn = ASRLearner.from_config(conf, db, m)
        learn.lang = lang

        return conf, lang, builder_train, builder_valid, db, m, learn
