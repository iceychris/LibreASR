import os
from functools import partial
from importlib import import_module
import collections.abc
from pathlib import Path

import torch
import yaml

from .utils import n_params, what, wrap_transform
from .language import get_language
from .builder import ASRDatabunchBuilder
from .data import ASRDatabunch
from .models import Transducer, CTCModel
from .learner import ASRLearner
from .lm import load_lm


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
    mod = import_module("lib.transforms")
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
    if conf["cuda"]["enable"]:
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
    X, Ym, _, _ = tpl[0]
    Y, Y_lens, X_lens = tpl[1]
    what(X), what(X_lens), what(Y), what(Y_lens)
    assert X_lens.size(0) == Y_lens.size(0)


def parse_and_apply_config(*args, inference=False, **kwargs):

    # open config
    conf = open_config(*args, **kwargs)

    # override config for inference + language
    overrides = []
    if inference:
        overrides.append("inference")
    lang = kwargs.get("lang", "")
    lang_name = lang
    if len(lang) > 0:
        overrides.append(lang)
    for override in overrides:
        update(conf, conf["overrides"][override])

    # torch-specific cuda settings
    apply_cuda_stuff(conf)

    # grab transforms
    tfms = parse_transforms(conf, inference=inference)

    if not inference:
        # grab builder
        builder_train = ASRDatabunchBuilder.from_config(conf, mode="train")
        builder_valid = ASRDatabunchBuilder.from_config(conf, mode="valid")

    # grab language + sanity check
    try:
        lang, _ = get_language(model_file=conf["tokenizer"]["model_file"])
    except:
        builder_train.train_tokenizer(
            model_file=conf["tokenizer"]["model_file"],
            vocab_sz=conf["model"]["vocab_sz"],
        )
        lang, _ = get_language(model_file=conf["tokenizer"]["model_file"])
    check_vocab_sz(conf)

    if not inference:
        # grab databunch + sanity check
        db = ASRDatabunch.from_config(conf, lang, builder_train, builder_valid, tfms)
        check_db(db)

    # load lm
    lm = None
    if inference and conf["lm"]["enable"]:
        try:
            lm = load_lm(conf, lang_name)
            print("LM: loaded.")
        except:
            print("LM: Failed to load.")

    # grab model
    m = Transducer.from_config(conf, lang)
    # print(n_params(m))

    if inference:
        # load weights
        from .model_utils import load_asr_model

        load_asr_model(m, lang_name, lang, conf["cuda"]["device"], lm=lm)
        m.lm = lm
        m.lang = lang

        # eval mode
        m.eval()

        return conf, lang, m, tfms
    else:
        # grab learner
        learn = ASRLearner.from_config(conf, db, m)

        return conf, lang, builder_train, builder_valid, db, m, learn
