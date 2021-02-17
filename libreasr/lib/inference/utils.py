import yaml
import collections


def update(d, u):
    "from: https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth"
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def apply_overrides(conf, config_paths):
    for cp in config_paths:
        p = conf
        for one in cp:
            p = p[one]
        update(conf, p)
    return conf


def load_config(path, lang=None):
    # load config
    with open(path, "r") as stream:
        conf = yaml.safe_load(stream)

    # override config for inference + language
    overrides = [["overrides", "inference"]]
    if lang is not None:
        overrides.append(["overrides", "languages", lang])
    conf = apply_overrides(conf, overrides)
    return conf
