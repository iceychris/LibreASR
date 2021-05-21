import yaml
import collections

from libreasr.lib.download import download_configs


def update(d, u):
    "from: https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth"
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def apply_overrides(conf, config_paths):
    try:
        for cp in config_paths:
            p = conf
            for one in cp:
                p = p[one]
            update(conf, p)
        return conf
    except:
        pass


def load_config(path, lang=None, do_overrides=True):
    # load config
    with open(path, "r") as stream:
        conf = yaml.safe_load(stream)

    # override config for inference + language
    if do_overrides:
        overrides = [["overrides", "inference"]]
        if lang is not None:
            overrides.append(["overrides", "languages", lang])
        conf = apply_overrides(conf, overrides)
    return conf


def assemble_models_df(paths):
    import pandas as pd

    df = None
    for path in paths:
        conf = load_config(path, do_overrides=False)
        info = conf["info"]
        if df is None:
            df = pd.DataFrame(columns=list(info.keys()))
        df = df.append(info, ignore_index=True)
    return df


def get_available_models():
    paths = download_configs()
    df = assemble_models_df(paths)
    return df
