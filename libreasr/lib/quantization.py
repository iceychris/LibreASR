import torch
import torch.nn as nn


def try_quantize(model, fn, debug=True):
    name = model.__class__.__name__
    new_model = None
    try:
        new_model = fn(model)
        if debug:
            print(f"[quantization] quantizing {name} done.")
    except Exception as e:
        new_model = model
        if debug:
            print(
                f"[quantization] post-quantizing {name} failed. Might lead to degraded model performance."
            )
            print(e)
    return new_model


def quantize(
    m,
    pre=lambda m: m.eval(),
    post=lambda m: m.eval(),
    mods={torch.nn.LSTM, torch.nn.Linear},
):
    print(f"[quantization] engine: {torch.backends.quantized.engine}")
    m = pre(m)
    m = torch.quantization.quantize_dynamic(
        m,  # the original model
        mods,  # a set of layers to dynamically quantize
        dtype=torch.qint8,
    )
    m = post(m)
    return m


def quantize_model(m, **kwargs):
    # prepare
    lang, lm = None, None

    def pre(m):
        nonlocal lang, lm
        lang, lm = getattr(m, "lang", None), getattr(m, "lm", None)
        m.lang = None
        m.lm = None
        m = m.cpu()
        m = m.eval()
        return m

    # post restore
    def post(m):
        nonlocal lang, lm
        m.lang = lang
        m.lm = lm
        m = m.eval()
        return m

    # quantize
    m = quantize(m, pre, post, **kwargs)
    m.quantization_fix()
    return m


def quantize_lm(m, **kwargs):
    if hasattr(m, "lm"):
        m.lm = quantize(m.lm, **kwargs)
        m.lm.quantization_fix()
    else:
        m = quantize(m, **kwargs)
        m.quantization_fix()
    return m


def save_quantized_model(m, paths):
    m = m.eval()
    lang = None
    if hasattr(m, "lang"):
        lang = m.lang
        m.lang = None
    torch.save(m.encoder, paths[0])
    torch.save(m.predictor, paths[1])
    torch.save(m.joint, paths[2])
    if lang is not None:
        m.lang = lang


def save_quantized_lm(m, path):
    m = m.eval()
    if hasattr(m, "lm"):
        torch.save(m.lm, path)
    else:
        torch.save(m, path)
    return m


def save_quantized(m, paths_model, path_lm):
    # first, quantize rnnt
    m = quantize_model(m)

    # then lm
    m = quantize_lm(m)

    # save
    save_quantized_model(m, paths_model)
    save_quantized_lm(m, path_lm)


def load_quantized_model(m, lang, paths):
    m = m.cpu()

    # first quantize
    m = quantize_model(m)

    # extract paths
    pe, pp, pj = paths

    # load
    kwargs = {"map_location": "cpu"}
    m.encoder = torch.load(pe, **kwargs)
    m.predictor = torch.load(pp, **kwargs)
    m.joint = torch.load(pj, **kwargs)

    # patch class (to fix eval, train and to)
    m.quantization_fix()

    # set to eval mode
    m = m.eval()
    # print("c", m)

    # debug
    print("[quantization] load_quantized_model(...) done.")

    return m


def load_quantized_lm(lm, path):
    lm = lm.cpu()
    # first quantize
    lm = quantize_lm(lm)
    kwargs = {"map_location": "cpu"}
    lm = torch.load(path, **kwargs)
    lm.quantization_fix()
    lm = lm.eval()
    print("[quantization] load_quantized_lm(...) done.")
    return lm
