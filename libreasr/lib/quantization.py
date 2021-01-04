import torch
import torch.nn as nn

DEFAULT_PATHS = [
    "./tmp/encoder.pth",
    "./tmp/predictor.pth",
    "./tmp/joint.pth",
]


def maybe_post_quantize(model, debug=True):
    name = model.__class__.__name__
    try:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
        )
        if debug:
            print(f"[quantization] post-quantizing {name} done.")
    except:
        if debug:
            print(
                f"[quantization] post-quantizing {name} failed. Might lead to degraded model performance."
            )
    return model


def quantize_rnnt(m, mods, backend='qnnpack'):

    # prepare
    torch.backends.quantized.engine = backend
    m = m.cpu()
    m = m.eval()
    lang, lm = getattr(m, 'lang', None),  getattr(m, 'lm', None)
    m.lang = None
    m.lm = None

    # quantize
    m = torch.quantization.quantize_dynamic(
        m,  # the original model
        mods, # a set of layers to dynamically quantize
        dtype=torch.qint8)

    # post restore
    m.lang = lang
    m.lm = lm

    return m


def save_quantized(m, paths=DEFAULT_PATHS):
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


def load_quantized(m, lang, paths=DEFAULT_PATHS):
    m = m.cpu()
    kwargs = {"map_location": "cpu"}

    # extract paths
    pe, pp, pj = paths

    # load
    m.encoder = torch.load(pe, **kwargs)
    m.predictor = torch.load(pp, **kwargs)
    m.joint = torch.load(pj, **kwargs)

    # patch class (to fix eval, train and to)
    m.post_quantize()

    # set to eval mode
    m = m.eval()

    # debug
    print("[quantization] load_quantized(...) done")

    return m
