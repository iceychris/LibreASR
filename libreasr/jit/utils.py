import torch
from torch import nn
from torch.quantization import quantize_dynamic


def jittify_preprocessor(m, inplace=False):
    p = m.preprocessor
    p.to_jit()
    scripted = torch.jit.script(p)
    if inplace:
        m.preprocessor = scripted
    return scripted


def jittify_encoder(m, inplace=False):
    e = m.encoder
    e.to_jit()
    scripted = torch.jit.script(e)
    if inplace:
        m.encoder = scripted
    return scripted


def jittify_predictor(m, inplace=False):
    p = m.predictor
    p.to_jit()
    scripted = torch.jit.script(p)
    if inplace:
        m.predictor = scripted
    return scripted


def jittify_joint(m, inplace=False):
    j = m.joint
    j.to_jit()
    scripted = torch.jit.script(j)
    if inplace:
        m.joint = scripted
    return scripted


def jit_model(m):
    jittify_preprocessor(m, inplace=True)
    jittify_encoder(m, inplace=True)
    jittify_predictor(m, inplace=True)
    jittify_joint(m, inplace=True)


def quantize(m):
    return quantize_dynamic(m, {nn.LSTM, nn.Linear}, dtype=torch.qint8)


def try_quantize(model, fn, debug=True):
    name = model.__class__.__name__
    new_model = None
    try:
        new_model = fn(model)
        if debug:
            print(f"[quantization] post-quantizing {name} done.")
    except Exception as e:
        new_model = model
        if debug:
            print(
                f"[quantization] post-quantizing {name} failed. Might lead to degraded model performance."
            )
            print(e)
    return new_model


def quantize_model_safely(m, debug=True):
    def fn(m):
        prepro, e, p, j = m.preprocessor, m.encoder, m.predictor, m.joint
        m.preprocessor = quantize(prepro)
        m.encoder = quantize(e)
        m.predictor = quantize(p)
        m.joint = quantize(j)

    try_quantize(m, fn, debug=debug)
