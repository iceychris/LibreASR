import torch


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
