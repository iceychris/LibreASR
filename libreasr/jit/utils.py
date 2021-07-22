import torch


def jittify_preprocessor(m):
    p = m.preprocessor
    p.to_jit()
    return torch.jit.script(p)


def jittify_encoder(m):
    e = m.encoder
    e.to_jit()
    return torch.jit.script(e)


def jittify_predictor(m):
    p = m.predictor
    p.to_jit()
    return torch.jit.script(p)


def jittify_joint(m):
    j = m.joint
    j.to_jit()
    return torch.jit.script(j)
