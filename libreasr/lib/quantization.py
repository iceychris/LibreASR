import torch
import torch.nn as nn

from libreasr.lib.models import Transducer


class QuantizedTransducer(Transducer):
    def eval(self):
        try_eval(self)
        return self

    def train(self):
        return self

    def to(self, _):
        return self


def try_eval(m):
    for c in m.children():
        if isinstance(c, torch.nn.quantized.modules.linear.LinearPackedParams):
            continue
        try:
            c.eval()
        except:
            pass
        try_eval(c)


def save_quantized(m):
    m = m.eval()
    lang = None
    if hasattr(m, "lang"):
        lang = m.lang
        del m.lang
    torch.save(m.encoder, "t-e.pth")
    torch.save(m.predictor, "t-p.pth")
    torch.save(m.joint, "t-j.pth")
    if lang is not None:
        m.lang = lang


def load_quantized(m, conf, lang):
    m = m.cpu()
    kwargs = {"map_location": "cpu"}

    # extract paths
    pe, pp, pj = conf["model"]["path"]

    # load
    m.encoder = torch.load(pe, **kwargs)
    m.predictor = torch.load(pp, **kwargs)
    m.joint = torch.load(pj, **kwargs)

    # set to eval mode
    try_eval(m)

    # patch class (to fix eval, train and to)
    m.__class__ = QuantizedTransducer

    return m
