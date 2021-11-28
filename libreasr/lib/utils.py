import argparse
import copy
import traceback
from collections import OrderedDict
from collections.abc import Mapping
import re
import sys
from typing import Generator

import matplotlib.pyplot as plt

import torch
import numpy as np

LANGS = "de,en,fr,sv,es,eo,it,nl".split(",")
LANG_TO_IDX = {l: i for i, l in enumerate(LANGS)}


def warn(msg, hard=False):
    m = f"[warning] {msg}"
    mn = m + "\n"
    print(m)
    if hard:
        print(m, file=sys.stderr)
        try:
            sys.__stdout__.write(mn)
            sys.__stderr__.write(mn)
            sys.__stdout__.flush()
            sys.__stderr__.flush()
        except:
            pass


def update(d, u, deepcpy=False):
    "from: https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth"
    if deepcpy:
        d = copy.deepcopy(d)
    for k, v in u.items():
        try:
            if isinstance(v, Mapping):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        except Exception as e:
            print(d, u, k, v)
            print(e)
            raise e
    return d


def dot(s):
    def inner(d):
        nonlocal s
        s = s.split(".")
        res = d
        for x in s:
            res = res[x]
        return res

    return inner


def defaults(d, default_val):
    return update(default_val.copy(), d)


class Text(str):
    def show(self, **kwargs):
        print(self)


def nearest_multiple(x, base, identity_if_divisible=True):
    if identity_if_divisible and x % base == 0:
        return x
    return base * round(x / base) - base


def stats(t):
    return t.mean(), t.std()


def n_params(m):
    """
    Get the number of parameters
    for a PyTorch nn.Module m
    """
    return sum(p.numel() for p in m.parameters())


def check(t):
    """
    Check if tensor t is numerically correct.
    NaNs?, Infs?, standard deviation ok?
    """
    error = False
    if torch.isnan(t).any():
        error = "NaNs found"
    elif torch.isinf(t).any():
        error = "Infs found"
    elif t.float().std() < 1e-5:
        error = "stds < 1e-5 found"
    if error:
        traceback.print_stack(limit=5)
        raise Exception(f"check failed: {error}! Use %pdb to debug.")


def what(o):
    "this can't handle generators :("
    if isinstance(o, torch.Tensor) or isinstance(o, np.ndarray):
        return "<shp=" + str(list(o.shape)) + ", type=" + str(o.dtype) + ">"
    elif isinstance(o, tuple):
        return str("(" + ", ".join([what(x) for x in o]) + ")")
    elif isinstance(o, list):
        return str("[" + ", ".join([what(x) for x in o]) + "]")
    elif o is None:
        return "None"
    else:
        return str(str(type(o)) + ", " + repr(o))


def chained_try(funcs, item, try_all=True, post=lambda x: ",".join(x)):
    results = []
    for f in funcs:
        try:
            results.append(str(f(item)))
        except:
            pass
    return post(results)


def qna(m, bins=40):
    """
    q and a about your model.
    Plot all parameters and the distribution of their weights
    """
    for n, p in m.named_parameters():
        if (p == float("nan")).any() or p.mean() == float("nan"):
            print(n, "is NaN")
        else:
            plt.hist(p.detach().flatten().cpu().numpy(), bins=bins)
            plt.title(n)
            plt.show()


def hook_debug(m, db):
    """
    print the mean and std of all activations
    while data is flowing through your model
    Warning: this removes all hooks from your model!
    """

    def hk(s, _in, _out):
        def chk_print(t, which, idx=" ", lvl=1, _o=50):
            if isinstance(t, torch.Tensor):
                q = str(s)[:_o].replace("\n", "").ljust(_o)
                print(
                    f"{which} |",
                    q,
                    "|",
                    f"{t.float().mean().item():03.2f}".ljust(7)
                    + " | "
                    + f"{t.float().std().item():03.2f}".ljust(7),
                )
            elif isinstance(t, tuple):
                for i, one in enumerate(t):
                    chk_print(one, which, i, lvl + 1)

        # chk_print(_in, "in ")
        chk_print(_out, "out")

    def cl(s):
        s._forward_hooks = OrderedDict()

    # clear all hooks
    m.apply(cl)

    # register hook
    m.apply(lambda sf: sf.register_forward_hook(hk))

    # grab batch and run
    b = db.one_batch()
    _ = m(b[0])

    # clear all hooks
    m.apply(cl)
    return


def noop(x=None, *args, **kwargs):
    "Do nothing"
    return x


class Noop(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Noop, self).__init__()

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            return args[0]
        return args

    def param_groups(self):
        return []


def wrap_transform(f, ignore="all"):
    def _inner(*args, **kwargs):
        if ignore == "all":
            if kwargs["random"]:
                return f()
            else:
                return noop
        else:
            return f(*args, **kwargs)

    return _inner


def tensorize(x):
    # torch already
    if torch.is_tensor(x):
        return x.float()

    # some kind of numpy bytes
    arr = x
    if hasattr(x, "audio"):
        x = x.audio
    if hasattr(x, "data") and isinstance(x.data, (bytes, bytearray)):
        arr = np.frombuffer(x.data, dtype=np.float32)
    elif isinstance(x, (bytes, bytearray)):
        arr = np.frombuffer(x, dtype=np.float32)

    # np array
    # copy to avoid warning
    arr = np.copy(arr)
    return torch.FloatTensor(arr)[None]


def cudaize(x, device=None):
    if torch.is_tensor(x):
        if device is None:
            return x.cuda()
        return x.to(device)
    if isinstance(x, tuple):
        return tuple([cudaize(t, device) for t in x])
    if isinstance(x, list):
        return [cudaize(t, device) for t in x]
    if isinstance(x, dict):
        for k, v in x.items():
            x[k] = cudaize(v, device)
        return x
    import pdb
    pdb.set_trace()
    return x


def standardize(t, eps=1e-5):
    t.add_(-t.mean())
    t.div_(t.std() + eps)


def try_eval(m):
    m.training = False
    for c in m.children():
        if isinstance(c, torch.nn.quantized.modules.linear.LinearPackedParams):
            continue
        try:
            c.training = False
        except:
            pass
        try_eval(c)


def sanitize_str(o):
    # remove everything in parens
    o = re.sub("[\(\[].*?[\)\]]", "", o)
    o = re.sub("{.*?}", "", o)
    o = o.replace(".", "")
    o = o.replace(",", "")
    o = o.replace("?", "")
    o = o.replace("!", "")
    o = o.replace(":", "")
    o = o.replace(";", "")
    o = o.replace("-", "")
    o = o.replace("_", "")
    o = o.replace('"', "")
    o = o.replace("(", "")
    o = o.replace(")", "")
    o = o.replace("[", "")
    o = o.replace("]", "")
    o = o.replace("{", "")
    o = o.replace("}", "")
    o = o.replace("@", "")
    o = o.replace("#", "")
    o = o.replace("“", "")
    # only one space allowed
    o = re.sub(" +", " ", o)
    o = o.lower().strip()
    # add space at beginning (for BPE)
    o = " " + o
    return o


def ok(a):
    """
    check if sth is None
    """
    if isinstance(a, list):
        return all([ok(x) for x in a])
    return a is not None


def eq(a, b):
    """
    check if two things (i.e. tensors) are the same
    """
    if a is None or b is None:
        return False
    if isinstance(a, (list, tuple)):
        return all([eq(x, y) for x, y in zip(a, b)])
    return (a == b).all().item()


class TensorRingBuffer:
    def __init__(
        self, max_size, shape, dim=1, device=torch.device("cpu"), dtype=torch.float32
    ):
        self.max_size = max_size
        self.shp = shape
        self.dim = dim
        self.device = device
        self.dtype = dtype
        self.clear()

    def append(self, x):
        if self.max_size == 0:
            return False
        self.tensor = torch.cat([self.tensor, x], dim=self.dim)[:, -self.max_size :]
        return self.tensor.size(self.dim) == self.max_size

    def get(self):
        return self.tensor

    def trim_to(self, n):
        self.tensor = self.tensor[:, -n:]

    def clear(self):
        self.tensor = torch.zeros(self.shp, device=self.device, dtype=self.dtype)


# /data/stt/data/yt/es/o4Eu7FtENbk.wav,1270,6910,y el fin de magris ganis es el portavoz,39,16000,False
class TupleGetter:
    @staticmethod
    def file(tpl):
        return tpl[0]

    @staticmethod
    def xstart(tpl):
        return int(tpl[1])

    @staticmethod
    def xlen(tpl):
        return int(tpl[2])

    @staticmethod
    def label(tpl):
        return tpl[3].decode("utf-8")

    @staticmethod
    def ylen(tpl):
        return int(tpl[4])

    @staticmethod
    def sr(tpl):
        return int(tpl[5])

    @staticmethod
    def bad(tpl):
        raise Exception("not implemented")


def warn_about_license(component: str, function: str, link: str):
    prefix = "[warning]"
    print(prefix, "*" * 64)
    print(prefix, f"Using {function} in {component}...")
    print(prefix, f"Check its license: {link}")
    print(prefix, "*" * 64)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
