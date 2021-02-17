import traceback
from collections import OrderedDict
import re

import matplotlib.pyplot as plt

import torch
import numpy as np

LANGS = "de,en,fr,sv,es,eo,it,nl".split(",")
LANG_TO_IDX = {l: i for i, l in enumerate(LANGS)}


class Text(str):
    def show(self, **kwargs):
        print(self)


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
    if hasattr(x, "data") and isinstance(x.data, (bytes, bytearray)):
        arr = np.frombuffer(x.data, dtype=np.float32)

    # np array
    # copy to avoid warning
    arr = np.copy(arr)
    return torch.FloatTensor(arr)[None]


def cudaize(x):
    if torch.is_tensor(x):
        return x.cuda()
    return [cudaize(t) for t in x]


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
