import tarfile
from pathlib import Path
import os
import hashlib
import glob

from IPython.core.debugger import set_trace

import torch

from fastai.learner import load_model

from libreasr.lib.quantization import load_quantized_model, try_quantize, quantize_model


_PATH_ARCHIVE = Path("libreasr-model.tar.gz")
_PATH_TOKENIZER = Path("tokenizer.yttm-model")
_PATH_MODEL = Path("model.pth")
_PATH_DEST = Path("./tmp")


def add(tar, p_archive, p_real):
    tar.addfile(tarfile.TarInfo(str(p_archive)), str(p_real))


def save_asr_model(
    lang,
    path_tokenizer=_PATH_TOKENIZER,
    path_model=_PATH_MODEL,
    path_archive=_PATH_ARCHIVE,
):
    """
    Bundles
    - tokenizer.yttm-model (tokenizer model)
    - model.pth (PyTorch model)
    into a single .tar.gz :path_archive:
    """
    p_base_real = _PATH_DEST / Path(lang)
    p_base_arc = Path(lang)
    tar = tarfile.open(path_archive, mode="w:gz")
    add(tar, p_base_arc / path_tokenizer, p_base_real / path_tokenizer)
    add(tar, p_base_arc / path_model, p_base_real / path_model)
    tar.close()


def extract_tars(paths_archive=None, path_dest=_PATH_DEST):
    """
    extract .tar.gz file
    """
    if paths_archive is None:
        paths_archive = glob.glob("./libreasr-model-*.tar.gz")
    for arc in paths_archive:
        tar = tarfile.open(arc)
        tar.extractall(path=path_dest)


def load_asr_model(
    model,
    lang_name,
    lang,
    pre_quantization,
    post_quantization,
    paths,
    device="cpu",
    lm=None,
    path_tokenizer=_PATH_TOKENIZER,
    path_archive=_PATH_ARCHIVE,
):
    """
    Loads an asr model from a given .tar.gz file
    """
    # delete attrs
    model.lang = None
    model.lm = None

    # model
    name = model.__class__.__name__
    at = None
    try:
        print(f"[load] {paths}")
        if pre_quantization:
            at = str(paths)
            model = load_quantized_model(model, lang, paths)
            model = model.eval()
        else:
            at = Path(paths)
            load_model(
                at,
                model,
                None,
                with_opt=False,
                device=device,
            )
    except Exception as e:
        print(f"Unable to load model ('{at}', '{name}')")
        print(" >", e)

    # quantize model after loading
    if post_quantization:
        qmodel = try_quantize(model, quantize_model)
        qmodel = qmodel.eval()
        return qmodel

    return model
