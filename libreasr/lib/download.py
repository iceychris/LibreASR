import os
import hashlib

import requests
from tqdm import tqdm

from libreasr.lib.defaults import ALIASES, DOWNLOADS


def check_hash(path, sha256):
    if sha256 == "SKIP":
        return True
    BUF_SIZE = 65536  # lets read stuff in 64kb chunks!
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            h.update(data)
    return h.hexdigest() == sha256


def download(id, sha256, path, storage="gdrive", **kwargs):
    if storage == "gdrive":
        if not os.path.exists(path) or not check_hash(path, sha256):
            head, tail = os.path.split(path)
            os.makedirs(head, exist_ok=True)

            URL = "https://docs.google.com/uc?export=download"

            session = requests.Session()
            response = session.get(URL, params={"id": id}, stream=True)

            token = None
            for key, value in response.cookies.items():
                if key.startswith("download_warning"):
                    token = value
                    break

            if token:
                params = {"id": id, "confirm": token}
                response = session.get(URL, params=params, stream=True)

            total = int(response.headers.get("content-length", 0))

            with open(path, "wb") as file, tqdm(
                desc=f"Downloading {tail} to {head}",
                total=total,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)

            # recheck hash
            if not check_hash(path, sha256):
                raise Exception("WARNING: SHA256-Hash mismatch for file " + path)


def get_lang_and_release(lang):
    ds = ALIASES
    if lang in ds.keys():
        return lang, ds[lang]
    for l, r in ALIASES.items():
        if r == lang:
            return l, r
    return lang, None


def download_all(lang, filter_fn=lambda x: x, base_path="~/.cache/LibreASR"):
    if lang is not None:
        base_path = os.path.expanduser(base_path)
        lang, release = get_lang_and_release(lang)
        base_path = os.path.join(base_path, release)
        files = DOWNLOADS.get(release, {})
        fnames = filter_fn(list(files.keys()))
        for fname in fnames:
            kwargs = files[fname]
            path = os.path.join(base_path, fname)
            download(path=path, **kwargs)
        return lang, release, os.path.join(base_path, "config.yaml")


def download_configs(fname="config.yaml", base_path="~/.cache/LibreASR"):
    base_path = os.path.expanduser(base_path)
    paths = []
    for release, files in DOWNLOADS.items():
        kwargs = files[fname]
        path = os.path.join(base_path, release, fname)
        paths.append(path)
        download(path=path, **kwargs)
    return paths
