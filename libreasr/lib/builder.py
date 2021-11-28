import os
from pathlib import Path

import numpy as np
import pandas as pd

import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from libreasr.lib.utils import sanitize_str

CSV = {
    "train": "asr-dataset-train.csv",
    "valid": "asr-dataset-valid.csv",
    "test": "asr-dataset-test.csv",
}

TOKENIZER_MODEL_FILE = "tmp/tokenizer.yttm-model"
CORPUS_FILE = "tmp/corpus.txt"
COLUMNS = [
    "file",
    "xstart",
    "xlen",
    "label",
    "ylen",
    "sr",
    "bad",
]
ENCODING = "utf-8"


def resolve_csv_path(path, mode, suffix):
    if isinstance(suffix, str):
        suffix = [suffix]
    ps = []
    fixed = 0
    for s in suffix:
        p = CSV[mode]
        p = p.replace("train", f"{s}-train")
        p = p.replace("valid", f"{s}-valid")
        p = p.replace("test", f"{s}-test")
        p = path / p
        if p.exists():
            ps.append(p)
            continue
        if fixed == 0:
            ps.append(path / CSV[mode])
            fixed += 1
    return ps


def get_single_dataset_info(conf, lang, mode):
    paths = []
    for x in conf["datasets"][lang][mode]:
        if isinstance(x, str):
            paths.append(conf["dataset_paths"][x])
        else:
            p = conf["dataset_paths"][x[0]]
            paths.append((p, *x[1:]))
    return paths


def get_dataset_paths(conf, mode):
    lang = conf["lang"]
    if lang == "multi":
        paths = []
        for l in conf["datasets"].keys():
            if conf["datasets"][l] is None:
                continue
            dses_l = get_single_dataset_info(conf, l, mode)
            [paths.append(x) for x in dses_l]
    else:
        paths = get_single_dataset_info(conf, lang, mode)
    return paths


class ASRDatabunchBuilder:
    def __init__(self):
        self.do_shuffle = False
        self.drop_labels = False
        self.mode = None
        self.suffix = ""

    @staticmethod
    def from_config(conf, mode):
        lang = conf["lang"]
        paths = get_dataset_paths(conf, mode)
        pcent = conf["pcent"][mode]
        suffix = conf.get("suffix", "")
        builder = (
            ASRDatabunchBuilder().set_mode(mode).set_suffix(suffix).multi(paths, pcent)
        )
        builder.set_drop_labels(conf.get("builder", {}).get("drop_labels", False))
        if conf["apply_limits"]:
            builder = builder.x_bounds(conf["almins"] * 1000.0, conf["almaxs"] * 1000.0)
            builder = builder.y_bounds(conf["y_min"], conf["y_max"]).set_max_words(
                conf["y_max_words"]
            )
        if conf["shuffle_builder"][mode]:
            builder.shuffle()
        builder.build()
        return builder

    def set_mode(self, mode):
        self.mode = mode
        return self

    def set_suffix(self, suffix):
        self.suffix = suffix
        return self

    def set_drop_labels(self, dl):
        self.drop_labels = dl
        return self

    def single(self, path):
        path = Path(path)
        q = resolve_csv_path(path, self.mode, self.suffix)
        self.df = pd.read_csv(q)
        print(f"[builder] [{self.mode}] df {q} loaded.")
        return self

    def multi(self, paths, pcent=1.0):
        dfs = []
        for x in paths:
            if isinstance(x, str):
                path = x
                pcent_one = 1.0
            else:
                path = x[0]
                pcent_one = x[1]
            path = Path(path)
            qs = resolve_csv_path(path, self.mode, self.suffix)
            for q in qs:
                df = pd.read_csv(q)
                if pcent_one != 1.0:
                    df = df.sample(frac=pcent_one, random_state=4471)
                if pcent != 1.0:
                    df = df.sample(frac=pcent, random_state=4471)
                print(f"[builder] [{self.mode}] df {q} loaded. len={len(df)}")
                dfs.append(df)
        # TODO set sort=False at some point (as it is not needed)
        self.df = pd.concat(dfs, ignore_index=True, copy=False, sort=True)
        return self

    def x_bounds(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max
        return self

    def y_bounds(self, y_min, y_max):
        self.y_min = y_min
        self.y_max = y_max
        return self

    def set_max_words(self, max_words):
        self.max_words = max_words
        return self

    def shuffle(self, b=True):
        self.do_shuffle = b
        return self

    def _assert_has_df(self):
        assert hasattr(self, "df")

    def _apply_limits(self):
        df = self.df
        if hasattr(self, "x_min"):
            df = df[df.xlen >= self.x_min]
        if hasattr(self, "x_max"):
            df = df[df.xlen <= self.x_max]
        if hasattr(self, "y_min"):
            df = df[df.ylen >= self.y_min]
        if hasattr(self, "y_max"):
            df = df[df.ylen <= self.y_max]
        if hasattr(self, "max_words"):

            def filt(row):
                l = sanitize_str(row.label)
                if len(l.split(" ")) <= self.max_words and len(l) > 5:
                    return True
                return False

            df = df[df.apply(filt, axis=1)]
        self.df = df

    def _fix_columns(self):
        df = self.df
        df["label"] = df["label"].astype(str)
        df["sr"] = df["sr"].astype(int)
        self.df = df

    def _remove_and_clean_columns(self):
        if self.drop_labels:
            self.df.label = ""
            self.df.ylen = 1
        try:
            self.df.drop(["bad", "lang"], axis="columns", inplace=True)
        except:
            pass

    def build(self, sort=False, seed=42):
        self._assert_has_df()
        self._fix_columns()
        self._apply_limits()
        self._remove_and_clean_columns()
        if self.do_shuffle:
            self.df = self.df.sample(frac=1, random_state=seed)
        if sort:
            self.df.sort_values(by="label", inplace=True)

        # reorder columns
        self.df = self.df.reindex(COLUMNS, axis=1)

        # encode labels
        self.df.label = self.df.label.str.encode(ENCODING)

        self.built = True
        return self

    def get(self):
        df = self.df
        _fs = df.file.tolist()
        _is = list(range(len(_fs)))
        _ts = list(df.itertuples(index=False))
        return _fs, _is, _ts, df

    def print(self):
        def output(a, b):
            a, b = str(a), str(b)
            la = len(a)
            b = b.rjust(40 - la)
            print(f"{a}: {b}")

        if self.built:
            output("mode", self.mode)
            output("num samples", len(self.df))
            output(f"num hours", f"{self.df.xlen.values.sum() / (1000.0 * 3600.0):.2f}")
            output(
                f"sample duration mean",
                f"{self.df.xlen.values.mean() / 1000.0:.2f} sec",
            )
            output(
                f"sample duration std", f"{self.df.xlen.values.std() / 1000.0:.2f} sec"
            )
            print()
            print(self.df.head())
            print()
        return self

    def dump_labels(self, to_file=CORPUS_FILE):
        assert self.built
        print(f"Dumping labels to {to_file}")
        os.makedirs(Path(to_file).parent, exist_ok=True)
        with open(to_file, "w") as f:
            for i, row in tqdm.tqdm(self.df.iterrows(), total=len(self.df)):
                # decode label
                l = row.label.decode(ENCODING)
                f.write(sanitize_str(l) + "\n")
        print("Done.")

    def data(self, as_list=True, to_file=CORPUS_FILE):
        assert self.built
        if as_list:
            lines = []
            for i, row in tqdm.tqdm(self.df.iterrows(), total=len(self.df)):
                # decode label
                l = row.label.decode(ENCODING)
                lines.append(sanitize_str(l))
            return lines
        else:
            self.dump_labels(to_file=to_file)
            return to_file

    def plot(self, save=False):
        if self.built:
            import matplotlib.pyplot as plt

            plt.hist(self.df.xlen.values, bins=50)
            plt.title(f"xlen ({self.df.xlen.values.sum() / 3600000.:.2f} hours)")
            plt.show()
            if save:
                plt.savefig("./plots/figures/data-x.png", dpi=300)
            plt.hist(self.df.ylen.values, bins=30)
            plt.title("ylen")
            plt.show()
            if save:
                plt.savefig("./plots/figures/data-y.png", dpi=300)
            try:
                plt.hist(self.df.xlen.values / self.df.ylen.values, bins=50)
                plt.title("xlen/ylen")
                plt.show()
                if save:
                    plt.savefig("./plots/figures/data-x-y.png", dpi=300)
            except:
                pass
        return self


if __name__ == "__main__":
    fs, _is, ts, df = (
        ASRDatabunchBuilder()
        .multi(
            [
                "/data/stt/data/common-voice/fr",
                "/data/stt/data/common-voice/de",
                "/data/stt/data/common-voice/en",
            ]
        )
        .x_bounds(0, 5000)
        .y_bounds(2, 20)
        .build()
        .print()
        .plot()
        .get()
    )
    print(fs[0], _is[0], ts[0])
    print("\n\nhead:", df.head(n=20))
    print("\n\ntail:", df.tail(n=20))
