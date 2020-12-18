"""
ASRDatabunchBuilder is responsible for loading datasets
in csv format.
"""

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


class ASRDatabunchBuilder:
    def __init__(self):
        self.do_shuffle = False
        self.mode = None

    @staticmethod
    def from_config(conf, mode):
        paths = [conf["dataset_paths"][x] for x in conf["datasets"]]
        pcent = conf["pcent"][mode]
        builder = ASRDatabunchBuilder().set_mode(mode).multi(paths, pcent)
        if conf["apply_limits"]:
            builder = (
                builder.x_bounds(conf["almins"] * 1000.0, conf["almaxs"] * 1000.0)
                .y_bounds(conf["y_min"], conf["y_max"])
                .set_max_words(conf["y_max_words"])
            )
            if conf["shuffle_builder"][mode]:
                builder.shuffle()
        builder.build()
        return builder

    def set_mode(self, mode):
        self.mode = mode
        return self

    def single(self, path):
        path = Path(path)
        self.df = pd.read_csv(path / CSV[self.mode])
        return self

    def multi(self, paths, pcent=1.0):
        dfs = []
        for path in paths:
            path = Path(path)
            df = pd.read_csv(path / CSV[self.mode])
            if pcent != 1.0:
                df = df.sample(frac=pcent)
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

    def _remove_columns(self):
        try:
            self.df.drop(["bad", "lang"], axis="columns", inplace=True)
        except:
            pass

    def build(self, sort=False, seed=42):
        self._assert_has_df()
        self._fix_columns()
        self._apply_limits()
        self._remove_columns()
        if self.do_shuffle:
            self.df = self.df.sample(frac=1, random_state=seed)
        if sort:
            self.df.sort_values(by="label", inplace=True)
        self.built = True
        return self

    def get(self):
        df = self.df
        _fs = df.file.tolist()
        _is = list(range(len(_fs)))
        _ts = list(df.itertuples(index=False))
        return _fs, _is, _ts, df

    def print(self):
        if self.built:
            print("mode:", self.mode)
            print("num samples:", len(self.df))
            print("num hours:", self.df.xlen.values.sum() / (1000.0 * 3600.0))
            print(self.df.head())
        return self

    def dump_labels(self, to_file=CORPUS_FILE):
        assert self.built
        print(f"Dumping labels to {to_file}")
        os.makedirs(Path(to_file).parent, exist_ok=True)
        with open(to_file, "w") as f:
            for i, row in tqdm.tqdm(self.df.iterrows(), total=len(self.df)):
                f.write(sanitize_str(row.label) + "\n")
        print("Done.")

    def train_tokenizer(
        self,
        corpus_file=CORPUS_FILE,
        model_file=TOKENIZER_MODEL_FILE,
        vocab_sz=50000,
        dump_labels=True,
    ):
        assert self.built
        import youtokentome as yttm

        # first we need to dump labels
        if dump_labels:
            self.dump_labels(corpus_file)

        # train model
        print("Training yttm model...")
        yttm.BPE.train(data=corpus_file, vocab_size=vocab_sz, model=model_file)
        print("Done.")

        # load model (for testing)
        print("Testing yttm model...")
        bpe = yttm.BPE(model=model_file)
        # Two types of tokenization
        test_text = "Are you freakin' crazy?"
        encoded1 = bpe.encode([test_text], output_type=yttm.OutputType.ID)
        encoded2 = bpe.encode([test_text], output_type=yttm.OutputType.SUBWORD)
        decoded = bpe.decode(encoded1)
        print(encoded1)
        print(encoded2)
        print(decoded)

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
            plt.hist(self.df.xlen.values / self.df.ylen.values, bins=50)
            plt.title("xlen/ylen")
            plt.show()
            if save:
                plt.savefig("./plots/figures/data-x-y.png", dpi=300)
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
