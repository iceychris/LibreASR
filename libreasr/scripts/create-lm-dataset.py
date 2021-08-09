import argparse
import os
from pathlib import Path

import pandas as pd
import tqdm

from libreasr.lib.config import open_config
from libreasr.lib.utils import sanitize_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lang", type=str, help="Language")
    parser.add_argument(
        "--config",
        type=str,
        help="Which model_id to load & convert to TorchScript.",
        default="./config/base.yaml",
    )
    parser.add_argument(
        "--split", type=str, help="Which split to use (train/valid)", default="train"
    )
    parser.add_argument("--out", type=str, help="Output file name", default="EMPTY")
    args = parser.parse_args()

    # setup args
    out = args.out
    if out == "EMPTY":
        out = f"corpus-{args.lang}-new.txt"

    # get paths
    c = open_config(path=args.config)
    dses = c["datasets"][args.lang][args.split]
    dses_paths = [
        Path(c["dataset_paths"][x]) / f"asr-dataset-{args.split}.csv" for x in dses
    ]

    # open dfs & fix text
    lines = []

    def fn(x):
        try:
            return sanitize_str(x.decode("utf-8"))
        except:
            return ""

    for ds in dses_paths:
        df = pd.read_csv(ds)
        df.label = df.label.str.encode("utf-8")
        lines.extend([fn(x) for x in df.label.tolist()])
        del df
    print(lines[:5], lines[-5:])
    print()

    # dump
    print(f"Dumping lines to {out}...")
    os.makedirs(Path(out).parent, exist_ok=True)
    with open(out, "w") as f:
        for line in tqdm.tqdm(lines, total=len(lines)):
            f.write(line + "\n")
    print("Done.")
