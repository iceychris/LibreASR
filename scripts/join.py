from functools import partial
from pathlib import Path
import multiprocessing
import glob

import tqdm
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def save(df, path, print_fun=print):
    df.to_csv(path, index=False)
    print_fun(f"> df saved to {path}")


def check_save(df, path, **kwargs):
    if path.exists():
        usr = input(f"> ! csv already exists. overwrite {path}? [y/N] \n")
        if usr.lower() != "y":
            return
    save(df, path, **kwargs)


def print_good(p, df):
    print(f"> df loaded from {p} (len={len(df)})")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path to the dataset")
    parser.add_argument(
        "-i", default="", type=str, help="input csv file"
    )
    parser.add_argument(
        "-o", default="", type=str, help="output csv file"
    )
    parser.add_argument(
        "--filters",
        type=str,
        metavar="N",
        nargs="+",
        help="csv files to filter out (blacklist)",
    )
    args = parser.parse_args()

    path = Path(args.path)
    p = path

    # asserts
    assert args.i != ""
    assert args.o != ""
    assert len(args.filters) != 0

    # prepare paths
    path_i = p / args.i
    path_o = p / args.o
    paths_f = [p / f for f in args.filters]

    # load dfs if they exist
    if path_i.exists():
        df_i = pd.read_csv(path_i)
        print_good(path_i, df_i)
    else:
        raise Exception(f"{path_i} does not exist")
    dfs_f = []
    for path_f in paths_f:
        if path_f.exists():
            df_f = pd.read_csv(path_f)
            dfs_f.append(df_f)
            print_good(path_f, df_f)
        else:
            raise Exception(f"{path_f} does not exist")

    # glue filters together
    df_f = pd.concat(dfs_f, ignore_index=True)
    print(f"> filter df created (len={len(df_f)})")

    # get unqiues
    uniq_f = pd.unique(df_f.file)
    print(f"> uniq_f len={len(uniq_f)}")

    # convert to stems
    stems_i = pd.Series([Path(x).stem for x in df_i.file])
    stems_f = pd.Series([Path(x).stem for x in uniq_f])
    print(">", stems_i[:3])
    print(">", stems_f[:3])

    # do filtering
    mask = stems_i.isin(stems_f)
    print(f"> mask (len={len(mask)}, sum={sum(mask)})")

    # apply mask
    df_filtered = df_i[~mask]
    print(f"> df_filtered len={len(df_filtered)}")

    # save
    check_save(df_filtered, path_o)

