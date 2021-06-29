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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path to the dataset")
    parser.add_argument(
        "--split", default=0.05, type=float, help="percentage of valid and test split"
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="random seed used while splitting"
    )
    parser.add_argument(
        "--out",
        default="asr-dataset.csv",
        type=str,
        help="name of the resulting csv file",
    )
    args = parser.parse_args()

    path = Path(args.path)
    p = path

    out = args.out
    df_path = p / args.out
    path_train = p / args.out.replace(".csv", "-train.csv")
    path_valid = p / args.out.replace(".csv", "-valid.csv")
    path_test = p / args.out.replace(".csv", "-test.csv")

    # check if exists
    if df_path.exists():
        df = pd.read_csv(df_path)
        print(f"> df loaded from {df_path}")
    else:
        raise Exception("asr-dataset.csv does not exist")

    # get unqiue files
    all_uniq = pd.unique(df.file)

    # first, train and non_train
    train, non_train = train_test_split(
        all_uniq, test_size=args.split * 2.0, random_state=args.seed
    )

    # then, valid and test
    valid, test = train_test_split(non_train, test_size=0.5, random_state=args.seed)

    # convert to dfs
    train = df[df.file.isin(train)]
    valid = df[df.file.isin(valid)]
    test = df[df.file.isin(test)]

    # save
    check_save(train, path_train)
    check_save(valid, path_valid)
    check_save(test, path_test)
