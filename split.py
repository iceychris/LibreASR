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
    if path_train.exists():
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
    args = parser.parse_args()

    path = Path(args.path)
    p = path

    df_path = p / "asr-dataset.csv"
    path_train = p / "asr-dataset-train.csv"
    path_valid = p / "asr-dataset-valid.csv"
    path_test  = p / "asr-dataset-test.csv"

    # check if exists
    if df_path.exists():
        df = pd.read_csv(df_path)
        print(f"> df loaded from {df_path}")
    else:
        raise Exception("asr-dataset.csv does not exist")

    # first, train and non_train
    train, non_train = train_test_split(df, test_size=args.split * 2., random_state=args.seed)

    # then, valid and test
    valid, test = train_test_split(non_train, test_size=0.5, random_state=args.seed)

    # save
    check_save(train, path_train)
    check_save(valid, path_valid)
    check_save(test, path_test)
