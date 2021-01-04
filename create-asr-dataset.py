from functools import partial
from pathlib import Path
import multiprocessing
import glob

import tqdm
import pandas as pd
import numpy as np

import torch
import torchaudio

# fastai2_audio
# add flac to supported audio types
import mimetypes
mimetypes.types_map[".flac"] = "audio/flac"

from fastai2_audio.core.all import get_audio_files

from libreasr.lib.utils import sanitize_str


PRINT_DROP = False


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
    print_fun(f"df saved to {path}")


def process_one(file, get_labels):
    rows = []
    try:
        if file.suffix == ".m4a":
            raise Exception("no audio file")
        aud, sr = torchaudio.load(file)
        assert aud.size(0) >= 1 and aud.size(1) >= 1
        xlen = int((aud.size(1) / float(sr)) * 1000.0)
        labels = get_labels(file, duration=xlen)
        for (xstart, spanlen, label, ylen) in labels:
            if ylen >= 2:
                bad = False
            else:
                bad = True
            if spanlen == -1:
                spanlen = xlen
            rows.append((str(file.absolute()), xstart, spanlen, label, ylen, sr, bad))
    except Exception as e:
        pass
    finally:
        if len(rows) == 0:
            xstart, xlen, label, ylen, sr, bad = 0, 0, "", 0, -1, True
            rows.append((str(file.absolute()), xstart, xlen, label, ylen, sr, bad))
    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path to the dataset")
    parser.add_argument(
        "dataset",
        type=str,
        help="which dataset: common-voice | tatoeba | tf-speech | librispeech | yt",
    )
    parser.add_argument(
        "--workers", default=1, type=int, help="how many pool workers to create"
    )
    parser.add_argument(
        "--block-size",
        default=2,
        type=int,
        help="in case of vtt format, how many sentences to collect together (min: 2)",
    )
    parser.add_argument(
        "--save-every-pcent",
        default=5,
        type=int,
        help="save resulting df every N% of all files",
    )
    parser.add_argument(
        "--print-every-pcent",
        default=5,
        type=int,
        help="print info every N% of all files",
    )
    parser.add_argument(
        "--lang", default="en", type=str, help="language",
    )
    parser.add_argument(
        "--out",
        default="asr-dataset.csv",
        type=str,
        help="name of the resulting csv file",
    )
    parser.add_argument(
        "--filter",
        default="",
        type=str,
        help="filter the files by this csv",
    )
    args = parser.parse_args()

    path = Path(args.path)
    dataset = args.dataset
    p = path
    save_path = path / args.out

    # create df
    # see if exists
    cols = [
        "file",
        "xstart",
        "xlen",
        "label",
        "ylen",
        "sr",
        "bad",
    ]
    if save_path.exists():
        df = pd.read_csv(save_path)
        print(f"> df restored from {save_path}")
    else:
        df = pd.DataFrame(columns=cols)
        print("> df NOT restored (not found?)")

    # grab all audio files
    files = get_audio_files(p)
    print("> raw files:", len(files))

    # filter out files that are already in the df
    files = pd.Series([str(x) for x in files])
    res = ~files.isin(df.file)

    # filter out files that are in --filter if specified
    if args.filter != "":
        df_filter = pd.read_csv(p / args.filter)
        res2 = files.isin(pd.unique(df_filter.file))
        res = res & res2

    files = [Path(x) for x in files[res].tolist()]
    print("> filtered files:", len(files))

    # get_labels for each dataset format
    if dataset == "common-voice":
        label_df = pd.read_csv(path / "validated.tsv", delimiter="\t")

        def get_labels(file, **kwargs):
            n = file.stem + ".mp3"
            l = label_df[label_df.path == n].sentence.iloc[0]
            return [(0, -1, l, len(l))]

    elif dataset == "tatoeba":
        fname = glob.glob(str(path) + "/dataset_*.csv")[0]
        label_df = pd.read_csv(fname)
        label_df["audio_id"] = label_df["audio_id"].astype("str")
        label_df["text"] = label_df["text"].astype("str")

        def get_labels(file, **kwargs):
            aid = file.stem
            l = label_df[label_df.audio_id == aid].text.iloc[0]
            return [(0, -1, l, len(l))]

    elif dataset == "tf-speech":

        def get_labels(file, **kwargs):
            l = file.parent.name
            if l == "_background_noise_":
                l = ""
            return [(0, -1, l, len(l))]

    elif dataset == "librispeech":

        def get_labels(file, **kwargs):
            # parse filename
            p = file
            ns = p.stem.split("-")
            n1 = ns[0]
            n2 = ns[1]
            n3 = ns[2]
            file_id = f"{n1}-{n2}-{n3}"

            # read labelfile
            l_fname = p.parent / f"{n1}-{n2}.trans.txt"
            content = open(l_fname, "r").read()

            # extract text
            l = ""
            llen = 0
            for line in content.split("\n"):
                if line.startswith(file_id):
                    l = line.split(" ", 1)[-1].lower()
                    llen = len(l)
                    return [(0, -1, l, llen)]
            return [(0, -1, l, llen)]

    elif dataset == "yt":

        # https://github.com/glut23/webvtt-py
        import webvtt
        from collections import Counter

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        def parse_timestamp(tsp):
            "00:04:06.209"
            a = tsp.split(":")
            hrs = int(a[0])
            min = int(a[1])
            b = a[2].split(".")
            sec = int(b[0])
            msec = int(b[1])
            return hrs, min, sec, msec

        def timestamp_to_millis(tsp):
            hrs, min, sec, msec = tsp
            return int(hrs * 3600 * 1000 + min * 60 * 1000 + sec * 1000 + msec)

        def get_labels(file, duration):

            # parse vtt file
            block_size = args.block_size
            label_file = f"{file.parent}/{file.stem}.{args.lang}.vtt"
            # print(label_file)
            try:
                vtt_all = webvtt.read(label_file)
            except:
                # print("read vtt error", label_file)
                return []
            vtt_blocks = list(chunks(vtt_all, block_size))

            transcripts = []
            printed = False
            for i, vtt in enumerate(vtt_blocks):
                if len(vtt) < 2:
                    continue
                transcript = ""

                lines = []
                for line in vtt:
                    t = line.text.strip().splitlines()
                    lines.extend(t)

                if i != 0:
                    # pop first line
                    # so we don't get repeated text
                    popped = lines.pop(0)

                # abandon lines which are the same
                previous = None
                for line in lines:
                    if line == previous:
                        continue
                    transcript += " " + line
                    previous = line

                start = timestamp_to_millis(parse_timestamp(vtt[0].start))
                end = timestamp_to_millis(parse_timestamp(vtt[-1].end))
                # print(i, start, end, transcript)

                # strip & fix whitespaces
                transcript = transcript.strip()
                transcript = transcript.replace("  ", " ")

                # sanity checks
                sanitized = sanitize_str(transcript)
                if start >= duration or (end - start) <= 0.0 or len(sanitized) < 3:
                    if PRINT_DROP:
                        print(
                            "drop",
                            file.stem,
                            start,
                            end,
                            duration,
                            vtt[0].start,
                            vtt[-1].end,
                            sanitized,
                        )
                    continue

                # keep automatic captions (drop closed captions)
                if int(end - start) % 1000 == 0:
                    # print("drop", file.stem, "closed captions")
                    continue

                # add span
                transcripts.append((start, end - start, transcript, len(transcript)))

            return transcripts

    # spawn a pool
    p = multiprocessing.Pool(args.workers)
    bads = 0
    with tqdm.tqdm(total=len(files)) as t:
        for i, tpls in enumerate(
            p.imap_unordered(partial(process_one, get_labels=get_labels), files)
        ):

            # iterate through labels
            data = None
            for tpl in tpls:

                # count bads
                if tpl[-1]:
                    bads += 1
                else:
                    data = tpl

            # print info
            if i % int(args.print_every_pcent * 0.01 * len(files) + 1) == 0:
                if data:
                    t.write("> data: " + str(tpl))
                t.write(f"> df len: {len(df)}")
                t.write(
                    "> pcent bad: " + f"{int((bads / (len(df)+len(tpls))) * 100.)}%"
                )

            # insert into df
            df = df.append(
                [{k: v for k, v in zip(cols, tpl)} for tpl in tpls], ignore_index=True
            )

            # save periodically
            if i % int(args.save_every_pcent * 0.01 * len(files) + 1) == 0:
                save(df, save_path, print_fun=t.write)

            # increment
            t.update()

    # filter out bad ones
    df = df[df.bad == False]

    # final save
    save(df, save_path)
