import argparse

from pathlib import Path

from fastai.data.external import untar_data


URL_CV = "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5.1-2020-06-22/"
URL_LS = "http://www.openslr.org/resources/12/"
FILES_LS = [
    "test-clean.tar.gz",
    "test-other.tar.gz",
    "dev-clean.tar.gz",
    "dev-other.tar.gz",
    "train-clean-100.tar.gz",
    "train-clean-360.tar.gz",
    "train-other-500.tar.gz",
]


def download_librispeech(args, files=FILES_LS):
    print("LibriSpeech...")
    path = args.path
    for _file in files:
        fname = Path(path) / _file
        print(f"  Downloading to {fname}...")
        url = URL_LS + _file
        untar_data(
            url,
            fname=fname,
            dest=Path(path) / "LibriSpeech",
            force_download=args.download,
        )


def download_common_voice(args, langs):
    print("CommonVoice...")
    path = args.path
    for lang in langs:
        _file = f"{lang}.tar.gz"
        fname = Path(path) / _file
        print(f"  Downloading to {fname}...")
        url = URL_CV + _file
        untar_data(
            url, fname=fname, dest=Path(path) / lang, force_download=args.download
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download common ASR corpora")
    parser.add_argument("language", type=str, default="en", help="language to download")
    parser.add_argument("--path", default="./data", help="destination path")
    parser.add_argument(
        "--download", action="store_true", help="redownload all archives again"
    )
    args = parser.parse_args()

    if args.language == "en":
        download_librispeech(args)
    download_common_voice(args, [args.language])
