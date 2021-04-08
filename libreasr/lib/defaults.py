DEFAULT_CONFIG_PATH = "./config/testing.yaml"
DEFAULT_SR = 16000
DEFAULT_BATCH_SIZE = 6

LM_ALPHA = 0.1
LM_THETA = 1.0
LM_TEMP = 1.0
LM_DEBUG = False

MODEL_TEMP = 1.0

# streaming
DEFAULT_STREAM_OPTS = {
    "reset_thresh": 7000,
    "buffer_n_frames": 3,
    "downsample": 8,
    "n_buffer": 2,
}

# alias languages
#  to releases with
#  pretrained models
ALIASES = {"de": "de-1.1.0"}

# ${lang}-${release}
DOWNLOADS = {
    "de-1.1.0": {
        "config.yaml": {
            "storage": "gdrive",
            "id": "1VhC8oBcMgh-yP9PvMbteVIDGous-aVy_",
            # "sha256": "535a9984d10bccbad61838ca45f850e04c2a665a5546cd34cf16cb2a6de8bfcc",
            "sha256": "SKIP",
        },
        "tokenizer.yttm-model": {
            "storage": "gdrive",
            "id": "1y2LDx4iTAI3-_mzevmbR-nGtZXiMwfCa",
            "sha256": "f035dd0c245a169cdca919394b1637861fc8d73da3203ff6e18c1a131f473af6",
        },
        "model.pth": {
            "storage": "gdrive",
            "id": "19LrGnj8DLbCw5BlabfuTUPmXYc0jzSdD",
            "sha256": "460a20f0c97302d86d0b1ffa01b952279dba3e409e71a564a69400e37cac0a6f",
        },
        "lm.pth": {
            "storage": "gdrive",
            "id": "1Ba74YKDwx4qusRS1SMnKrz-SQr5Ity3p",
            "sha256": "e010207c6ec871dc310c27432188e38a6ac983837903e96b6c974d956d1acf49",
        },
    },
}
