DEFAULT_CONFIG_PATH = "./config/testing.yaml"
DEFAULT_SR = 16000
DEFAULT_BATCH_SIZE = 6

# language model params
LM_ALPHA = 0.1
LM_THETA = 1.0
LM_TEMP = 1.0
LM_DEBUG = False

# model params
MODEL_TEMP = 1.0

# beam search
DEFAULT_MAX_ITERS = 5
DEFAULT_BEAM_SEARCH_OPTS = {
    "beam_width": 8,
    "topk_next": 2,
    "predictor_cache_sz": 1024,
    "joint_cache_sz": 1024,
    "score_cache_sz": 1024,
    "debug": False,
}

# streaming
DEFAULT_STREAM_CHUNK_SZ = 0.08  # secs
DEFAULT_STREAM_BUFFER_N_FRAMES = 2
DEFAULT_STREAM_OPTS = {
    "buffer_n_frames": DEFAULT_STREAM_BUFFER_N_FRAMES,
    "sr": 16000,
    "chunk_sz": DEFAULT_STREAM_CHUNK_SZ,
}

# example audio
WAVS = [
    "./assets/samples/common_voice_de_17672459.wav",
    "./assets/samples/common_voice_de_18227443.wav",
    "./assets/samples/common_voice_de_18520948.wav",
    "./assets/samples/common_voice_de_17516889.wav",
    "./assets/samples/common_voice_de_18818000.wav",
    # 798460, 52470
    "./assets/samples/yt_de_TzSV5FJyEeg.wav",
]

# example labels
LABELS = [
    "als die stadtmauern errichtet wurden hieß es dass sie unbezwingbar seien",
    "insbesondere keine topflappen",
    "durch einen elektrisierten weidezaun ist die koppel begrenzt",
    "die beamten gehen nun verstärkt gegen illegale straßenrennen vor",
    "frau senninger aus dem zweiten stock hat bei einem sturz einen oberschenkelhalsbruch erlitten",
    """angebracht ist also wenn ich die spd richtig verstanden habe vergangene woche ist von ihrer spitze von canalis her kategorisch erklärt mehr auf die fresse nahles keine katze erklärt keine bundeswehrbeteiligung mit man alles also sind wir demnächst formales los und bomben in syrien ich verstehe aber nicht gut aber es wurde also in der vergangenen woche als die gerüchteküche kirche so bearbeitete war auch sogar die version solle man präventiv eine schnecke machen auch wennn also chemie einem waffeneinsatz sagen es hat android soll man dann schon das wurde inzwischen relativ klar ausgefallen gesagt nee allerdings wenn denn dann tatsächlich nachweisbar eingesetzt ist lassen da empfehle ich noch mal den film minority""",
]

# zipped audio & labels
AUDIOS = list(zip(WAVS, LABELS))

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
