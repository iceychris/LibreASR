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
    "implementation": "libreasr",
    "beam_width": 2,
    "topk_next": 2,
    "predictor_cache_sz": 4,  # 128, # 1024,
    "joint_cache_sz": 4,  # 128, # 1024,
    "score_cache_sz": 4,  # 128, # 1024,
    "state_beam": 2.3,
    "expand_beam": 2.3,
    "debug": False,
}

# streaming
DEFAULT_STREAM_CHUNK_SZ = 0.08  # secs
DEFAULT_STREAM_BUFFER_N_FRAMES = 2
DEFAULT_STREAM_OPTS = {
    "buffer_n_frames": DEFAULT_STREAM_BUFFER_N_FRAMES,
    "sr": 16000,
    "chunk_sz": DEFAULT_STREAM_CHUNK_SZ,
    "assistant": False,
    "assistant_keywords": ["computer"],
    "debug": False,
}
# stream:
#   - name: Resample
#     partial: true
#     args:
#       resample_sr: 16000
DEFAULT_STREAM_TRANSFORMS = [
    {"name": "Resample", "partial": True, "args": {"resample_sr": 16000}}
]

# PyTorch
TORCH_NUM_CPU_THREADS = 4

# example audio
WAVS = [
    # en
    "./assets/samples/common_voice_en_22738408.wav",

    # de
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
ALIASES = {
    "de": "libreasr/de-1.1.0",
    "en": "libreasr/en-1.1.0",
}

# ${lang}-${release}
DOWNLOADS = {
    "libreasr/de-1.1.0": {
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
            "wandb": "20cirnro",
        },
        "lm.pth": {
            "storage": "gdrive",
            "id": "1Ba74YKDwx4qusRS1SMnKrz-SQr5Ity3p",
            "sha256": "e010207c6ec871dc310c27432188e38a6ac983837903e96b6c974d956d1acf49",
        },
    },
    "libreasr/en-1.1.0": {
        "config.yaml": {
            "storage": "gdrive",
            "id": "1pOQOgKjlv70PIN-gTSk50GBrN89sS5V1",
            "sha256": "b351afde976381a96d2a346b7b0ba94b097129f7b845ddea480a26941bb59cc5",
        },
        "tokenizer.yttm-model": {
            "storage": "gdrive",
            "id": "1Njjp75rjfS341oKhw9Ale0HA4z8E9otR",
            "sha256": "f11dcff85225eaf4ef5c590ba7169f491fc31cde024989d8be36f8b3af1aca1e",
        },
        "model.pth": {
            "storage": "gdrive",
            "id": "1RPrKkdoOLS55Sa_dWtT87FGRGHms6Lha",
            "sha256": "0ba28243783453f5039edb95862b9ee023b5b08a9ae473d5426340dc85bbe2a4",
            "wandb": "3eaqlb1s",
        },
    },
}

# aliases for model sources
SOURCES = {
    "lasr": "LibreASR",
    "hf": "Hugging Face",
    "sb": "Speech Brain",
}
SOURCE_TO_MODULE = {
    "lasr": "LibreASRInstance",
    "hf": "HuggingFaceInstance",
    "sb": "SpeechBrainInstance",
}


# pretrained models
#  from LibreASR
LASR_MODELS = {
    "de": {
        "id": "libreasr/de-1.1.0",
        "tested-on": "common-voice-de-test",
        "wer": -1.0,
        "stream": True,
    },
    "en": {
        "id": "libreasr/en-1.1.0",
        "tested-on": "common-voice-en-valid",
        "wer": 36.6,
        "stream": True,
    },
}
LASR_LANG_TO_MODEL = {k: v["id"] for k, v in LASR_MODELS.items()}

# pretrained models
#  from huggingface
HF_MODELS = {
    "en": {
        "id": "facebook/wav2vec2-large-960h-lv60-self",
        "tested-on": "librispeech-other",
        "wer": 3.9,
        "stream": False,
    },
    "de": {
        "id": "facebook/wav2vec2-large-xlsr-53-german",
        "tested-on": "common-voice-de-test",
        "wer": 18.5,
        "stream": False,
    },
    "fr": {
        "id": "facebook/wav2vec2-large-xlsr-53-french",
        "tested-on": "common-voice-fr-test",
        "wer": 25.2,
        "stream": False,
    },
    "es": {
        "id": "facebook/wav2vec2-large-xlsr-53-spanish",
        "tested-on": "common-voice-es-test",
        "wer": 17.6,
        "stream": False,
    },
    "it": {
        "id": "facebook/wav2vec2-large-xlsr-53-italian",
        "tested-on": "common-voice-it-test",
        "wer": 22.1,
        "stream": False,
    },
}
HF_LANG_TO_MODEL = {k: v["id"] for k, v in HF_MODELS.items()}

# pretrained models
#  from Speech Brain
SB_MODELS = {}
SB_LANG_TO_MODEL = {k: v["id"] for k, v in SB_MODELS.items()}

# all usable models
#  within LibreASR
#  {
#    "source": {
#      "lang-code": {
#        "id": "model-id",
#        "tested-on": "common-voice"
#        "wer": 14.5,
#        "stream": False,
#      }
#    }
#  }
MODELS = {
    "lasr": LASR_MODELS,
    "hf": HF_MODELS,
    "sb": SB_MODELS,
}

# all available language
#  codes
LANGUAGES = set(
    [
        *list(LASR_MODELS.keys()),
        *list(HF_MODELS.keys()),
        *list(SB_MODELS.keys()),
    ]
)

# all available model ids
MODEL_IDS = set(
    [
        *[x["id"] for x in LASR_MODELS.values()],
        *[x["id"] for x in HF_MODELS.values()],
        *[x["id"] for x in SB_MODELS.values()],
    ]
)

# lang-code -> model_id
LANG_TO_MODEL_ID = {**SB_LANG_TO_MODEL}
LANG_TO_MODEL_ID.update(HF_LANG_TO_MODEL)
LANG_TO_MODEL_ID.update(LASR_LANG_TO_MODEL)

# map each model id
#  to its source
def model_id_to_module(model_id):
    assert model_id in MODEL_IDS
    for s in SOURCES.keys():
        for l in MODELS[s].keys():
            _id = MODELS[s][l]["id"]
            if _id == model_id:
                return SOURCE_TO_MODULE[s]
