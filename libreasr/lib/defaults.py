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
    "implementation": "speechbrain",
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
    "es": "libreasr/es-1.1.0",
    "en": "libreasr/en-1.1.0",
}

# ${lang}-${release}
DOWNLOADS = {
    "libreasr/de-1.1.0": {
        "config.yaml": {
            "storage": "gdrive",
            "src": "14vqVD-CdEKyyKUQwWwDfxY_BsbaIoHo4",
            "sha256": "e8d8654e85bdc3d6e42f032e43c1d646f0a3c9661358ce86859ad149f3e7f8b8",
        },
        "tokenizer.yttm-model": {
            "storage": "gdrive",
            "src": "1onxiXkfQZgZbI8JwmRirAAY68N0_4FIU",
            "sha256": "42964d4ed5a0725fd6e410c2bc71607f6ad77d9ad38031cb3c742b119812ad7f",
        },
        "model.pth": {
            "storage": "gdrive",
            "src": "1BaCjwjXHmoxqiWjN-310zRRKQX1ngWaP",
            "sha256": "d0c6c4467be6644d81b99a2349aec5ea8490472ea0926528bdf5dfb6118714dd",
            "wandb": "20cirnro",
            "license": "CC BY-NC-SA 4.0",
        },
    },
    "libreasr/es-1.1.0": {
        "config.yaml": {
            "storage": "gdrive",
            "src": "1ZDn2p-OTIYMF-nziQwmUSZjO2G7MLKep",
            "sha256": "1b10535a708f10ea3954d6192cbc5741ab22d5b4b75fb39cd7145317bb0b39aa",
        },
        "tokenizer.yttm-model": {
            "storage": "gdrive",
            "src": "16rTNjxWt0XoJ-97tv-89aw4VJLadtxCF",
            "sha256": "253c02e76a7a255e518cfe3366ff08f3205e3d0bf53a7ab0c28ebd498aec2373",
        },
        "model.pth": {
            "storage": "gdrive",
            "src": "141-rLdKhqtUpA7PJXqNlOml8PVFVSs3C",
            "sha256": "6ea3dca8024e952b9c6cbde33cc3e2610b272e6aced1bed107a28c7e9d65a7d3",
            "wandb": "3jjv7tem",
            "license": "CC BY-NC-SA 4.0",
        },
    },
    "libreasr/en-1.1.0": {
        "config.yaml": {
            "storage": "gdrive",
            "src": "1heDhgChjU1OnSTWE5DNJNTtvfpzNilcn",
            "sha256": "b351afde976381a96d2a346b7b0ba94b097129f7b845ddea480a26941bb59cc5",
        },
        "tokenizer.yttm-model": {
            "storage": "gdrive",
            "src": "1nXsnvGqjDetqwtDNpFZNzuc9nnRq8Fmh",
            "sha256": "f11dcff85225eaf4ef5c590ba7169f491fc31cde024989d8be36f8b3af1aca1e",
        },
        "model.pth": {
            "storage": "gdrive",
            "src": "10fIY6ijpZSUQdy9b5HsMrp8aKZujeHmw",
            "sha256": "0ba28243783453f5039edb95862b9ee023b5b08a9ae473d5426340dc85bbe2a4",
            "wandb": "3eaqlb1s",
            "license": "CC BY-NC-SA 4.0",
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
    "es": {
        "id": "libreasr/es-1.1.0",
        "tested-on": "common-voice-es-valid",
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
