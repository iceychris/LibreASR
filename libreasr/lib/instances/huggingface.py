import pathlib

import torch
from torch import nn
import torchaudio

from libreasr.lib.defaults import HF_LANG_TO_MODEL
from libreasr.lib.utils import warn_about_license
from libreasr.lib.instances import BaseInstance


class HuggingFaceInstance(BaseInstance):
    def __init__(self, model_name):
        from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC

        # resolve model name
        if model_name in HF_LANG_TO_MODEL:
            model_name = HF_LANG_TO_MODEL[model_name]
        self.model_name = model_name

        warn_about_license(
            "HuggingFaceInstance",
            "external speech recognition model",
            f"https://huggingface.co/{model_name}",
        )

        # load model and tokenizer
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

    def transcribe(self, sth, batch=True, **kwargs):
        # listify
        speech = []
        only_one = False
        if torch.is_tensor(sth) or isinstance(sth, (str, pathlib.Path)):
            only_one = True
            sth = [sth]

        # grab first channel
        for f in sth:
            if torch.is_tensor(f):
                audio = f.numpy()
            else:
                audio, _ = torchaudio.load(f)
                audio = audio[0].numpy()
            speech.append(audio)

        # prepare for model
        inp = self.tokenizer(
            speech, return_tensors="pt", padding="longest"
        ).input_values

        # retrieve logits
        logits = self.model(inp).logits

        # take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.tokenizer.batch_decode(predicted_ids)
        transcription = [x.lower() for x in transcription]

        # return transcript
        if only_one:
            return transcription[0]
        return transcription

    def stream(self, sth, batch=False, **kwargs):
        raise NotImplementedError("Streaming not implement for HuggingFace models.")
