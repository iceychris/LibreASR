"""
LibreASR source code
"""
import gc
import math

import numpy as np

from libreasr.lib.defaults import DEFAULT_CONFIG_PATH
from libreasr.lib.inference.utils import load_config, get_available_models
from libreasr.lib.instance import LibreASRTraining, LibreASRInference


class LibreASR:
    def __init__(self, lang=None, config_path=None, **kwargs):
        """
        Create a new LibreASR instace for a specific language
        """
        self.lang = lang
        self.config_path = config_path
        self.kwargs = kwargs
        self.inst = None
        self.mode = None

    def _load_eval(self, pcent):
        def training_hook(conf):
            conf["cuda"]["enable"] = False
            conf["pcent"]["valid"] = pcent
            conf["apply_limits"] = False
            conf["apply_x_limits"] = False
            conf["apply_y_limits"] = False
            conf["suffix"] = ""
            conf["model"]["load"] = False
            conf["lm"]["enable"] = False

        lt = LibreASR(
            lang=None, config_path=self.config_path, config_hook=training_hook
        ).load_training()
        li = LibreASR(lang=self.lang, config_path=None).load_inference()
        return lt, li

    def _collect_garbage(self, ok, new_mode):
        if ok and self.mode != new_mode:
            if self.inst is not None:
                self.inst = None
                self.mode = None
                gc.collect()
                print("[LibreASR] garbage collected")
            return True
        return False

    def load_training(self, do_gc=True):
        m = "training"
        if self._collect_garbage(do_gc, m):
            self.inst = LibreASRTraining(self.lang, self.config_path, **self.kwargs)
            self.mode = m
        return self

    def load_inference(self, do_gc=True):
        m = "inference"
        if self._collect_garbage(do_gc, m):
            self.inst = LibreASRInference(self.lang, self.config_path, **self.kwargs)
            self.mode = m
        return self

    def transcribe(self, sth, batch=True, load=True, **kwargs):
        """
        Transcribe files or tensors
        """
        if load:
            self.load_inference()
        return self.inst.transcribe(sth, batch=batch, **kwargs)

    def stream(self, sth, load=True, **kwargs):
        """
        Transcribe stuff in a stream
        """
        if load:
            self.load_inference()
        return self.inst.stream(sth, **kwargs)

    def train(self):
        """
        Train a model
        """
        self.load_training()

    def validate(self, pcent=1.0):
        """
        Validate a saved model.
        This uses the `valid` dataset.
        """
        assert self.lang is not None
        assert self.config_path is not None

        instances = self._load_eval(pcent=pcent)
        stuff = _run_transcribe(*instances)
        metrics = _calculate_metrics(*stuff)
        return metrics

    def test(self, pcent=1.0):
        """
        Test a saved model.
        This uses the `test` dataset.
        """
        self.load_inference()

    def serve(self, port):
        """
        Run the LibreASR API-Server
        """
        pass

    def get_grpc_port(self):
        assert self.mode is not None
        return self.inst.conf.get("grpc_port", 50051)

    def available_models(self):
        return get_available_models()


def _run_transcribe(lt, li):
    df = lt.inst.builder_valid.df
    files = df.file
    labels = df.label
    transcripts = li.transcribe(files.tolist())
    return files, transcripts, labels


def _calculate_metrics(files, transcripts, labels):
    from libreasr.lib.metrics import wer, cer
    from libreasr.lib.utils import sanitize_str

    cers, wers = [], []
    for f, t, l in zip(files, transcripts, labels):
        l = sanitize_str(l.decode("utf-8"))
        _cer = cer(t, l)
        _wer = wer(t, l)
        cers.append(_cer)
        wers.append(_wer)
    return {
        "cer": np.array(cers).mean(),
        "wer": np.array(wers).mean(),
    }
