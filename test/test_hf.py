import unittest
import time

import torch
import torchaudio

from libreasr import LibreASR
from libreasr.lib.defaults import WAVS, LABELS


class TestHuggingFace(unittest.TestCase):
    """
    Check if transcribing files using
    LibreASR's Hugging Face wrapper is working
    properly.
    """

    def setUp(self):
        self.l = LibreASR.from_huggingface("de")

    def test_transcribe_one(self):
        self.l.transcribe(WAVS[0])

    def test_transcribe_multi(self):
        self.l.transcribe(WAVS[:2])


if __name__ == "__main__":
    unittest.main()
