import unittest
import time

import torch
import torchaudio

from libreasr import LibreASR

WAVS = [
    "./assets/samples/common_voice_de_17672459.wav",
    "./assets/samples/common_voice_de_18227443.wav",
    "./assets/samples/common_voice_de_18520948.wav",
    "./assets/samples/common_voice_de_17516889.wav",
    "./assets/samples/common_voice_de_18818000.wav",
]
LABELS = [
    "als die stadtmauern errichtet wurden hieß es dass sie unbezwingbar seien",
    "insbesondere keine topflappen",
    "durch einen elektrisierten weidezaun ist die koppel begrenzt",
    "die beamten gehen nun verstärkt gegen illegale straßenrennen vor",
    "frau senninger aus dem zweiten stock hat bei einem sturz einen oberschenkelhalsbruch erlitten",
]


class TestTranscribe(unittest.TestCase):
    """
    Note: these tests do not work properly
    when models are quantized (non-deterministic...)
    """

    def setUp(self):
        self.l = LibreASR("de")
        self.l.load_inference()

    def test_unbatched(self):
        res = self.l.transcribe(WAVS, batch=False)
        self.assertEqual(len(WAVS), len(res))
        self.assertTrue(all([isinstance(x, str) for x in res]))
        [print(" >b1< " + x) for x in res]

    def test_batched(self):
        res = self.l.transcribe(WAVS, batch=True)
        self.assertEqual(len(WAVS), len(res))
        self.assertTrue(all([isinstance(x, str) for x in res]))
        [print(" >bX< " + x) for x in res]

    def test_unbatched_equals_batched(self):
        a = self.l.transcribe(WAVS, batch=False)
        b = self.l.transcribe(WAVS, batch=True)
        self.assertEqual(a, b, msg="Outputs for unbatched and batched should match")

    def test_unbatched_equals_batched_lm(self):
        a = self.l.transcribe(WAVS, batch=False)
        b = self.l.transcribe(WAVS, batch=True)
        self.assertEqual(a, b, msg="Outputs for unbatched and batched should match")

    def test_stream(self):
        aud, sr = torchaudio.load(WAVS[0])

        def gen_fn():
            step = int(sr * 0.08 * 1)
            for i in range(0, aud.size(1), step):
                inp = aud[:, i : i + step]
                yield inp

        l = self.l
        astream, bstream = l.stream(gen_fn, sr=sr), l.stream(gen_fn(), sr=sr)
        for _, ya in astream:
            print(ya)
            self.assertTrue(isinstance(ya, str))
        for _, yb in bstream:
            self.assertTrue(isinstance(yb, str))


if __name__ == "__main__":
    unittest.main()
