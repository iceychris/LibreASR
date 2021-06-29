import unittest
import time

import torch
import torchaudio

from libreasr import LibreASR
from libreasr.lib.defaults import WAVS, LABELS


class TestTranscribe(unittest.TestCase):
    """
    Note: these tests do not work properly
    when models are quantized (non-deterministic...)
    """

    def setUp(self):
        self.l = LibreASR(auto=True)

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

    """
    def test_unbatched_equals_batched(self):
        a = self.l.transcribe(WAVS, batch=False)
        b = self.l.transcribe(WAVS, batch=True)
        self.assertEqual(a, b, msg="Outputs for unbatched and batched should match")
    """

    def test_stream(self):
        f = WAVS[0]
        aud, sr = torchaudio.load(f)

        def gen_fn():
            step = int(sr * 0.08 * 1)
            for i in range(0, aud.size(1), step):
                inp = aud[:, i : i + step]
                pad = torch.zeros(aud.size(0), step - inp.size(1))
                inp = torch.cat([inp, pad], dim=1)
                yield inp

        kwargs = {
            "beam_search_opts": {
                "beam_width": 2,
                "topk_next": 2,
            }
        }
        l = self.l
        astream = l.stream(f, **kwargs)
        bstream = l.stream(gen_fn, sr=sr, **kwargs)
        cstream = l.stream(gen_fn(), sr=sr, **kwargs)
        self.assertTrue(isinstance(astream, str))
        for yb in bstream:
            pass
        for yc in cstream:
            pass


if __name__ == "__main__":
    unittest.main()
