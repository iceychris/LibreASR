import torch
import torch.quantization
import torch.nn as nn
import torch.nn.functional as F

from libreasr.lib.utils import standardize
from libreasr.lib.quantization import maybe_post_quantize


ALPHA = 0.1
THETA = 1.0
MIN_VAL = -10.0

DEBUG = False


class LM(nn.Module):
    def __init__(self, vocab_sz, embed_sz, hidden_sz, num_layers, p=0.2, **kwargs):
        super(LM, self).__init__()
        self.embed = nn.Embedding(vocab_sz, embed_sz, padding_idx=0)
        self.rnn = nn.LSTM(embed_sz, hidden_sz, batch_first=True, num_layers=num_layers)
        self.drop = nn.Dropout(p)
        self.linear = nn.Linear(hidden_sz, vocab_sz)
        if embed_sz == hidden_sz:
            # tie weights
            self.linear.weight = self.embed.weight

    def forward(self, x, state=None):
        x = self.embed(x)
        if state:
            x, state = self.rnn(x, state)
        else:
            x, state = self.rnn(x)
        x = self.drop(x)
        x = self.linear(x)
        x = F.log_softmax(x, dim=-1)
        return x, state


class LMFuser:
    def __init__(self, lm):
        self.lm = lm
        self.lm_logits = None
        self.lm_state = None
        self.has_lm = self.lm is not None

    def advance(self, y_one_char):
        if self.has_lm:
            self.lm_logits, self.lm_state = self.lm(y_one_char, self.lm_state)
            standardize(self.lm_logits)
            self.lm_logits[:, :, 0] = MIN_VAL

    def fuse(self, joint_out, prob, pred, alpha=ALPHA, theta=THETA):
        lm_logits = self.lm_logits
        if self.has_lm and torch.is_tensor(lm_logits):
            standardize(joint_out)
            joint_out[:, :, :, 0] = MIN_VAL
            if DEBUG:
                print(
                    "lm:",
                    lm_logits.shape,
                    lm_logits.mean(),
                    lm_logits.std(),
                    lm_logits.max(),
                )
                print(
                    "joint:",
                    joint_out.shape,
                    joint_out.mean(),
                    joint_out.std(),
                    joint_out.max(),
                )
            fused = alpha * lm_logits + theta * joint_out
            prob, pred = fused.max(-1)
            return fused, prob, pred
        return joint_out, prob, pred

    def reset(self):
        self.lm_logits = None
        self.lm_state = None


def load_lm(conf, lang):

    # create model
    lm = LM(**conf["lm"])
    lm.eval()

    # load model
    lm.load_state_dict(torch.load(conf["lm"]["path"]))
    lm.eval()

    # quantize
    lm = maybe_post_quantize(lm)
    lm.eval()

    return lm
