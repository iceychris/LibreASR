import torch
import torch.nn as nn
import torch.nn.functional as F

from libreasr.lib.utils import standardize, try_eval
from libreasr.lib.defaults import LM_ALPHA, LM_THETA, LM_TEMP, LM_DEBUG

from IPython.core.debugger import set_trace


class LM(nn.Module):
    def __init__(self, vocab_sz, embed_sz, hidden_sz, num_layers, p=0.2, **kwargs):
        super(LM, self).__init__()
        self.shape_logits = [1, 1, vocab_sz]
        self.shape_state = [num_layers, 1, hidden_sz]
        self.embed = nn.Embedding(vocab_sz, embed_sz, padding_idx=0)
        self.rnn = nn.LSTM(embed_sz, hidden_sz, batch_first=True, num_layers=num_layers)
        self.drop = nn.Dropout(p)
        self.linear = nn.Linear(hidden_sz, vocab_sz)
        if embed_sz == hidden_sz:
            # tie weights
            self.linear.weight = self.embed.weight

    def forward(self, x, state=None, temp=LM_TEMP, softmax=True, log=True):
        x = self.embed(x)
        if state:
            x, state = self.rnn(x, state)
        else:
            x, state = self.rnn(x)
        x = self.drop(x)
        x = self.linear(x)
        if softmax:
            f = F.softmax
            if log:
                f = F.log_softmax
            x = f(x / temp, dim=-1)
        return x, state

    def generate(self, prefix: str, lang, steps=64):
        device = next(self.parameters()).device
        seq = []
        x = lang.numericalize(prefix)
        seq.extend(x)
        x = torch.LongTensor(x)[None].to(device)
        x, s = self(x)
        x = x.argmax(-1)
        seq.append(x[0][-1].item())
        x = torch.LongTensor([[seq[-1]]]).to(device)
        for _ in range(steps):
            x, s = self(x, s)
            x = x[0][0].argmax(-1)
            seq.append(x.item())
            x = torch.LongTensor([[seq[-1]]]).to(device)
        return lang.denumericalize(seq)


def masked_state_update(mask, state, new_state, device):
    state = list(state)
    for i, (ps, nps) in enumerate(zip(state, new_state)):
        ps, nps = ps.permute(1, 0, 2), nps.permute(1, 0, 2)
        state[i] = torch.where(mask, ps, nps).permute(1, 0, 2)
        if device != torch.device("cpu"):
            state[i] = state[i].contiguous()
    state = tuple(state)
    return state


class LMFuser:
    def __init__(self, lm):
        self.lm = lm
        self.lm_logits = None
        self.lm_state = None
        self.has_lm = self.lm is not None

    def advance(self, y_one_char, temp=LM_TEMP):
        if self.has_lm:
            self.lm_logits, self.lm_state = self.lm(
                y_one_char, self.lm_state, temp=temp, softmax=True, log=False
            )

    def fuse(self, joint_out, prob, pred, alpha=LM_ALPHA, theta=LM_THETA):
        lm_logits = self.lm_logits
        if self.has_lm and torch.is_tensor(lm_logits):
            fused = alpha * lm_logits + theta * joint_out
            fused[..., 0] = 0.0

            # sample
            prob, new_pred = fused.max(-1)
            new_pred[pred == 0] = 0
            pred = new_pred

            return fused, prob, pred
        return joint_out, prob, pred

    def reset(self):
        self.lm_logits = None
        self.lm_state = None


class LMFuserBatch(LMFuser):
    def advance(self, y_one_char, keep_mask, temp=LM_TEMP):
        if self.has_lm:

            # init
            dev = y_one_char.device
            if self.lm_logits is None:
                shl, shs = self.lm.shape_logits, self.lm.shape_state
                shl[0], shs[1] = y_one_char.size(0), y_one_char.size(0)
                self.lm_logits = torch.zeros(shl, device=dev)
                z = torch.zeros(shs, device=dev)
                self.lm_state = (z, z)

            # infer
            logits, state = self.lm(
                y_one_char, self.lm_state, temp=temp, softmax=True, log=False
            )

            # update
            self.lm_logits = torch.where(keep_mask, self.lm_logits, logits)
            self.lm_state = masked_state_update(keep_mask, self.lm_state, state, dev)

    def fuse(self, joint_out, prob, pred, alpha=LM_ALPHA, theta=LM_THETA):
        lm_logits = self.lm_logits
        if self.has_lm and torch.is_tensor(lm_logits):
            if lm_logits.dim() == 3:
                lm_logits = lm_logits.unsqueeze(1)
            fused = alpha * lm_logits + theta * joint_out
            fused[..., 0] = 0.0
            assert joint_out.shape == fused.shape

            # sample
            prob, new_pred = fused.max(-1)
            new_pred[pred == 0] = 0
            pred = new_pred

            return fused, prob, pred
        return joint_out, prob, pred


def load_lm(lm_conf, lm_path, load=False, device="cpu"):

    # create model
    lm = LM(**lm_conf).to(device)
    lm.eval()

    if load:
        # load lm
        lm.load_state_dict(torch.load(lm_path))
        lm = lm.eval()
        print("[load]", lm_path)

    # shove over to device again?
    lm = lm.to(device)

    return lm
