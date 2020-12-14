import torch

import string

import youtokentome as yttm

# tasks
# - normal: just normal STT
# - fill: mark start and end tokens, rest is space
# - words: mark words with w
TASK = "normal"


class Language:
    def __init__(self, tokens):
        self.token_list = list(tokens.keys())
        self.idx_list = list(tokens.values())
        self.t2i = tokens
        self.i2t = {i: t for (t, i) in tokens.items()}

    def numericalize(self, text, sos=False):
        text = text.lower()
        text = text.strip()
        text = text.replace(self.SOS, "")
        text = text.replace(self.EOS, "")
        if sos:
            nummed = [self.iSOS]
        else:
            nummed = []
        if TASK == "fill":
            sp = text.split(" ")
            newtxt = []
            for word in sp:
                newtxt.append("x" * len(word))
            text = " ".join(newtxt)
        elif TASK == "words":
            sp = text.split(" ")
            newtxt = []
            for word in sp:
                newtxt.append("x")
            text = " ".join(newtxt)
        for c in text:
            try:
                nummed.append(self.get_idx(c))
            except:
                continue
        return nummed + [self.iEOS]

    def denumericalize(self, nummed, strip_zeros=True):
        text = ""
        if not isinstance(nummed, list):
            nummed = [nummed]
        nummed = list(filter(lambda x: x != self.iSOS or x != self.iEOS, nummed))
        for n in nummed:
            try:
                text += self.get_token(n, strip_zeros=strip_zeros)
            except:
                continue
        return text

    def get_idx(self, tok):
        return self.t2i[tok]

    def get_token(self, num, strip_zeros=False):
        if strip_zeros:
            if num == 0:
                return ""
        if isinstance(num, list):
            num = num[0]
        return self.i2t[num]

    @property
    def SOS(self):
        return self.token_list[1]

    @property
    def EOS(self):
        return self.token_list[2]

    @property
    def iSOS(self):
        return self.idx_list[1]

    @property
    def iEOS(self):
        return self.idx_list[2]

    @property
    def replaceable(self):
        l = list(self.t2i.values())
        return l[11:]

    def randomize(self, t, p):
        x = t.clone()
        rpl = self.replaceable
        mask = torch.zeros(t.shape).uniform_() > p
        vals = torch.randint(low=min(rpl), high=max(rpl) + 1, size=t.shape)
        return torch.where(mask, x, vals)

    def __repr__(self):
        toks = list(self.t2i.keys())
        return str((toks[:5], "...", toks[-5:], len(self)))

    def __len__(self):
        return len(self.t2i)

    def __getitem__(self, idx):
        self.get_token(idx)


class TokenizedLanguage(Language):
    def __init__(self, *args, model_file="tmp/tokenizer.yttm-model", **kwargs):
        super().__init__(*args, **kwargs)
        self.mf = model_file

        # load tokenizer
        self.tokenizer = yttm.BPE(model=model_file)

    def numericalize(self, text, sos=False, dropout=0):
        text = text.lower()
        text = text.strip()
        text = text.replace(self.SOS, "")
        text = text.replace(self.EOS, "")
        res = self.tokenizer.encode(
            [text], output_type=yttm.OutputType.ID, dropout_prob=dropout
        )
        # unpack
        res = res[0]
        return res

    def denumericalize(self, nummed, strip_zeros=True):
        text = ""
        if not isinstance(nummed, list):
            nummed = [nummed]
        res = self.tokenizer.decode([nummed], ignore_ids=[0])
        # unpack
        res = res[0]
        return res

    def get_idx(self, tok):
        return self.numericalize(tok)[0]

    def get_token(self, num, strip_zeros=False):
        return self.denumericalize(num)[0]

    def __len__(self):
        return self.tokenizer.vocab_size()

    def __repr__(self):
        bpe = self.tokenizer
        return str((bpe.vocab()[:5], "...", bpe.vocab()[-5:], len(self)))


def get_language(
    tokens=["<BLK>", "<s>", "</s>", "<UNK>", " ", ".", "!", "?", ",", "'", "-"],
    cls=TokenizedLanguage,
    **kwargs
):

    # create dictionary
    tokens = dict(zip(tokens, range(len(tokens))))
    possible_chars = string.ascii_lowercase  # + string.digits#  + string.punctuation
    tokens.update(
        {_char: len(tokens) + _idx for (_idx, _char) in enumerate(possible_chars)}
    )

    # create a language
    lang = cls(tokens, **kwargs)
    vocab_sz = len(lang)

    return lang, vocab_sz


def get_tokenizer(*args, **kwargs):
    return get_language(*args, **kwargs)
