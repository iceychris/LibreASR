import string
import pickle

import torch

import youtokentome as yttm

# tasks
# - normal: just normal STT
# - fill: mark start and end tokens, rest is space
# - words: mark words with w
TASK = "normal"

# other
TOKENIZER_MODEL_FILE = "tmp/tokenizer.yttm-model"
CORPUS_FILE = "tmp/corpus.txt"


def noop(ret=list):
    def inner(*args, **kwargs):
        return ret()


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


class BPETokenizer(Language):
    def __init__(
        self,
        tokens,
        data_fn=noop(),
        model_file="tmp/tokenizer.yttm-model",
        vocab_sz=4096,
        **kwargs,
    ):
        super().__init__(tokens, **kwargs)
        self.data_fn = data_fn
        self.mf = model_file
        self.vocab_sz = vocab_sz

        # load tokenizer
        try:
            self.tokenizer = yttm.BPE(model=model_file)
            print("[load]", model_file)
        except:
            self.tokenizer = self.train()
            print("[create]", model_file)

    def train(
        self,
        corpus_file=CORPUS_FILE,
        dump_labels=True,
    ):
        import youtokentome as yttm

        # grab settings
        vocab_sz = self.vocab_sz
        model_file = self.mf

        # first we need to dump labels
        if dump_labels:
            self.data_fn(as_list=False, to_file=corpus_file)

        # train model
        print("Training yttm model...")
        yttm.BPE.train(data=corpus_file, vocab_size=vocab_sz, model=model_file)
        print("Done.")

        # load model (for testing)
        print("Testing yttm model...")
        bpe = yttm.BPE(model=model_file)
        # Two types of tokenization
        test_text = "Are you freakin' crazy?"
        encoded1 = bpe.encode([test_text], output_type=yttm.OutputType.ID)
        encoded2 = bpe.encode([test_text], output_type=yttm.OutputType.SUBWORD)
        decoded = bpe.decode(encoded1)
        print(encoded1)
        print(encoded2)
        print(decoded)
        return bpe

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


class CharTokenizer(Language):
    def __init__(
        self,
        tokens,
        data_fn=noop(),
        model_file="tmp/tokenizer.yttm-model",
        vocab_sz=32,
        **kwargs,
    ):
        super().__init__(tokens, **kwargs)
        self.mf = model_file

        # load tokenizer
        try:
            self.vocab = pickle.load(open(model_file, "rb"))
            print("[load]", model_file, len(self.vocab))
            print(self.vocab)
        except:
            data = data_fn()
            assert not len(data) == 0
            text = "".join(data)
            vocab = set(text)
            m = max(tokens.values())
            for i, c in enumerate(vocab):
                tokens[c] = m + i + 1
            self.vocab = tokens
            pickle.dump(self.vocab, open(model_file, "wb"))
            print("[create]", model_file, len(self.vocab))
            print(self.vocab)

        # compute inverse
        self.t2i = self.vocab
        self.i2t = {i: t for (t, i) in self.vocab.items()}

    def numericalize(self, text, sos=False, dropout=0):
        res = []
        for c in text:
            res.append(self.t2i.get(c, None))
        res = list(filter(lambda x: x is not None, res))
        return res

    def denumericalize(self, nummed, strip_zeros=True):
        res = []
        if not isinstance(nummed, list):
            nummed = [nummed]
        for n in nummed:
            if strip_zeros and n == 0:
                continue
            res.append(self.i2t.get(n, None))
        res = list(filter(lambda x: x is not None, res))
        return "".join(res)

    def get_idx(self, tok):
        return self.numericalize(tok)[0]

    def get_token(self, num, strip_zeros=False):
        return self.denumericalize(num)[0]

    def __len__(self):
        return len(self.vocab)

    def __repr__(self):
        return f"<CharTokenizer {str(self.vocab)[:32]}...>"


def get_language(
    tokens=["<BLK>", "<s>", "<BOS>", "<UNK>"], type="bpe", data_fn=noop(), **kwargs
):

    # create dictionary
    tokens = dict(zip(tokens, range(len(tokens))))

    # select class
    if type == "bpe":
        cls = BPETokenizer
    elif type == "char":
        cls = CharTokenizer
    else:
        raise Exception(f"No tokenizer '{type}' implemented")

    # create a language
    lang = cls(tokens, data_fn, **kwargs)
    vocab_sz = len(lang)

    return lang, vocab_sz


def get_tokenizer(*args, **kwargs):
    return get_language(*args, **kwargs)
