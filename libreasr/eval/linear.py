import tqdm
import torch
import numpy as np

from sklearn import linear_model

from libreasr.eval.evaluator import Evaluator
from libreasr.data.basic import grab_data


class FeatureExtractor(torch.nn.Module):
    def __init__(self, preprocessor, encoder, **kwargs):
        super().__init__()
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.kwargs = kwargs

    def forward(self, x, xl):
        while x.dim() < 4:
            x = x.unsqueeze(-1)
        x, xl = self.preprocessor(x, xl, **self.kwargs)
        x = self.encoder(x, lengths=xl)
        return x, None


def convert(model):
    return FeatureExtractor(model.preprocessor, model.encoder, no_augmentation=True)


def linear_evaluation(
    model, dl_train, dl_valid, num_samples=800, C=0.316, device="cpu"
):
    """
    Evaluate a `model` (e.g. an audio encoder)
    by training a classifier (`LogisticRegression`) on
    `dl_train`.
    The accuracy of this classifier on `dl_valid` is returned.
    """

    # abbreviations
    n = num_samples
    t = 16000  # 81
    t_dim = -1
    dev = device

    # compute features
    # train
    x_train = []
    y_train = []
    for i, item in tqdm.tqdm(enumerate(dl_train), total=n):
        inp = item["samples"][0]
        if i >= n:
            break
        if inp.size(t_dim) != t:
            continue

        # move to device
        x, xl = inp.to(dev), torch.LongTensor([t]).to(dev)

        # infer
        with torch.no_grad():
            enc, _ = model(x, xl)
        x_train.append(enc.reshape(-1).cpu().numpy())
        y_train.append(item["labels"][0].item())

    # valid
    x_valid_inp = []
    x_valid_raw = []
    x_valid = []
    y_valid = []
    for i, item in tqdm.tqdm(enumerate(dl_valid), total=n):
        inp = item["samples"][0]
        if i >= n:
            break
        if inp.size(t_dim) != t:
            continue

        # move to device
        x, xl = inp.to(dev), torch.LongTensor([t]).to(dev)

        # infer
        with torch.no_grad():
            enc, _ = model(x, xl)
        x_valid_inp.append(x.cpu().numpy())
        x_valid_raw.append(enc.cpu().numpy())
        x_valid.append(enc.reshape(-1).cpu().numpy())
        y_valid.append(item["labels"][0].item())

    # fit classifier
    print(
        f"[eval] Training LogisticRegression on {len(x_train)} samples, validating on {len(x_valid)} samples..."
    )
    reg = linear_model.LogisticRegression(random_state=0, C=C, max_iter=10000)
    reg.fit(x_train, y_train)
    reg.coef_.shape

    # Evaluate using the logistic regression classifier
    predictions = reg.predict(x_valid)
    res = LinearEvaluationResult(
        x_train, y_train, x_valid_inp, x_valid_raw, x_valid, y_valid, predictions
    )
    print(f"[eval] x: {x_valid[0].shape}, y: {y_valid[:3]}, pred: {predictions[:3]}")
    print(f"[eval] Accuracy: {res.accuracy:.3f}%")
    return res


class LinearEvaluationResult:
    def __init__(self, xt, yt, xvi, xvr, xv, yv, pred):
        self.xt = xt
        self.yt = yt
        self.xvi = xvi
        self.xvr = xvr
        self.xv = xv
        self.yv = yv
        self.pred = pred

    @property
    def accuracy(self):
        return np.mean((self.yv == self.pred).astype(np.float)) * 100.0

    def plot(self, n=3, b=0, sr=16000, fsz=64):
        from IPython.core.display import display
        from IPython.display import Audio
        import matplotlib.pyplot as plt

        for i in range(n):
            xi, xr, y, p = self.xvi[i], self.xvr[i], self.yv[i], self.pred[i]
            if xr.ndim == 3:
                xr = xr[b]
            xr = xr[..., :fsz]
            plt.imshow(xr)
            plt.title("xr")
            plt.show()
            display(Audio(xi, rate=sr))
        pass


class LinearEvaluator(Evaluator):
    def __init__(self, convert_fn=convert, device="cpu"):
        self.convert_fn = convert_fn
        self.device = device
        self.t, self.v = grab_data()
        self.best = float("inf")

    def eval(self, learn, ddp=False, save_fn=lambda x: None, num_samples=800, **kwargs):

        # first, convert the model
        #  to an appropriate feature extractor
        model = self.convert_fn(learn.model)

        # run linear eval
        res = linear_evaluation(
            model, self.t, self.v, device=self.device, num_samples=num_samples
        )
        acc = res.accuracy

        # save if best so far
        if acc > self.best:
            save_fn(f"linear-eval-{acc:.2f}acc")
        self.best = acc

        yield {
            "metrics/accuracy": acc,
        }
