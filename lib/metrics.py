import editdistance
from jiwer import wer as _wer


def cer(_pred, _true, norm=True):
    """
    Computes the Character Error Rate, defined as the edit distance.
    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """
    _pred, _true, = _pred.replace(" ", ""), _true.replace(" ", "")
    if norm:
        l = len(_true) if len(_true) > 0 else 1
        return float(editdistance.distance(_pred, _true)) / l
    else:
        return float(editdistance.distance(_pred, _true))


def wer(_pred, _true, norm=False, **kwargs):
    if norm:
        splitted = _true.split(" ")
        l = len(splitted) if len(splitted) > 0 else 1
        return float(_wer(_true, _pred, **kwargs)) / l
    else:
        return float(_wer(_true, _pred, **kwargs))
