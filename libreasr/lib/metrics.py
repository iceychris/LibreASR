import editdistance
from jiwer import wer as _wer


def cer(_pred, _true, norm=True):
    """
    Computes the Character Error Rate using the `editdistance` library.

    Parameters
    ----------
    _pred : str
        space-separated sentence (prediction)
    _true : str
        space-separated sentence (ground truth)
    norm : bool
        divide by length of ground truth
    """
    _pred, _true, = _pred.replace(" ", ""), _true.replace(" ", "")
    if norm:
        l = len(_true) if len(_true) > 0 else 1
        return float(editdistance.distance(_pred, _true)) / l
    else:
        return float(editdistance.distance(_pred, _true))


def wer(_pred, _true, norm=False, **kwargs):
    """
    Computes the Word Error Rate using the `jiwer` library.

    Parameters
    ----------
    _pred : str
        space-separated sentence (prediction)
    _true : str
        space-separated sentence (ground truth)
    norm : bool
        divide by length of ground truth
    """
    if norm:
        splitted = _true.split(" ")
        l = len(splitted) if len(splitted) > 0 else 1
        return float(_wer(_true, _pred, **kwargs)) / l
    else:
        return float(_wer(_true, _pred, **kwargs))
