"""
[Unused] incomplete implementation of a CTC Decoder.
"""

from itertools import groupby
from operator import itemgetter

import torch
import torch.nn as nn
import torch.nn.functional as F


def remove_duplicates(l):
    return list(map(itemgetter(0), groupby(l)))


def remove_blanks(l, blank=0):
    return list(filter(lambda x: x != blank, l))


def ctc_decode_greedy(acts, denumericalize_func, blank=0):
    """
    acts: output activations of the model (shape [N x T x V] or [T x V])
    blank: the blank symbol
    returns: a list of denumericalized items
    """
    if len(acts.shape) == 2:
        acts = acts[None]

    results = []
    for batch in acts:
        # batch is of shape [T x V]

        # greedy
        idxes = batch.argmax(dim=-1).cpu().numpy().tolist()

        # decode
        idxes = remove_duplicates(idxes)
        idxes = remove_blanks(idxes, blank=blank)

        # denumericalize
        results.append(denumericalize_func(idxes))

    if len(results) == 1:
        return results[0]
    return results


if __name__ == "__main__":

    print("ctc:")
    l = [0, 1, 1, 1, 2, 2, 1, 0, 3, 0, 3]
    print(l)
    l = remove_duplicates(l)
    print(l)
    l = remove_blanks(l)
    print(l)
