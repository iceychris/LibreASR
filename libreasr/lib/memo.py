from functools import partial

# pypi: lru-dict
from lru import LRU


def mk_memo(func, sz=16, initial={}):
    """
    Memoize a function `func` by hashing its input
    and constructing a Least-Recently-Used Cache
    of size `sz`.
    `func` gets passed `cache` which
    may be used like a regular dictionary.
    `cache` will get filled by `initial`.
    """
    cache = LRU(sz)
    for k, v in initial.items():
        cache[k] = v
    func = partial(func, cache=cache)
    return func
