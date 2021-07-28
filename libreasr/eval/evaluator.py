import abc


class Evaluator(abc.ABC):
    def __init__(self):
        """
        Constructor, load datasets or similar stuff
        """

    def eval(self, learn, ddp=False, save_fn=lambda x: None, **kwargs):
        """
        Receives the Learner `learn` to use for evaluation.
        `save_fn` should be called
        whenever the target metric has improved.
        `ddp` is indicating if `DistributedDataParallel` is used.
        This should return one or multiple dicts of different metrics.
        """
        yield {}
