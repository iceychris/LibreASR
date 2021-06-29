from abc import ABC


class BaseInstance(ABC):
    def transcribe(self, sth, batch=True, **kwargs):
        raise NotImplementedError()

    def stream(self, sth, batch=False, **kwargs):
        raise NotImplementedError()
