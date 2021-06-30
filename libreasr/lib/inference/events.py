from abc import ABC
from enum import Enum
from typing import List, Callable


class EventTag(Enum):
    START = 1
    STOP = 2
    SILENCE = 3
    HYPOTHESIS = 4
    TRANSCRIPT = 5
    SENTENCE = 6
    IDLE = 7
    WAKEWORD = 8
    ASSISTANT_ANSWER = 9
    TTS = 10


EVENTS_FROM_MODEL = (
    EventTag.START,
    EventTag.IDLE,
    EventTag.HYPOTHESIS,
    EventTag.STOP,
)


EVENTS_FOR_OUTPUT = (
    EventTag.HYPOTHESIS,
    EventTag.TRANSCRIPT,
    EventTag.SENTENCE,
    EventTag.ASSISTANT_ANSWER,
    EventTag.TTS,
)


class InferenceEvent(ABC):
    def __init__(self, tag):
        self.tag = tag


class StartEvent(InferenceEvent):
    def __init__(self, beamer):
        super().__init__(EventTag.START)
        self.beamer = beamer


class IdleEvent(InferenceEvent):
    def __init__(self):
        super().__init__(EventTag.IDLE)


class StopEvent(InferenceEvent):
    def __init__(self):
        super().__init__(EventTag.STOP)


class SilenceEvent(InferenceEvent):
    def __init__(self, since):
        super().__init__(EventTag.SILENCE)
        self.since = since


class HypothesisEvent(InferenceEvent):
    def __init__(self, hyps):
        super().__init__(EventTag.HYPOTHESIS)
        self.hyps = hyps

    def __repr__(self):
        return f"<{self.__class__.__name__} hyps={len(self.hyps)} {self.hyps[0][0]}>"


class WakewordEvent(InferenceEvent):
    def __init__(self, wakeword):
        super().__init__(EventTag.WAKEWORD)
        self.wakeword = wakeword

    def __repr__(self):
        return f"<{self.__class__.__name__} wakeword='{self.wakeword}'>"


class TranscriptEvent(InferenceEvent):
    def __init__(self, transcript):
        super().__init__(EventTag.TRANSCRIPT)
        self.transcript = transcript

    def __repr__(self):
        return f"<{self.__class__.__name__} transcript='{self.transcript}'>"

    def to_protobuf(self):
        import libreasr.api.interfaces.libreasr_pb2 as ap

        return ap.Event(te=ap.TranscriptEvent(transcript=self.transcript))


class SentenceEvent(InferenceEvent):
    def __init__(self, transcript):
        super().__init__(EventTag.SENTENCE)
        self.transcript = transcript

    def __repr__(self):
        return f"<{self.__class__.__name__} transcript='{self.transcript}'>"

    def to_protobuf(self):
        import libreasr.api.interfaces.libreasr_pb2 as ap

        return ap.Event(se=ap.SentenceEvent(transcript=self.transcript))


class AssistantAnswerEvent(InferenceEvent):
    def __init__(self, answer: str):
        super().__init__(EventTag.ASSISTANT_ANSWER)
        self.answer = answer

    def __repr__(self):
        return f"<{self.__class__.__name__} answer='{self.answer}'>"


class TTSEvent(InferenceEvent):
    def __init__(self, audio):
        super().__init__(EventTag.TTS)
        self.audio = audio

    def __repr__(self):
        return f"<{self.__class__.__name__} audio=({len(self.audio)})>"


class EventBus:
    def __init__(self, listeners: List[Callable] = []):
        self.listeners = listeners

    def register(self, listener: Callable):
        self.listeners.append(listener)

    def emit(self, event: InferenceEvent):
        for l in self.listeners:
            l(event, self)
