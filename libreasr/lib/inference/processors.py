from abc import ABC
from functools import partial
from typing import List, Dict
import pprint
import subprocess

import numpy as np
import torch

from libreasr.lib.inference.events import *
from libreasr.lib.utils import warn_about_license


class InferenceProcessor(ABC):
    def __init__(self, debug=False):
        self.dbg = debug

    def on_speech(self, segment: torch.Tensor, ctx: Dict = {}) -> List[torch.Tensor]:
        """
        Callback for receiving/processing
        Segments of speech
        """
        return [segment]

    def on_event(self, event: InferenceEvent, bus: EventBus):
        """
        Callback called whenever an output event
        is generated (either by the model or on the `EventBus`)
        """
        pass

    def debug(self, *args, **kwargs):
        if self.dbg:
            print(f"[{self.__class__.__name__}]", *args, **kwargs)

    def warn(self, *args, **kwargs):
        pfix = f"[{self.__class__.__name__}] [warning]"
        print(pfix, *args, **kwargs)


class MultiInferenceProcessor:
    def __init__(self, processors: List[InferenceProcessor], flatten_fn=None):
        self.processors = processors
        assert flatten_fn is not None
        self.flatten_fn = flatten_fn

        # create EventBus and
        #  register ourselves
        self.event_bus = EventBus([x.on_event for x in processors])
        self.event_bus.register(self.on_event)

        # vars
        self.outputs = []
        self.ctx = {}

    def __call__(self, gen):
        for proc in self.processors:
            # call on speech (with preloaded context)
            func = partial(proc.on_speech, ctx=self.ctx)
            gen = map(func, gen)

            # flatten
            gen = self.flatten_fn(gen)
        yield from gen

    def on_speech(self, segment: torch.Tensor, ctx: Dict = {}) -> List[torch.Tensor]:
        raise NotImplementedError()

    def on_event(self, event: InferenceEvent, bus: EventBus):
        if event.tag in EVENTS_FOR_OUTPUT:
            self.outputs.append(event)

    def get_output_events(self):
        return self.outputs

    def clear_output_events(self):
        self.outputs = []

    def emit_model_event(self, event: InferenceEvent):
        self.event_bus.emit(event)


class VADProcessor(InferenceProcessor):
    def __init__(
        self,
        sr=16000,
        vad_duration=0.25,
        cooldown_duration=0.25,
        threshold=0.1,
        debug=False,
    ):
        super().__init__(debug=debug)

        warn_about_license(
            "VADProcessor", "Silero VAD", "https://github.com/snakers4/silero-vad"
        )

        # load silero vad
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
        )
        (
            get_speech_ts,
            get_speech_ts_adaptive,
            save_audio,
            read_audio,
            state_generator,
            single_audio_stream,
            collect_chunks,
        ) = utils
        self.model = model

        # args
        self.sr = sr
        self.vad_duration = vad_duration
        self.cooldown_duration = cooldown_duration
        self.threshold = threshold

        # vars
        self.buffer = []
        self.cooldown = 0.0

    def on_speech(self, segment: torch.Tensor, ctx: Dict = {}) -> List[torch.Tensor]:

        # pass thru if we have recognized speech before
        is_cooldown = False
        segment_sz = segment.size(1) / self.sr
        if self.cooldown > 0:
            is_cooldown = True
            self.cooldown -= segment_sz
        else:
            self.cooldown = 0.0

        # do vad
        vad_outs = self.model(segment.squeeze())
        confidence = vad_outs[0][1].item()
        voice_detected = False
        if confidence > self.threshold:
            voice_detected = True
            self.cooldown = self.cooldown_duration

        # return or not
        self.debug(vad_outs, is_cooldown, voice_detected)
        if is_cooldown or voice_detected:
            return [segment]
        else:
            return [None]


class WakewordProcessor(InferenceProcessor):
    def __init__(self, sr=16000, frame_length=512, keywords=["computer"], debug=False):
        super().__init__(debug=debug)

        warn_about_license(
            "WakewordProcessor",
            "Picovoice Porcupine Wake-Word-Detection",
            "https://github.com/Picovoice/porcupine",
        )

        # load pvporcupine
        import pvporcupine

        keyword_paths = [pvporcupine.KEYWORD_PATHS[x] for x in keywords]
        porcupine = pvporcupine.create(keyword_paths=keyword_paths)

        # args
        self.sr = sr
        self.frame_length = frame_length
        self.keywords = keywords

        # vars
        self.porcupine = porcupine
        self.muted = True
        self.bus = None

    def on_speech(self, segment: torch.Tensor, ctx: Dict = {}) -> List[torch.Tensor]:

        # if segment is None, bail
        if segment is None:
            return [None]

        # accumulate pieces of size self.frame_length
        #  and convert into correct format
        pcm = segment
        pcm = (pcm * 255.0).int()[0].tolist()
        self.debug("pcm stats", segment.mean(), segment.std())

        # run detection
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        for chunk in chunks(pcm, self.frame_length):
            result = self.porcupine.process(chunk)
            if result >= 0:
                kw = self.keywords[result]
                self.debug(f"detected '{kw}'")
                self.muted = False
                self.bus.emit(WakewordEvent(kw))

        if not self.muted:
            return [segment]
        else:
            return [None]

    def on_event(self, event: InferenceEvent, bus: EventBus):
        if self.bus is None:
            self.bus = bus
        if not self.muted and event.tag == EventTag.SILENCE:
            self.muted = True


class EventDebugProcessor(InferenceProcessor):
    def on_speech(self, segment: torch.Tensor, ctx: Dict = {}) -> List[torch.Tensor]:
        return [segment]

    def on_event(self, event: InferenceEvent, bus: EventBus):
        print("[EventBus]", event)


class TranscriptProcessor(InferenceProcessor):
    """
    Turns HypothesisEvent into TranscriptEvent.
    To Do:
    - don't emit event if last transcript is same
    """

    def __init__(self, denumericalizer, choose_best=True, blank=0, debug=False):
        super().__init__(debug=debug)
        self.denumericalizer = denumericalizer
        self.choose_best = choose_best
        self.blank = blank
        self.last_transcript = ""

    def on_event(self, event: InferenceEvent, bus: EventBus):
        if event.tag == EventTag.HYPOTHESIS:
            hyps = event.hyps
            hyp = []
            if self.choose_best:
                hyp = sorted(hyps, key=lambda x: x[-1], reverse=True)[0]
            else:
                raise NotImplementedError()

            # only tokens
            transcript = hyp[0]

            # no blanks
            transcript = list(filter(lambda x: x != self.blank, transcript))

            if len(transcript) > 1:
                transcript = self.denumericalizer(transcript[1:])
            else:
                transcript = ""
            if transcript != self.last_transcript:
                self.last_transcript = transcript
                bus.emit(TranscriptEvent(transcript))


class EOSProcessor(InferenceProcessor):
    """
    Detects the end of a sentence after silence
    of duration `silence`.
    Fires `SilenceEvent`.
    """

    def __init__(self, sr=16000, silence=2.0, debug=False):
        super().__init__(debug=debug)

        # args
        self.sr = sr
        self.silence = silence
        self.frame_length = 2560
        self.chunk = -1

        # vars
        self.reset()
        self.bus = None
        self.blocked = True

    def reset(self):
        self.segment_counter = 0
        self.event_counter = 0
        self.last_hypothesis_at = 0

    def detect_silence(self):
        diff = (self.segment_counter - self.last_hypothesis_at) * self.chunk
        is_silence = diff >= self.silence and not self.blocked
        if is_silence:
            self.bus.emit(SilenceEvent(diff))
            self.blocked = True
        self.debug(
            is_silence,
            self.segment_counter,
            self.last_hypothesis_at,
            diff,
            self.blocked,
        )
        return is_silence

    def on_speech(self, segment: torch.Tensor, ctx: Dict = {}) -> List[torch.Tensor]:
        if segment is None:
            pass
        else:
            self.frame_length = segment.size(-1)
            self.chunk = self.frame_length / self.sr
        self.segment_counter += 1

        sil = self.detect_silence()
        if sil:
            self.reset()

        return [segment]

    def on_event(self, event: InferenceEvent, bus: EventBus):
        if event.tag == EventTag.START:
            self.bus = bus
        if event.tag in (EventTag.TRANSCRIPT,):
            self.event_counter += 1
            self.last_hypothesis_at = self.segment_counter
            self.blocked = False


class SentenceProcessor(InferenceProcessor):
    """
    Turns `TranscriptEvent` into `SentenceEvent`
    when silence is hit.
    """

    def __init__(self, debug=False):
        super().__init__(debug=debug)
        self.handle = None
        self.transcript = ""

    def emit(self, bus):
        t = self.transcript
        if t != "":
            t = t + "."
            self.handle.reset()
            bus.emit(SentenceEvent(t))
            self.transcript = ""

    def on_event(self, event: InferenceEvent, bus: EventBus):
        # grab handle to beamer
        if event.tag == EventTag.START:
            self.handle = event.beamer

        # grab latest transcript
        if event.tag == EventTag.TRANSCRIPT:
            self.transcript = event.transcript

        # maybe output (last) sentence
        if event.tag in (EventTag.SILENCE, EventTag.STOP):
            self.emit(bus)


class AssistantProcessor(InferenceProcessor):
    """
    Represents the brain of
    a simple virtual assistant
    * Consumes `SentenceEvent`
    * Produces `AssitantAnswerEvent`
    """

    def answer(self, question):
        text = question
        answer = "Ich habe deine Frage nicht verstanden."
        if "hallo" in text or "welt" in text:
            answer = "Hallo ich bin ein automatisches Spracherkennungssystem."
        if "wetter" in text and ("heute" in text or "morgen" in text):
            # TODO: use wttr.in to get the weather :)
            answer = "Ich weiß nicht. Frag doch Google oder Siri!"
        if "licht" in text or "aus" in text:
            answer = "Licht aus."
        if "licht" in text and ("ein" in text or "an" in text):
            answer = "Licht ein."
        return answer

    def on_event(self, event: InferenceEvent, bus: EventBus):
        if event.tag == EventTag.SENTENCE:
            text = event.transcript
            answer = self.answer(text)
            bus.emit(AssistantAnswerEvent(answer))


class TTSProcessor(InferenceProcessor):
    """
    * Consumes `AssitantAnswerEvent`
    * Produces `TTSEvent`
    """

    def tts(self, text, lang="de", speed=160):
        p = subprocess.Popen(
            ["espeak", "--stdout", f"-v{lang}", f"-s {speed}", text],
            stdout=subprocess.PIPE,
        )
        audio, err = p.communicate()
        audio = np.frombuffer(audio, dtype=np.int16)
        return audio

    def on_event(self, event: InferenceEvent, bus: EventBus):
        if event.tag == EventTag.ASSISTANT_ANSWER:
            text = event.answer
            audio = self.tts(text)
            bus.emit(TTSEvent(audio))
