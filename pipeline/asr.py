from dataclasses import dataclass

import numpy as np

from . import config

_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        _model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=config.NUM_THREADS)
    return _model


@dataclass
class Utterance:
    start: float
    end: float
    text: str
    no_speech_prob: float
    speaker: str = "unknown"

    @property
    def duration(self) -> float:
        return self.end - self.start


def transcribe(audio: np.ndarray) -> tuple[list[Utterance], str, float]:
    segments, info = _get_model().transcribe(
        audio,
        beam_size=1,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    utts = [
        Utterance(float(s.start), float(s.end), s.text.strip(), float(s.no_speech_prob))
        for s in segments
        if s.text.strip()
    ]
    return utts, info.language, float(info.language_probability)
