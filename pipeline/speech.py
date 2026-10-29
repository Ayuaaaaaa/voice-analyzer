"""Speech activity, overlapped speech and silence from pyannote segmentation-3.0 (single pass)."""
from dataclasses import dataclass, field

import numpy as np
import torch

from . import config

_inference = None


def _get_inference():
    global _inference
    if _inference is None:
        from pyannote.audio import Inference, Model

        model = Model.from_pretrained(config.SEGMENTATION_MODEL, token=config.HF_TOKEN)

        def hook(scores: np.ndarray) -> np.ndarray:
            # scores: (chunks, frames, local_speakers); max -> speech, 2nd max -> overlap
            ordered = np.sort(scores, axis=-1)
            return np.stack([ordered[..., -1], ordered[..., -2]], axis=-1)

        _inference = Inference(model, step=config.SEG_STEP_SECONDS, pre_aggregation_hook=hook, batch_size=16)
    return _inference


def _regions(mask: np.ndarray, frame_times: np.ndarray, frame_step: float) -> list[tuple[float, float]]:
    out = []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for s, e in zip(edges[::2], edges[1::2]):
        out.append((float(frame_times[s]), float(frame_times[e - 1] + frame_step)))
    return out


def _merge(regions, max_gap):
    merged = []
    for s, e in regions:
        if merged and s - merged[-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


@dataclass
class SpeechInfo:
    duration: float
    speech_regions: list = field(default_factory=list)
    overlap_regions: list = field(default_factory=list)
    speech_seconds: float = 0.0
    speech_ratio: float = 0.0
    overlap_seconds: float = 0.0
    max_overlap_region: float = 0.0
    max_silence: float = 0.0
    max_internal_silence: float = 0.0
    lead_silence: float = 0.0
    trail_silence: float = 0.0


def analyze_speech(audio: np.ndarray) -> SpeechInfo:
    inf = _get_inference()
    wav = torch.from_numpy(audio)[None, :]
    feats = inf({"waveform": wav, "sample_rate": config.SAMPLE_RATE})
    data = feats.data
    sw = feats.sliding_window
    times = np.array([sw[i].start for i in range(data.shape[0])])
    duration = len(audio) / config.SAMPLE_RATE

    speech = _merge(_regions(data[:, 0] > config.VAD_ONSET, times, sw.step), max_gap=0.3)
    speech = [(s, min(e, duration)) for s, e in speech if e - s >= 0.15]
    overlap = [(s, e) for s, e in _regions(data[:, 1] > config.OSD_ONSET, times, sw.step) if e - s >= 0.05]

    info = SpeechInfo(duration=duration, speech_regions=speech, overlap_regions=overlap)
    info.speech_seconds = sum(e - s for s, e in speech)
    info.speech_ratio = info.speech_seconds / duration if duration else 0.0
    info.overlap_seconds = sum(e - s for s, e in overlap)
    info.max_overlap_region = max((e - s for s, e in overlap), default=0.0)
    if speech:
        gaps = [b[0] - a[1] for a, b in zip(speech, speech[1:])]
        info.max_internal_silence = max(gaps, default=0.0)
        info.lead_silence = speech[0][0]
        info.trail_silence = max(0.0, duration - speech[-1][1])
        info.max_silence = max(info.max_internal_silence, info.lead_silence, info.trail_silence)
    else:
        info.max_silence = info.lead_silence = duration
    return info


def non_speech_mask(n_samples: int, speech_regions, pad: float = 0.2) -> np.ndarray:
    mask = np.ones(n_samples, dtype=bool)
    sr = config.SAMPLE_RATE
    for s, e in speech_regions:
        mask[max(0, int((s - pad) * sr)): min(n_samples, int((e + pad) * sr))] = False
    return mask
