"""Label each utterance as agent or customer.

Calls are mono and the voice agent opens every call with a scripted greeting, so the
agent voice is anchored on that greeting (or the first substantial utterance).
One embedding per utterance keeps this cheap.
"""
import re

import numpy as np
import torch

from . import config
from .asr import Utterance

_embedder = None
MIN_EMBED_SECONDS = 0.6
MIN_ANCHOR_SECONDS = 1.5
SAME_SPEAKER_THRESHOLD = 0.45
AGENT_OPENER = re.compile(
    r"(how (can|may) i help|this is \w+ (from|with)|i'?m \w+ (from|with)|thanks? for calling|"
    r"c[oó]mo (puedo|le puedo) ayudar|soy \w+ de|gracias por llamar)",
    re.I,
)


def _get_embedder():
    global _embedder
    if _embedder is None:
        from pyannote.audio import Inference, Model

        model = Model.from_pretrained(config.SPEAKER_EMBEDDING_MODEL, token=config.HF_TOKEN)
        _embedder = Inference(model, window="whole")
    return _embedder


def _embed(audio: np.ndarray, start: float, end: float) -> np.ndarray:
    sr = config.SAMPLE_RATE
    chunk = audio[int(start * sr): int(end * sr)]
    emb = _get_embedder()({"waveform": torch.from_numpy(chunk)[None, :], "sample_rate": sr})
    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    return emb / (np.linalg.norm(emb) + 1e-9)


def assign_speakers(audio: np.ndarray, utts: list[Utterance]) -> dict:
    if not utts:
        return {"method": "none"}
    if len(utts) == 1:
        utts[0].speaker = "customer"
        return {"method": "single_utterance"}

    # ASR timestamps can run past the end of the decoded audio.
    audio_end = len(audio) / config.SAMPLE_RATE
    embeddable = [u for u in utts if min(u.end, audio_end) - u.start >= MIN_EMBED_SECONDS]
    if len(embeddable) < 2:
        utts[0].speaker = "agent"
        for u in utts[1:]:
            u.speaker = "customer"
        return {"method": "first_utterance_only"}

    embs = {id(u): _embed(audio, u.start, u.end) for u in embeddable}
    early = [u for u in embeddable if u.start < 30.0]
    anchor_utt = next((u for u in early if AGENT_OPENER.search(u.text)), None)
    method = "opener"
    if anchor_utt is None:
        anchor_utt = next((u for u in embeddable if u.duration >= MIN_ANCHOR_SECONDS), embeddable[0])
        method = "first_substantial"
    anchor = embs[id(anchor_utt)]
    # Refine the agent centroid with utterances that clearly match the anchor.
    sims = {k: float(v @ anchor) for k, v in embs.items()}
    agent_keys = [k for k, s in sims.items() if s >= SAME_SPEAKER_THRESHOLD + 0.15] or [id(anchor_utt)]
    centroid = np.mean([embs[k] for k in agent_keys], axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-9

    for u in utts:
        if id(u) in embs:
            sim = float(embs[id(u)] @ centroid)
            u.speaker = "agent" if sim >= SAME_SPEAKER_THRESHOLD else "customer"
        else:
            u.speaker = "unknown"
    anchor_utt.speaker = "agent"
    # Short unembeddable fragments inherit the nearest labelled neighbour.
    for i, u in enumerate(utts):
        if u.speaker == "unknown":
            prev = next((utts[j].speaker for j in range(i - 1, -1, -1) if utts[j].speaker != "unknown"), "customer")
            u.speaker = "customer" if prev == "agent" else prev
    return {"method": method, "similarities": [round(float(embs[id(u)] @ centroid), 2) if id(u) in embs else None for u in utts]}
