"""Customer emotion features: acoustic (emotion2vec+) and text (multilingual sentiment + cues)."""
import re

import numpy as np

from . import config
from .asr import Utterance

_ser = None
_sentiment = None
_heads = None

SER_CLASSES = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]
MIN_SER_SECONDS = 0.8
MAX_SER_SECONDS = 20.0
SER_BUDGET_SECONDS = 60.0


def _get_ser():
    global _ser
    if _ser is None:
        from funasr import AutoModel

        _ser = AutoModel(model=config.EMOTION_MODEL, hub="hf", disable_update=True, disable_pbar=True, log_level="ERROR")
    return _ser


def _get_heads():
    """Logistic-regression heads on emotion2vec+ embeddings (trained by eval/eval_emotion.py)."""
    global _heads
    if _heads is None:
        import os

        import joblib

        path = os.path.join(config.MODEL_DIR, "emotion_heads.joblib")
        _heads = joblib.load(path) if os.path.exists(path) else {}
    return _heads


def calibrate_confidence(raw: float) -> float:
    cal = _get_heads().get("confidence_calibrator")
    return float(cal.predict([raw])[0]) if cal is not None else raw


def apply_heads(embeddings: np.ndarray, weights: np.ndarray, heads: dict | None = None) -> dict:
    heads = _get_heads() if heads is None else heads
    if not heads or len(embeddings) == 0:
        return {}
    w = weights / weights.sum()
    tone = heads["tone"]
    p = (tone.predict_proba(embeddings) * w[:, None]).sum(axis=0)
    inten = heads["intensity"]
    p_int = (inten.predict_proba(embeddings) * w[:, None]).sum(axis=0)
    return {
        "tone_probs": {c: round(float(v), 3) for c, v in zip(tone.classes_, p)},
        "p_high": round(float(p_int[list(inten.classes_).index("high")]), 3),
    }


def _get_sentiment():
    global _sentiment
    if _sentiment is None:
        from transformers import pipeline

        _sentiment = pipeline("text-classification", model=config.TEXT_SENTIMENT_MODEL, top_k=None, device=-1)
    return _sentiment


def _label_key(label: str) -> str:
    return label.split("/")[-1].strip().lower().replace("<unk>", "unknown")


def acoustic_emotion(audio: np.ndarray, utts: list[Utterance]) -> dict:
    sr = config.SAMPLE_RATE
    model = _get_ser()
    probs, weights = [], []
    selected, budget = [], SER_BUDGET_SECONDS
    for u in sorted((u for u in utts if u.duration >= MIN_SER_SECONDS), key=lambda u: -u.duration):
        if budget <= 0:
            break
        selected.append(u)
        budget -= min(u.duration, MAX_SER_SECONDS)
    embs = []
    for u in sorted(selected, key=lambda u: u.start):
        end = min(u.end, u.start + MAX_SER_SECONDS, len(audio) / sr)
        if end - u.start < MIN_SER_SECONDS:
            continue
        chunk = audio[int(u.start * sr): int(end * sr)]
        res = model.generate(chunk, granularity="utterance", extract_embedding=True)[0]
        p = dict.fromkeys(SER_CLASSES, 0.0)
        for lab, sc in zip(res["labels"], res["scores"]):
            p[_label_key(lab)] = float(sc)
        probs.append([p[c] for c in SER_CLASSES])
        embs.append(np.asarray(res["feats"], dtype=np.float32).reshape(-1))
        weights.append(end - u.start)
    if not probs:
        return {"n": 0}
    probs = np.array(probs)
    w = np.array(weights) / np.sum(weights)
    mean = dict(zip(SER_CLASSES, (probs * w[:, None]).sum(axis=0).round(3).tolist()))
    peak = dict(zip(SER_CLASSES, probs.max(axis=0).round(3).tolist()))
    return {"n": len(probs), "seconds": round(float(np.sum(weights)), 1), "mean": mean, "peak": peak,
            **apply_heads(np.stack(embs), np.array(weights))}


PROFANITY = re.compile(
    r"\b(fuck\w*|shit\w*|damn\w*|bullshit|asshole|stupid|idiot|ridiculous|"
    r"mierda|pendej\w*|mamah?u[eé]vo|mamag[uü][eé]vo|carajo|cabr[oó]n|idiota|est[uú]pid\w*|joder|co[ñn]o)\b",
    re.I,
)
ROBOT_CHECK = re.compile(r"(real person|a human|talk to (a )?(person|human|someone)|representative|persona real|un humano)", re.I)
THANKS = re.compile(r"\b(thank(s| you)|appreciate|perfect|great|wonderful|awesome|gracias|perfecto|excelente)\b", re.I)
DISTRESS = re.compile(r"\b(emergency|help me|accident|scared|please hurry|crying|urgent|emergencia|ay[uú]dame|accidente)\b", re.I)


def text_emotion(utts: list[Utterance]) -> dict:
    text = " ".join(u.text for u in utts).strip()
    out = {"chars": len(text)}
    if not text:
        return out
    clf = _get_sentiment()
    scores = clf(text[:1500], truncation=True)[0]
    out["sentiment"] = {s["label"].lower(): round(float(s["score"]), 3) for s in scores}
    words = re.findall(r"\w+", text.lower())
    hello = sum(1 for w in words if w in ("hello", "hola", "aló", "alo"))
    out["hello_repeats"] = hello
    out["profanity"] = len(PROFANITY.findall(text))
    out["robot_check"] = len(ROBOT_CHECK.findall(text))
    out["thanks"] = len(THANKS.findall(text))
    out["distress_words"] = len(DISTRESS.findall(text))
    out["exclaim_or_question"] = text.count("?") + text.count("!")
    return out


def prosody(audio: np.ndarray, utts: list[Utterance]) -> dict:
    """Loudness/pitch variability of the customer relative to the agent (never used alone)."""
    import librosa

    sr = config.SAMPLE_RATE

    def stats(sel):
        if not sel:
            return None
        seg = np.concatenate([audio[int(u.start * sr): int(u.end * sr)] for u in sel])[: sr * 60]
        if len(seg) < sr // 2:
            return None
        rms = librosa.feature.rms(y=seg, frame_length=1024, hop_length=320)[0]
        f0 = librosa.yin(seg, fmin=70, fmax=400, sr=sr, frame_length=1024, hop_length=320)
        voiced = f0[(rms > np.percentile(rms, 40))]
        return {
            "rms_db": float(20 * np.log10(np.mean(rms) + 1e-9)),
            "f0_median": float(np.median(voiced)) if voiced.size else 0.0,
            "f0_iqr": float(np.subtract(*np.percentile(voiced, [75, 25]))) if voiced.size else 0.0,
        }

    return {
        "customer": stats([u for u in utts if u.speaker == "customer"]),
        "agent": stats([u for u in utts if u.speaker == "agent"]),
    }
