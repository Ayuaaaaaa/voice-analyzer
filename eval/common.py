"""Shared helpers for proxy validation: data paths, telephone simulation, mixing, metrics."""
import json
import os
import sys

import numpy as np
import soundfile as sf
import soxr
from scipy.signal import butter, sosfilt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = os.environ.get("VA_DATA_DIR", r"D:\va_env\data")
RESULTS_DIR = os.path.join(ROOT, "eval", "results")
SR = 16000
os.makedirs(RESULTS_DIR, exist_ok=True)


def load(path: str) -> np.ndarray:
    x, sr = sf.read(path, always_2d=True, dtype="float32")
    x = x.mean(axis=1)
    return soxr.resample(x, sr, SR).astype(np.float32) if sr != SR else x


_SOS = butter(4, [300, 3400], btype="bandpass", fs=8000, output="sos")


def telephone(x: np.ndarray) -> np.ndarray:
    """Narrowband phone channel: 8 kHz, 300-3400 Hz, mu-law, back to 16 kHz."""
    y = soxr.resample(x, SR, 8000)
    y = sosfilt(_SOS, y)
    y = np.clip(y, -1, 1)
    mu = 255.0
    q = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
    q = np.round(q * 127) / 127
    y = np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu
    return soxr.resample(y.astype(np.float32), 8000, SR).astype(np.float32)


def normalize(x: np.ndarray, peak: float = 0.5) -> np.ndarray:
    m = np.abs(x).max()
    return (x / m * peak).astype(np.float32) if m > 0 else x


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


def mix_at_snr(speech: np.ndarray, speech_mask: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale noise so active-speech RMS / noise RMS equals snr_db; noise is looped to length."""
    reps = int(np.ceil(len(speech) / len(noise)))
    n = np.tile(noise, reps)[: len(speech)]
    s_rms = rms(speech[speech_mask]) if speech_mask.any() else rms(speech)
    gain = s_rms / (rms(n) * 10 ** (snr_db / 20))
    return np.clip(speech + gain * n, -1, 1).astype(np.float32)


def save_json(name: str, obj) -> str:
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    return path


def report(name: str, y_true, y_pred, labels) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    y_true, y_pred = [str(v) for v in y_true], [str(v) for v in y_pred]
    labels = [str(l) for l in labels]
    out = {
        "n": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "macro_f1": round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 3),
        "per_class": {
            k: {m: round(v, 3) for m, v in d.items()}
            for k, d in classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0).items()
            if k in labels
        },
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    plot_confusion(name, out)
    print(f"\n== {name}: n={out['n']} accuracy={out['accuracy']} macro_f1={out['macro_f1']}")
    w = max(len(l) for l in labels) + 2
    print(" " * w + "".join(f"{l[:10]:>11}" for l in labels) + "   <- predicted")
    for l, row in zip(labels, out["confusion_matrix"]):
        print(f"{l:<{w}}" + "".join(f"{v:>11}" for v in row))
    return out


def plot_confusion(name: str, rep: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(rep["confusion_matrix"])
    labels = rep["labels"]
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 2, 1.0 * len(labels) + 1.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"{name}\nacc={rep['accuracy']}  macro-F1={rep['macro_f1']}")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"cm_{name}.png"), dpi=110)
    plt.close(fig)
