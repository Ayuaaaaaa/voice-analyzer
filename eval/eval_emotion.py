"""Emotion validation on RAVDESS (telephone-simulated), grouped by actor.

Dev actors 1-12 may be used for fitting/tuning; test actors 13-24 are only used for reporting.
Approaches:
  A  handcrafted acoustic features (pitch/energy/spectral/MFCC) + logistic regression
  B  emotion2vec+ class scores + the production rules (pipeline/rules.py, acoustic cues only)
  C  emotion2vec+ utterance embedding + logistic regression (transfer learning)
"""
import glob
import os
import pickle
import time

import numpy as np
from common import DATA_DIR, SR, load, normalize, report, save_json, telephone

from pipeline import rules
from pipeline.emotion import SER_CLASSES, _get_ser, _label_key

RAV_DIR = os.path.join(DATA_DIR, "ravdess")
CACHE = os.path.join(DATA_DIR, "cache_ravdess.pkl")
EMO = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad", "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}
TONES = ["neutral", "satisfied", "frustrated", "upset", "distressed"]


def proxy_label(emo: str, strong: bool) -> str | None:
    """Map acted RAVDESS emotions onto the trial's customer-tone classes."""
    return {
        "neutral": "neutral", "calm": "neutral", "happy": "satisfied", "disgust": "frustrated",
        "angry": "upset" if strong else "frustrated", "sad": "distressed" if strong else None,
        "fearful": "distressed", "surprised": None,
    }[emo]


def handcrafted(x: np.ndarray) -> np.ndarray:
    import librosa

    f0 = librosa.yin(x, fmin=70, fmax=400, sr=SR, frame_length=1024, hop_length=320)
    r = librosa.feature.rms(y=x, frame_length=1024, hop_length=320)[0]
    voiced = f0[r > np.percentile(r, 40)]
    mf = librosa.feature.mfcc(y=x, sr=SR, n_mfcc=13, hop_length=320)
    cen = librosa.feature.spectral_centroid(y=x, sr=SR, hop_length=320)[0]
    zcr = librosa.feature.zero_crossing_rate(x, hop_length=320)[0]
    rdb = 20 * np.log10(r + 1e-6)
    feats = [
        np.mean(voiced), np.std(voiced), np.subtract(*np.percentile(voiced, [90, 10])),
        np.mean(rdb), np.std(rdb), np.percentile(rdb, 95) - np.percentile(rdb, 20),
        np.mean(cen), np.std(cen), np.mean(zcr), np.mean(r > np.percentile(r, 30)),
    ]
    return np.concatenate([feats, mf.mean(axis=1), mf.std(axis=1)]).astype(np.float32)


def extract():
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    model = _get_ser()
    rows = []
    files = sorted(glob.glob(os.path.join(RAV_DIR, "Actor_*", "*.wav")))
    t0 = time.time()
    for i, p in enumerate(files):
        parts = os.path.basename(p)[:-4].split("-")
        emo, strong, actor = EMO[parts[2]], parts[3] == "02", int(parts[6])
        label = proxy_label(emo, strong)
        if label is None:
            continue
        x = normalize(telephone(load(p)))
        res = model.generate(x, granularity="utterance", extract_embedding=True)[0]
        scores = dict.fromkeys(SER_CLASSES, 0.0)
        for lab, sc in zip(res["labels"], res["scores"]):
            scores[_label_key(lab)] = float(sc)
        rows.append({
            "file": os.path.basename(p), "actor": actor, "emotion": emo, "strong": strong, "label": label,
            "scores": scores, "embedding": np.asarray(res["feats"], dtype=np.float32).reshape(-1),
            "handcrafted": handcrafted(x),
        })
        if i % 100 == 0:
            print(f"  {i}/{len(files)} files, {time.time() - t0:.0f}s", flush=True)
    pickle.dump(rows, open(CACHE, "wb"))
    return rows


def rules_predict(row, heads=None, with_conf=False):
    from pipeline.emotion import apply_heads

    s = row["scores"]
    extra = apply_heads(row["embedding"][None, :], np.ones(1), heads) if heads else {}
    feats = {
        "emotion_acoustic": {"n": 1, "mean": s, "peak": s, **extra},
        "emotion_text": {"chars": 30},
        "prosody": {},
    }
    tone, intensity, conf = rules._emotion(feats, [])
    return (tone, intensity, conf) if with_conf else (tone, intensity)


def fit_heads(rows):
    emo = [r for r in rows if r["label"] != "neutral"]
    return {
        "tone": make_lr().fit(np.stack([r["embedding"] for r in rows]), [r["label"] for r in rows]),
        "intensity": make_lr().fit(np.stack([r["embedding"] for r in emo]), [strength(r) for r in emo]),
    }


def ece(conf, correct, bins=10) -> float:
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return round(float(total), 3)


def calibrate(dev, test, heads_dev):
    """Isotonic map raw rule confidence -> P(tone correct), fit on grouped CV predictions over dev actors."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import GroupKFold

    conf_cv, ok_cv = [], []
    for tr, va in GroupKFold(n_splits=4).split(dev, groups=[r["actor"] for r in dev]):
        heads = fit_heads([dev[i] for i in tr])
        for i in va:
            t, _, c = rules_predict(dev[i], heads, with_conf=True)
            conf_cv.append(c)
            ok_cv.append(t == dev[i]["label"])
    iso = IsotonicRegression(y_min=0.05, y_max=0.95, out_of_bounds="clip").fit(conf_cv, ok_cv)
    preds = [rules_predict(r, heads_dev, with_conf=True) for r in test]
    raw = [p[2] for p in preds]
    ok = [p[0] == r["label"] for p, r in zip(preds, test)]
    cal = iso.predict(raw)
    out = {"ece_raw": ece(raw, ok), "ece_calibrated": ece(cal, ok),
           "mean_conf_raw": round(float(np.mean(raw)), 3), "mean_conf_calibrated": round(float(np.mean(cal)), 3),
           "accuracy": round(float(np.mean(ok)), 3)}
    plot_reliability(raw, cal, ok)
    print("calibration:", out)
    return out, iso


def plot_reliability(raw, cal, ok, bins=10):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from common import RESULTS_DIR

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for vals, name in [(raw, "raw"), (cal, "calibrated")]:
        vals, okf = np.asarray(vals), np.asarray(ok, dtype=float)
        edges = np.linspace(0, 1, bins + 1)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (vals > lo) & (vals <= hi)
            if m.sum() >= 5:
                xs.append(vals[m].mean())
                ys.append(okf[m].mean())
        ax.plot(xs, ys, "o-", label=f"{name} (ECE {ece(vals, ok)})")
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title("Emotion confidence reliability (test actors)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "reliability_emotion.png"), dpi=110)
    plt.close(fig)


def make_lr():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))


def lr_predict(train, test, key, target=lambda r: r["label"]):
    clf = make_lr()
    clf.fit(np.stack([r[key] for r in train]), [target(r) for r in train])
    return list(clf.predict(np.stack([r[key] for r in test])))


def strength(r) -> str:
    return "high" if r["strong"] else "medium"


def train_final(rows, calibrator=None):
    """Fit the production heads on all actors (only after the held-out numbers are reported)."""
    import joblib

    from pipeline import config

    heads = fit_heads(rows)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    path = os.path.join(config.MODEL_DIR, "emotion_heads.joblib")
    joblib.dump({**heads, "confidence_calibrator": calibrator,
                 "trained_on": "RAVDESS all actors, telephone-simulated"}, path)
    print("saved production heads ->", path)


def main():
    rows = extract()
    dev = [r for r in rows if r["actor"] <= 12]
    test = [r for r in rows if r["actor"] > 12]
    print(f"clips: dev={len(dev)} test={len(test)}")
    y = [r["label"] for r in test]
    out = {"split": "dev actors 1-12, test actors 13-24", "n_dev": len(dev), "n_test": len(test)}

    out["A_handcrafted_lr"] = report("emotion_A_handcrafted_lr", y, lr_predict(dev, test, "handcrafted"), TONES)
    preds_b = [rules_predict(r) for r in test]
    out["B_e2v_rules"] = report("emotion_B_e2v_rules", y, [p[0] for p in preds_b], TONES)
    out["C_e2v_embedding_lr"] = report("emotion_C_e2v_embedding_lr", y, lr_predict(dev, test, "embedding"), TONES)

    emo_test = [(r, p) for r, p in zip(test, preds_b) if r["label"] != "neutral"]
    out["B_intensity"] = report(
        "intensity_B_e2v_rules", [strength(r) for r, _ in emo_test], [p[1] for _, p in emo_test], ["medium", "high"],
    )
    emo_dev = [r for r in dev if r["label"] != "neutral"]
    out["C_intensity_embedding_lr"] = report(
        "intensity_C_e2v_embedding_lr", [strength(r) for r, _ in emo_test],
        lr_predict(emo_dev, [r for r, _ in emo_test], "embedding", strength), ["medium", "high"],
    )
    heads_dev = fit_heads(dev)
    preds_b2 = [rules_predict(r, heads_dev) for r in test]
    out["B2_final_rules_with_e2v_heads"] = report("emotion_B2_final", y, [p[0] for p in preds_b2], TONES)
    out["B2_intensity"] = report(
        "intensity_B2_final", [strength(r) for r, _ in emo_test],
        [p[1] for r, p in zip(test, preds_b2) if r["label"] != "neutral"], ["medium", "high"],
    )
    out["confidence_calibration"], iso = calibrate(dev, test, heads_dev)
    print("\nsaved", save_json("emotion_validation.json", out))
    train_final(rows, iso)


if __name__ == "__main__":
    main()
