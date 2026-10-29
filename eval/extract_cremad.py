"""Extract emotion2vec+ scores and embeddings for CREMA-D (telephone-simulated).

Labels use what crowd raters perceived from the voice alone (summaryTable.csv: VoiceVote, VoiceLevel),
not the emotion the actor intended: the trial's labels are also human judgements of how a caller sounds.
"""
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from common import DATA_DIR, load, normalize, telephone

from pipeline.emotion import SER_CLASSES, _get_ser, _label_key

CREMA_DIR = os.path.join(DATA_DIR, "cremad")
CACHE = os.path.join(DATA_DIR, "cache_cremad.pkl")
STRONG_LEVEL = 65.0  # perceived intensity (0-100); between the mean of MD (60) and HI (76) anger clips


def proxy_label(vote: str, level: float) -> str | None:
    strong = level >= STRONG_LEVEL
    return {
        "N": "neutral", "H": "satisfied", "D": "frustrated", "F": "distressed",
        "A": "upset" if strong else "frustrated", "S": "distressed" if strong else None,
    }[vote]


def table() -> pd.DataFrame:
    d = pd.read_csv(os.path.join(CREMA_DIR, "summaryTable.csv"))
    d = d[~d.VoiceVote.str.contains(":")].copy()
    d["level"] = d.VoiceLevel.astype(float)
    d["label"] = [proxy_label(v, l) for v, l in zip(d.VoiceVote, d.level)]
    d = d[d.label.notna()]
    d["actor"] = d.FileName.str[:4].astype(int)
    d["intended"] = d.FileName.str.split("_").str[2]
    return d


def main():
    d = table()
    done = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else []
    have = {r["file"] for r in done}
    model = _get_ser()
    t0, rows = time.time(), done
    todo = [r for r in d.itertuples() if r.FileName not in have]
    print(f"{len(d)} labelled clips, {len(todo)} to extract", flush=True)
    for i, r in enumerate(todo):
        p = os.path.join(CREMA_DIR, "data", "AudioWAV", r.FileName + ".wav")
        try:
            x = normalize(telephone(load(p)))
        except Exception as e:
            print("skip", r.FileName, e, file=sys.stderr)
            continue
        res = model.generate(x, granularity="utterance", extract_embedding=True)[0]
        scores = dict.fromkeys(SER_CLASSES, 0.0)
        for lab, sc in zip(res["labels"], res["scores"]):
            scores[_label_key(lab)] = float(sc)
        rows.append({
            "file": r.FileName, "dataset": "cremad", "actor": int(r.actor), "intended": r.intended,
            "vote": r.VoiceVote, "level": float(r.level), "strong": bool(r.level >= STRONG_LEVEL),
            "label": r.label, "scores": scores,
            "embedding": np.asarray(res["feats"], dtype=np.float32).reshape(-1),
        })
        if (i + 1) % 250 == 0:
            pickle.dump(rows, open(CACHE, "wb"))
            el = time.time() - t0
            print(f"  {i + 1}/{len(todo)}  {el:.0f}s  eta {el / (i + 1) * (len(todo) - i - 1) / 60:.1f} min", flush=True)
    pickle.dump(rows, open(CACHE, "wb"))
    print("done", len(rows), "rows in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
