"""Emotion heads v2: RAVDESS + CREMA-D, model selection on dev actors only, one test run.

Splits (by speaker): RAVDESS dev actors 1-12 / test 13-24; CREMA-D dev odd actor ids / test even ids.
Selection uses grouped 5-fold CV over dev speakers and scores the production path
(rules.py with the emotion2vec heads), averaging macro-F1 over the two corpora.

usage: python eval_emotion_combined.py select   # dev-only, prints the CV table
       python eval_emotion_combined.py test     # fits the chosen config on dev, reports test once, trains final heads
"""
import json
import os
import pickle
import shutil
import sys

import numpy as np
from common import DATA_DIR, RESULTS_DIR, report, save_json
from eval_emotion import TONES, ece, extract, rules_predict, strength
from extract_cremad import CACHE as CREMA_CACHE

CHOICE_FILE = os.path.join(RESULTS_DIR, "emotion_v2_choice.json")


def load_rows():
    rav = extract()
    for r in rav:
        r["dataset"] = "ravdess"
    crema = pickle.load(open(CREMA_CACHE, "rb"))
    rows = rav + crema
    for r in rows:
        r["group"] = f"{r['dataset']}:{r['actor']}"
        r["test"] = r["actor"] > 12 if r["dataset"] == "ravdess" else r["actor"] % 2 == 0
    return rows


def make_lr(C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, class_weight="balanced", C=C))


def _weights(rows, balance):
    if not balance:
        return np.ones(len(rows))
    counts = {}
    for r in rows:
        counts[r["dataset"]] = counts.get(r["dataset"], 0) + 1
    return np.array([len(rows) / (len(counts) * counts[r["dataset"]]) for r in rows])


def fit_heads(rows, cfg):
    train = rows if cfg["data"] == "both" else [r for r in rows if r["dataset"] == "ravdess"]
    emo = [r for r in train if r["label"] != "neutral"]
    tone = make_lr(cfg["C"]).fit(np.stack([r["embedding"] for r in train]), [r["label"] for r in train],
                                 logisticregression__sample_weight=_weights(train, cfg["balance"]))
    inten = make_lr(cfg["C"]).fit(np.stack([r["embedding"] for r in emo]), [strength(r) for r in emo],
                                  logisticregression__sample_weight=_weights(emo, cfg["balance"]))
    return {"tone": tone, "intensity": inten}


def macro_f1(y, p):
    from sklearn.metrics import f1_score

    return f1_score(y, p, labels=TONES, average="macro", zero_division=0)


def cv_predictions(dev, cfg, folds=5):
    from sklearn.model_selection import GroupKFold

    preds = [None] * len(dev)
    for tr, va in GroupKFold(n_splits=folds).split(dev, groups=[r["group"] for r in dev]):
        heads = fit_heads([dev[i] for i in tr], cfg)
        for i in va:
            preds[i] = rules_predict(dev[i], heads, with_conf=True)
    return preds


CONFIGS = [{"data": d, "C": C, "balance": b} for d in ("ravdess", "both") for C in (0.1, 0.5)
           for b in ((False,) if d == "ravdess" else (False, True))]


def select(rows):
    dev = [r for r in rows if not r["test"]]
    table = []
    for cfg in CONFIGS:
        preds = cv_predictions(dev, cfg)
        res = {"cfg": cfg}
        for ds in ("ravdess", "cremad"):
            idx = [i for i, r in enumerate(dev) if r["dataset"] == ds]
            res[f"{ds}_f1"] = round(macro_f1([dev[i]["label"] for i in idx], [preds[i][0] for i in idx]), 3)
        res["score"] = round((res["ravdess_f1"] + res["cremad_f1"]) / 2, 3)
        print(json.dumps(res), flush=True)
        table.append(res)
    best = max(table, key=lambda r: r["score"])
    print("best:", best)
    save_json("emotion_v2_selection_dev_cv.json", table)
    with open(CHOICE_FILE, "w") as f:
        json.dump(best["cfg"], f)


def test(rows):
    cfg = json.load(open(CHOICE_FILE))
    dev = [r for r in rows if not r["test"]]
    tst = [r for r in rows if r["test"]]
    heads = fit_heads(dev, cfg)
    preds = [rules_predict(r, heads, with_conf=True) for r in tst]
    out = {"config": cfg, "n_dev": len(dev), "n_test": len(tst)}
    for ds in ("ravdess", "cremad", "all"):
        idx = [i for i, r in enumerate(tst) if ds == "all" or r["dataset"] == ds]
        out[f"tone_{ds}"] = report(f"emotion_v2_{ds}", [tst[i]["label"] for i in idx], [preds[i][0] for i in idx], TONES)
        emo = [i for i in idx if tst[i]["label"] != "neutral"]
        out[f"intensity_{ds}"] = report(f"intensity_v2_{ds}", [strength(tst[i]) for i in emo],
                                        [preds[i][1] for i in emo], ["medium", "high"])

    # Same heads trained on RAVDESS only, scored on CREMA-D test: the cross-corpus gap the old model had.
    old = fit_heads(dev, {**cfg, "data": "ravdess"})
    cr = [r for r in tst if r["dataset"] == "cremad"]
    out["tone_cremad_with_ravdess_only_heads"] = report(
        "emotion_v2_cremad_ravdess_only_heads", [r["label"] for r in cr], [rules_predict(r, old)[0] for r in cr], TONES)

    from sklearn.isotonic import IsotonicRegression

    cv = cv_predictions(dev, cfg)
    iso = IsotonicRegression(y_min=0.05, y_max=0.95, out_of_bounds="clip").fit(
        [p[2] for p in cv], [p[0] == r["label"] for p, r in zip(cv, dev)])
    raw = [p[2] for p in preds]
    ok = [p[0] == r["label"] for p, r in zip(preds, tst)]
    out["confidence_calibration"] = {"ece_raw": ece(raw, ok), "ece_calibrated": ece(iso.predict(raw), ok),
                                     "accuracy": round(float(np.mean(ok)), 3)}
    print("calibration:", out["confidence_calibration"])
    print("saved", save_json("emotion_v2_validation.json", out))
    train_final(rows, cfg)


def train_final(rows, cfg):
    import joblib
    from sklearn.isotonic import IsotonicRegression

    from pipeline import config

    heads = fit_heads(rows, cfg)
    cv = cv_predictions(rows, cfg)
    iso = IsotonicRegression(y_min=0.05, y_max=0.95, out_of_bounds="clip").fit(
        [p[2] for p in cv], [p[0] == r["label"] for p, r in zip(cv, rows)])
    path = os.path.join(config.MODEL_DIR, "emotion_heads.joblib")
    if os.path.exists(path) and not os.path.exists(path + ".v1"):
        shutil.copy(path, path + ".v1")
    joblib.dump({**heads, "confidence_calibrator": iso, "config": cfg,
                 "trained_on": "RAVDESS (all actors) + CREMA-D (all actors, perceived voice labels), telephone-simulated"},
                path)
    print("saved production heads ->", path)


if __name__ == "__main__":
    rows = load_rows()
    print({ds: sum(r["dataset"] == ds for r in rows) for ds in ("ravdess", "cremad")})
    {"select": select, "test": test}[sys.argv[1]](rows)
