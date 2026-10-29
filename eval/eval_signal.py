"""Synthetic validation for background noise, audio quality, overlap and long silence.

Base "calls" alternate two RAVDESS actors (telephone-simulated). Splits never share
actors or ESC-50 folds:  dev = actors 1-12 + ESC folds 1-3,  test = actors 13-24 + folds 4-5.
Ground truth comes from how each clip was constructed:
  noise     ESC-50 class mixed at 20 / 10 / 0 dB SNR  -> severity low / medium / high
  quality   clean, mild or severe clipping / packet loss / low volume / muffling
  overlap   second speaker starts 0 or 0.25 s (false) vs 1 or 2 s (true) before the first ends
  silence   inserted gap of 2 / 5 / 8 s (false) vs 12 / 20 s (true)
Usage: python eval_signal.py [dev|test] [noise,quality,overlap]
"""
import glob
import os
import random
import sys
import time

import librosa
import numpy as np
import pandas as pd
from common import DATA_DIR, SR, load, mix_at_snr, normalize, report, rms, save_json, telephone

from pipeline import noise as noise_mod
from pipeline import quality as quality_mod
from pipeline import rules, speech

RAV_DIR = os.path.join(DATA_DIR, "ravdess")
ESC_DIR = os.path.join(DATA_DIR, "ESC-50-master")
ESC_TO_TYPE = {
    "dog": "dog barking", "crying_baby": "baby crying", "keyboard_typing": "keyboard typing", "wind": "wind",
    "engine": "road noise", "vacuum_cleaner": "mechanical noise", "washing_machine": "mechanical noise", "siren": "siren",
}
SNR_TO_SEVERITY = {20: "low", 10: "medium", 0: "high"}
CLIPS_PER_CLASS = 3


def rav_utterances(actors):
    out = {}
    for a in actors:
        files = sorted(glob.glob(os.path.join(RAV_DIR, f"Actor_{a:02d}", "03-01-0[12]-*.wav")))
        utts = []
        for p in files[:8]:
            x, _ = librosa.effects.trim(load(p), top_db=30)
            utts.append(x)
        out[a] = utts
    return out


def make_call(rng, utts, a, b, n_turns=8, gap=(0.4, 1.2), overlap=None, silence=None):
    """Alternate speakers a/b; optionally overlap one turn or insert one long silence."""
    parts, mask_parts, t = [], [], 0
    sig = np.zeros(0, dtype=np.float32)
    mask = np.zeros(0, dtype=bool)
    special = n_turns // 2
    for i in range(n_turns):
        x = utts[a if i % 2 == 0 else b][rng.randrange(len(utts[a if i % 2 == 0 else b]))]
        g = rng.uniform(*gap)
        if silence is not None and i == special:
            g = silence
        if overlap is not None and i == special:
            start = max(0, len(sig) - int(overlap * SR))
        else:
            sig = np.concatenate([sig, np.zeros(int(g * SR), np.float32)])
            mask = np.concatenate([mask, np.zeros(int(g * SR), bool)])
            start = len(sig)
        end = start + len(x)
        if end > len(sig):
            sig = np.concatenate([sig, np.zeros(end - len(sig), np.float32)])
            mask = np.concatenate([mask, np.zeros(end - len(mask), bool)])
        sig[start:end] += x
        mask[start:end] = True
    sig = np.concatenate([np.zeros(int(0.5 * SR), np.float32), sig, np.zeros(int(0.8 * SR), np.float32)])
    mask = np.concatenate([np.zeros(int(0.5 * SR), bool), mask, np.zeros(int(0.8 * SR), bool)])
    y = normalize(telephone(sig), 0.5)
    n = min(len(y), len(mask))
    y, mask = y[:n], mask[:n]
    # Real lines are never digitally silent: add a barely perceptible floor 45 dB below speech.
    floor = telephone(np.random.default_rng(n).standard_normal(n).astype(np.float32))[:n]
    floor = np.pad(floor, (0, n - len(floor)))
    y = y + floor * (rms(y[mask]) / rms(floor) * 10 ** (-45 / 20))
    return y.astype(np.float32), mask


def degrade(x, kind, level, rng):
    y = x.copy()
    if kind == "clipping":
        g = {"mild": 4.0, "severe": 20.0}[level]
        y = np.clip(y * g, -1, 1)
    elif kind == "packet_loss":
        rate = {"mild": 0.04, "severe": 0.18}[level]
        hop = int(0.02 * SR)
        for i in range(0, len(y), hop):
            if rng.random() < rate:
                y[i: i + hop] = 0
    elif kind == "low_volume":
        y = y * {"mild": 10 ** (-24 / 20), "severe": 10 ** (-42 / 20)}[level]
    elif kind == "muffled":
        from scipy.signal import butter, sosfilt

        sos = butter(6, {"mild": 1200, "severe": 500}[level], fs=SR, output="sos")
        y = sosfilt(sos, y).astype(np.float32)
        y = normalize(y, 0.5)
    return y.astype(np.float32)


def speech_feats(x):
    sp = speech.analyze_speech(x)
    return sp


def noise_features(x, sp, cached=None):
    """Full noise features, or reuse cached AST outputs and recompute only the signal stats."""
    ns = speech.non_speech_mask(len(x), sp.speech_regions)
    if cached is None or not isinstance(cached.get("gap_groups"), dict):
        return {"noise": noise_mod.analyze_noise(x, ns)}
    return {"noise": {
        "gap_groups": cached["gap_groups"], "full_groups": cached["full_groups"], "tv_label_max": cached["tv_label"],
        "snr": noise_mod.snr_estimate(x, ns), "gap_stats": noise_mod.gap_signal_stats(x, ns),
    }}


def quality_features(x, sp):
    q = {"dnsmos": quality_mod.dnsmos(x, sp.speech_regions),
         **quality_mod.signal_checks(x, {"source_peak": float(np.abs(x).max())}, sp.speech_regions)}
    return {"quality": q, "duration": len(x) / SR}


def run_noise(split, rng, utts, actors, rows, cache=None):
    meta = pd.read_csv(os.path.join(ESC_DIR, "meta", "esc50.csv"))
    folds = [1, 2, 3] if split == "dev" else [4, 5]
    meta = meta[meta.fold.isin(folds) & meta.category.isin(ESC_TO_TYPE)]
    jobs = []
    for cat, grp in meta.groupby("category"):
        picks = grp.sample(n=min(CLIPS_PER_CLASS, len(grp)), random_state=1)
        for _, r in picks.iterrows():
            for snr in SNR_TO_SEVERITY:
                jobs.append((cat, r.filename, snr))
    n_clean = len(jobs) // 3
    jobs += [(None, None, None)] * n_clean
    for i, (cat, fn, snr) in enumerate(jobs):
        a, b = rng.sample(actors, 2)
        x, mask = make_call(rng, utts, a, b)
        if cat:
            nz = load(os.path.join(ESC_DIR, "audio", fn))
            x = mix_at_snr(x, mask, telephone(nz), snr)
        sp = speech_feats(x)
        f = noise_features(x, sp, cache[i] if cache else None)
        why = []
        present, kind, sev, _ = rules._noise(f, why)
        gs = f["noise"]["gap_stats"]
        rows.append({
            "task": "noise", "true_present": cat is not None, "pred_present": present,
            "true_type": ESC_TO_TYPE.get(cat, ""), "pred_type": kind,
            "true_severity": SNR_TO_SEVERITY.get(snr, "none"), "pred_severity": sev, "esc": cat, "snr": snr,
            "clicks_per_min": gs.get("clicks_per_min"), "gap_hf_ratio": gs.get("gap_hf_ratio"),
            "gap_p99_db": gs.get("gap_p99_db"), "gap_floor_db": gs.get("gap_floor_db"),
            "floor_snr_db": gs.get("floor_snr_db"), "gap_hf_rel_speech_db": gs.get("gap_hf_rel_speech_db"),
            "events_rel_speech_db": gs.get("events_rel_speech_db"),
            "tv_label": f["noise"]["tv_label_max"], "gap_groups": f["noise"]["gap_groups"],
            "full_groups": f["noise"]["full_groups"], "speech_level_db": f["noise"]["snr"].get("speech_level_db"),
            "reason": why[0] if why else "",
        })
        if i % 20 == 0:
            print(f"  noise {i}/{len(jobs)}", flush=True)
    # Degraded-but-noise-free clips: noise must not be inferred from poor quality.
    j = len(jobs)
    for kind in ["clipping", "packet_loss", "low_volume", "muffled"]:
        for _ in range(4):
            a, b = rng.sample(actors, 2)
            x, _ = make_call(rng, utts, a, b)
            x = degrade(x, kind, "severe", rng)
            sp = speech_feats(x)
            f = noise_features(x, sp, None)
            why = []
            present, k, sev, _ = rules._noise(f, why)
            rows.append({"task": "noise_vs_quality", "degradation": kind, "pred_present": present, "pred_type": k,
                         "gap_groups": f["noise"]["gap_groups"], "full_groups": f["noise"]["full_groups"],
                         "tv_label": f["noise"]["tv_label_max"], "reason": why[0] if why else "",
                         **{k: f["noise"]["gap_stats"].get(k) for k in ("clicks_per_min", "gap_hf_rel_speech_db", "gap_floor_db", "gap_p99_db", "speech_level_db")}})
            j += 1


def run_quality(split, rng, utts, actors, rows):
    conds = [("none", "clean")] + [(k, l) for k in ["clipping", "packet_loss", "low_volume", "muffled"] for l in ["mild", "severe"]]
    truth = {"clean": "clear", "mild": "slightly_impaired", "severe": "severely_impaired"}
    for rep in range(6):
        a, b = rng.sample(actors, 2)
        base, _ = make_call(rng, utts, a, b)
        for kind, level in conds:
            x = base if kind == "none" else degrade(base, kind, level, rng)
            sp = speech_feats(x)
            qf = quality_features(x, sp)
            label, _ = rules._quality(qf, [])
            q = qf["quality"]
            rows.append({"task": "quality", "degradation": kind, "level": level, "true": truth[level], "pred": label,
                         "sig": q["dnsmos"]["sig"], "ovr": q["dnsmos"]["ovr"], "clip": q["clip_fraction"],
                         "rms": q["speech_rms_db"], "rolloff": q["rolloff_99_hz"], "dropouts": q["dropouts"],
                         "dur": qf["duration"]})


def run_overlap_silence(split, rng, utts, actors, rows):
    from pipeline import config

    for rep in range(8):
        a, b = rng.sample(actors, 2)
        for ov in [None, 0.25, 1.0, 2.0]:
            x, _ = make_call(rng, utts, a, b, overlap=ov)
            sp = speech_feats(x)
            rows.append({"task": "overlap", "overlap_s": ov or 0.0, "true": bool(ov and ov >= 1.0),
                         "pred": sp.max_overlap_region >= config.OVERLAP_MIN_REGION_SECONDS})
        for gap in [2, 5, 8, 12, 20]:
            x, _ = make_call(rng, utts, a, b, silence=gap)
            sp = speech_feats(x)
            silent = max(sp.max_internal_silence, sp.lead_silence) >= config.LONG_SILENCE_SECONDS
            rows.append({"task": "silence", "gap_s": gap, "true": gap >= 10, "pred": silent})


def summarize(split, rows):
    df = pd.DataFrame(rows)
    df.to_pickle(os.path.join(DATA_DIR, f"signal_rows_{split}.pkl"))
    out = {"split": split}
    n = df[df.task == "noise"]
    if len(n):
        out["noise_present"] = report(f"noise_present_{split}", n.true_present, n.pred_present, [False, True])
        pos = n[n.true_present & n.pred_present].assign(type_ok=lambda d: d.true_type == d.pred_type)
        out["noise_type_given_detected"] = {
            "n": len(pos), "accuracy": round(float(pos.type_ok.mean()), 3) if len(pos) else None,
            "by_class": pos.groupby("true_type").type_ok.mean().round(2).to_dict(),
            "recall_by_snr": n[n.true_present].groupby("snr").pred_present.mean().round(2).to_dict(),
            "recall_by_class": n[n.true_present].groupby("true_type").pred_present.mean().round(2).to_dict(),
        }
        print("noise type/recall:", out["noise_type_given_detected"])
        out["noise_severity"] = report(f"noise_severity_{split}", n.true_severity, n.pred_severity, ["none", "low", "medium", "high"])
    nq = df[df.task == "noise_vs_quality"]
    if len(nq):
        out["false_noise_on_degraded_audio"] = nq.groupby("degradation").pred_present.mean().round(2).to_dict()
        print("false noise on degraded (should be 0):", out["false_noise_on_degraded_audio"])
    q = df[df.task == "quality"]
    if len(q):
        out["quality"] = report(f"quality_{split}", q.true, q.pred, ["clear", "slightly_impaired", "severely_impaired"])
        out["quality_by_condition"] = q.assign(ok=q.true == q.pred, cond=q.degradation + "/" + q.level).groupby("cond").ok.mean().round(2).to_dict()
        print("quality by condition:", out["quality_by_condition"])
    o = df[df.task == "overlap"]
    if len(o):
        out["overlap"] = report(f"overlap_{split}", o.true, o.pred, [False, True])
        out["overlap_by_duration"] = o.groupby("overlap_s").pred.mean().round(2).to_dict()
        print("overlap detection rate by duration:", out["overlap_by_duration"])
    s = df[df.task == "silence"]
    if len(s):
        out["silence"] = report(f"silence_{split}", s.true, s.pred, [False, True])
        out["silence_by_gap"] = s.groupby("gap_s").pred.mean().round(2).to_dict()
        print("silence detection rate by gap:", out["silence_by_gap"])
    df.to_csv(os.path.join(os.path.dirname(save_json(f"signal_validation_{split}.json", out)), f"signal_rows_{split}.csv"), index=False)


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    tasks = (sys.argv[2] if len(sys.argv) > 2 else "noise,quality,overlap").split(",")
    if tasks == ["summary"]:
        summarize(split, pd.read_pickle(os.path.join(DATA_DIR, f"signal_rows_{split}.pkl")).to_dict("records"))
        return
    cache = None
    if tasks == ["noise_fast"]:
        prev = pd.read_pickle(os.path.join(DATA_DIR, f"signal_rows_{split}.pkl"))
        cache = prev[prev.task.isin(["noise", "noise_vs_quality"])].to_dict("records")
        tasks = ["noise"]
    rng = random.Random(0 if split == "dev" else 1)
    actors = list(range(1, 13)) if split == "dev" else list(range(13, 25))
    utts = rav_utterances(actors)
    rows = []
    t0 = time.time()
    if "overlap" in tasks:
        run_overlap_silence(split, rng, utts, actors, rows)
    if "quality" in tasks:
        run_quality(split, rng, utts, actors, rows)
    if "noise" in tasks:
        run_noise(split, rng, utts, actors, rows, cache)
    print(f"{len(rows)} clips in {time.time() - t0:.0f}s")
    summarize(split, rows)


if __name__ == "__main__":
    main()
