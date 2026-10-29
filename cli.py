"""Analyze every audio file in a folder.

Usage: python cli.py <folder> [--out results_dir] [--debug]
Writes predictions.csv (name,result_json) and predictions.json. If the folder has a
labels CSV with result_json filled in, a per-field comparison is printed.
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

from pipeline.audio import SUPPORTED_EXTENSIONS  # noqa: E402
from pipeline.schema import FIELDS  # noqa: E402


def read_labels(folder: str) -> dict:
    for path in glob.glob(os.path.join(folder, "*.csv")):
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if rows and "name" in rows[0]:
            out = {}
            for r in rows:
                raw = (r.get("result_json") or "").strip()
                if raw:
                    try:
                        out[r["name"].strip()] = json.loads(raw)
                    except json.JSONDecodeError:
                        pass
            return out
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="results")
    ap.add_argument("--debug", action="store_true", help="also save features and reasons")
    args = ap.parse_args()

    from pipeline.analyzer import analyze

    files = sorted(p for p in glob.glob(os.path.join(args.folder, "*")) if os.path.splitext(p)[1].lower() in SUPPORTED_EXTENSIONS)
    labels = read_labels(args.folder)
    os.makedirs(args.out, exist_ok=True)
    preds, debug, errors = {}, {}, {}
    for p in files:
        name = os.path.basename(p)
        try:
            r = analyze(p)
        except Exception as e:
            errors[name] = f"{type(e).__name__}: {e}"
            print(f"{name}: FAILED {errors[name]}")
            continue
        preds[name] = r["result"]
        debug[name] = r
        t = r["features"]["timings"]["total"]
        print(f"{name}: {r['features']['duration']:.1f}s audio in {t:.1f}s")
        for line in r["reasons"]:
            print("   ", line)

    with open(os.path.join(args.out, "predictions.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "result_json"])
        for name, res in preds.items():
            w.writerow([name, json.dumps(res, ensure_ascii=False)])
    with open(os.path.join(args.out, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump({"predictions": preds, "errors": errors}, f, ensure_ascii=False, indent=2)
    if args.debug:
        with open(os.path.join(args.out, "debug.json"), "w", encoding="utf-8") as f:
            json.dump(debug, f, ensure_ascii=False, indent=1, default=str)

    if labels:
        print("\nfield-by-field vs labels (prediction | label):")
        hits = dict.fromkeys(FIELDS[:-1], 0)
        n = 0
        for name, res in preds.items():
            if name not in labels:
                continue
            n += 1
            lab = labels[name]
            diffs = []
            for k in FIELDS[:-1]:
                ok = str(res[k]).lower() == str(lab.get(k)).lower()
                hits[k] += ok
                if not ok:
                    diffs.append(f"{k}: {res[k]!r} | {lab.get(k)!r}")
            print(f"  {name}: {'all match' if not diffs else '; '.join(diffs)}")
        if n:
            print("  accuracy:", {k: f"{v}/{n}" for k, v in hits.items()})


if __name__ == "__main__":
    main()
