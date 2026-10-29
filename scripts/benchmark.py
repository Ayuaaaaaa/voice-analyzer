"""Latency and cost benchmark.

Measures cold start (model loading), per-file wall time, real-time factor (processing time / audio time),
per-stage breakdown and peak memory, then converts to $ per audio minute for a list of CPU instances.
Calls are processed one at a time per worker, so cost/min = RTF x hourly price / 60.

usage: python scripts/benchmark.py <audio files or folder> --label laptop --repeats 2
"""
import argparse
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Approximate on-demand list prices (USD/hour); verify current pricing and edit to re-price.
INSTANCES = {
    "GCE e2-standard-4 us-central1 (4 vCPU, 16 GB, this deployment)": 0.1340,
    "Hetzner CX32 (4 vCPU, 8 GB)": 0.0113,
    "AWS c7i.large (2 vCPU, 4 GB)": 0.0893,
    "AWS c7i.xlarge (4 vCPU, 8 GB)": 0.1785,
    "HF Spaces CPU upgrade (8 vCPU, 32 GB)": 0.03,
}
BUDGET_PER_MIN = 0.003


def peak_rss_mb():
    try:
        import psutil

        p = psutil.Process()
        info = p.memory_info()
        return round(getattr(info, "peak_wset", info.rss) / 1e6)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--label", default=platform.node())
    ap.add_argument("--repeats", type=int, default=1)
    a = ap.parse_args()

    from pipeline import config
    from pipeline.analyzer import analyze, warmup
    from pipeline.audio import SUPPORTED_EXTENSIONS

    files = []
    for x in a.inputs:
        if os.path.isdir(x):
            files += sorted(os.path.join(x, n) for n in os.listdir(x) if os.path.splitext(n)[1].lower() in SUPPORTED_EXTENSIONS)
        else:
            files.append(x)

    t0 = time.perf_counter()
    warmup()
    cold = time.perf_counter() - t0
    print(f"cold start (load all models): {cold:.1f}s", flush=True)

    runs = []
    for rep in range(a.repeats):
        for f in files:
            out = analyze(f)
            tm, dur = out["features"]["timings"], out["features"]["duration"]
            runs.append({"file": os.path.basename(f), "repeat": rep, "audio_seconds": dur,
                         "processing_seconds": tm["total"], "rtf": round(tm["total"] / dur, 3), "stages": tm})
            print(f"  {os.path.basename(f)}  audio {dur:.0f}s  processing {tm['total']:.1f}s  RTF {tm['total'] / dur:.2f}", flush=True)

    audio = sum(r["audio_seconds"] for r in runs)
    proc = sum(r["processing_seconds"] for r in runs)
    rtf = proc / audio
    stages = {}
    for r in runs:
        for k, v in r["stages"].items():
            if k != "total":
                stages[k] = stages.get(k, 0) + v
    out = {
        "label": a.label, "machine": {"platform": platform.platform(), "cpu_count": os.cpu_count(),
                                      "threads_used": config.NUM_THREADS, "processor": platform.processor()},
        "cold_start_seconds": round(cold, 1), "peak_memory_mb": peak_rss_mb(),
        "files": len(files), "repeats": a.repeats,
        "total_audio_seconds": round(audio, 1), "total_processing_seconds": round(proc, 1),
        "rtf_overall": round(rtf, 3),
        "latency_per_file_seconds": {"min": min(r["processing_seconds"] for r in runs),
                                     "max": max(r["processing_seconds"] for r in runs)},
        "stage_share": {k: round(v / proc, 3) for k, v in sorted(stages.items(), key=lambda kv: -kv[1])},
        "cost_per_audio_minute_usd": {k: round(rtf * p / 60, 5) for k, p in INSTANCES.items()},
        "budget_per_audio_minute_usd": BUDGET_PER_MIN,
        "runs": runs,
    }
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=2))
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "results", f"benchmark_{a.label}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
