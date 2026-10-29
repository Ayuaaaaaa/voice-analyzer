"""Edge cases the dashboard test does not cover: over-long file, agent-only audio, a call
without the agent's opener, and pure background noise. Checks the pipeline never crashes
and that each case is handled sensibly.

usage: python scripts/test_edge_cases.py <call.ogg>
"""
import os
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.analyzer import analyze  # noqa: E402
from pipeline.audio import AudioError, load_audio  # noqa: E402
from pipeline.config import SAMPLE_RATE as SR  # noqa: E402


def run(name, x, tmp):
    path = os.path.join(tmp, name + ".wav")
    sf.write(path, x, SR, subtype="PCM_16")
    print(f"\n== {name} ({len(x) / SR:.0f}s)")
    try:
        out = analyze(path)
    except AudioError as e:
        print("  AudioError:", e)
        return
    f = out["features"]
    print("  result:", out["result"])
    print("  speakers:", f["speaker_assignment"], "| utterances:",
          [(u["speaker"], u["text"][:40]) for u in f["utterances"][:6]])
    for r in out["reasons"]:
        print("   ", r)


def main():
    audio, _ = load_audio(sys.argv[1])
    tmp = tempfile.mkdtemp(prefix="va_edge_")
    run("agent_opening_only", audio[: 12 * SR], tmp)
    run("no_opener", audio[20 * SR: 80 * SR], tmp)
    rng = np.random.default_rng(0)
    run("noise_only", (0.05 * rng.standard_normal(20 * SR)).astype(np.float32), tmp)
    reps = int(np.ceil(31 * 60 * SR / len(audio)))
    run("too_long_31min", np.tile(audio, reps)[: 31 * 60 * SR], tmp)


if __name__ == "__main__":
    main()
