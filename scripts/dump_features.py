"""Extract raw features for every audio file in a folder and save them as JSON for rule design."""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from pipeline.analyzer import extract_features  # noqa: E402
from pipeline.audio import SUPPORTED_EXTENSIONS  # noqa: E402

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else "features.json"
out = {}
for path in sorted(glob.glob(os.path.join(src, "*"))):
    if os.path.splitext(path)[1].lower() not in SUPPORTED_EXTENSIONS:
        continue
    t0 = time.time()
    try:
        feats = extract_features(path)
    except Exception as e:
        print(os.path.basename(path), "FAILED", type(e).__name__, e, flush=True)
        continue
    feats["wall_seconds"] = round(time.time() - t0, 1)
    out[os.path.basename(path)] = feats
    print(os.path.basename(path), feats["duration"], "s ->", feats["wall_seconds"], "s", feats["timings"], flush=True)

with open(dst, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1, default=str)
print("saved", dst)
