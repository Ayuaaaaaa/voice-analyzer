"""Download and load every model once (used at Docker build time so the Space starts without downloads)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.analyzer import warmup  # noqa: E402

t0 = time.time()
warmup()
print(f"all models ready in {time.time() - t0:.0f}s")
