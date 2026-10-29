"""Transcribe the sample calls locally to understand content and turn structure."""
import glob
import os
import sys
import time

import numpy as np
import soundfile as sf
import soxr
from faster_whisper import WhisperModel

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "../Voice Analyzer Assessment"
MODEL = os.environ.get("WHISPER_MODEL", "small")

model = WhisperModel(MODEL, device="cpu", compute_type="int8")

for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.ogg"))):
    x, sr = sf.read(path, always_2d=True, dtype="float32")
    mono = x.mean(axis=1)
    audio = soxr.resample(mono, sr, 16000).astype(np.float32)
    t0 = time.time()
    segments, info = model.transcribe(audio, language=None, vad_filter=False)
    segments = list(segments)
    dt = time.time() - t0
    dur = len(audio) / 16000
    print(f"\n===== {os.path.basename(path)}  dur={dur:.1f}s  lang={info.language}  asr_time={dt:.1f}s =====")
    for s in segments:
        print(f"[{s.start:6.1f}-{s.end:6.1f}] {s.text.strip()}")
