"""Run pyannote diarization on the sample calls and summarize speakers and overlap."""
import glob
import os
import sys
import time

import numpy as np
import soundfile as sf
import soxr
import torch
from pyannote.audio import Pipeline

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "../Voice Analyzer Assessment"
PIPELINE = os.environ.get("DIAR_PIPELINE", "pyannote/speaker-diarization-3.1")

token = os.environ.get("HF_TOKEN")
if not token and os.path.exists(r"D:\va_env\.env"):
    for line in open(r"D:\va_env\.env", encoding="ascii"):
        if line.startswith("HF_TOKEN="):
            token = line.strip().split("=", 1)[1]

torch.set_num_threads(os.cpu_count() or 4)
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio.pipelines import speaker_diarization as _sd

# PLDA is only used by VBx clustering and lives in a separately gated repo.
_orig_get_plda = _sd.get_plda
_sd.get_plda = lambda plda, **kw: None if plda is None else _orig_get_plda(plda, **kw)

pipe = SpeakerDiarization(
    plda=None,
    segmentation_step=float(os.environ.get("SEG_STEP", "0.1")),
    embedding_batch_size=32,
    segmentation="pyannote/segmentation-3.0",
    embedding="pyannote/wespeaker-voxceleb-resnet34-LM",
    clustering="AgglomerativeClustering",
    token=token,
)
pipe.instantiate({
    "segmentation": {"min_duration_off": 0.0},
    "clustering": {"method": "centroid", "min_cluster_size": 12, "threshold": 0.7045654963945799},
})

for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.ogg"))):
    x, sr = sf.read(path, always_2d=True, dtype="float32")
    audio = soxr.resample(x.mean(axis=1), sr, 16000).astype(np.float32)
    wav = torch.from_numpy(audio)[None, :]
    t0 = time.time()
    out = pipe({"waveform": wav, "sample_rate": 16000})
    dt = time.time() - t0
    ann = getattr(out, "speaker_diarization", out)
    dur = len(audio) / 16000
    print(f"\n===== {os.path.basename(path)} dur={dur:.1f}s diar_time={dt:.1f}s =====")
    totals = {}
    for seg, _, spk in ann.itertracks(yield_label=True):
        totals[spk] = totals.get(spk, 0) + seg.duration
        print(f"  [{seg.start:6.1f}-{seg.end:6.1f}] {spk}")
    ov = ann.get_overlap()
    ov_total = sum(s.duration for s in ov)
    speech = ann.get_timeline().support()
    speech_total = sum(s.duration for s in speech)
    gaps = speech.gaps()
    max_gap = max([g.duration for g in gaps], default=0.0)
    lead_silence = speech[0].start if len(speech) else dur
    print(f"  speakers={ {k: round(v, 1) for k, v in totals.items()} }")
    print(f"  overlap total={ov_total:.2f}s regions={len(ov)} -> {[(round(s.start,1), round(s.duration,2)) for s in ov]}")
    print(f"  speech={speech_total:.1f}s max_gap={max_gap:.1f}s lead_silence={lead_silence:.1f}s trailing={dur - (speech[-1].end if len(speech) else 0):.1f}s")
