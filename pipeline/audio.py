import os
import subprocess

import numpy as np
import soundfile as sf
import soxr

from .config import MAX_FILE_SECONDS, MIN_FILE_SECONDS, SAMPLE_RATE

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".opus", ".flac", ".m4a", ".aac", ".webm", ".wma", ".amr", ".mp4"}


class AudioError(Exception):
    """Raised when a file cannot be turned into usable audio; message is shown to the user."""


def _decode_ffmpeg(path: str) -> np.ndarray:
    import imageio_ffmpeg

    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-i", path,
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=300)
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "ignore").strip().splitlines()
        raise AudioError(f"could not decode audio ({msg[-1] if msg else 'ffmpeg error'})")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def load_audio(path: str) -> tuple[np.ndarray, dict]:
    """Decode any supported file to mono float32 at SAMPLE_RATE. Returns (audio, info)."""
    info = {}
    if os.path.getsize(path) == 0:
        raise AudioError("file is empty (0 bytes)")
    try:
        x, sr = sf.read(path, always_2d=True, dtype="float32")
        info["channels"] = x.shape[1]
        info["source_sample_rate"] = sr
        # Raw peak before downmix, used for clipping detection.
        info["source_peak"] = float(np.abs(x).max()) if x.size else 0.0
        mono = x.mean(axis=1)
        audio = soxr.resample(mono, sr, SAMPLE_RATE).astype(np.float32) if sr != SAMPLE_RATE else mono
    except Exception:
        audio = _decode_ffmpeg(path)
        info["channels"] = None
        info["source_sample_rate"] = None
        info["source_peak"] = float(np.abs(audio).max()) if audio.size else 0.0

    duration = len(audio) / SAMPLE_RATE
    info["duration"] = duration
    if duration < MIN_FILE_SECONDS:
        raise AudioError(f"audio too short ({duration:.2f}s)")
    if duration > MAX_FILE_SECONDS:
        raise AudioError(f"audio too long ({duration / 60:.1f} min; limit {MAX_FILE_SECONDS // 60} min)")
    if not np.isfinite(audio).all():
        raise AudioError("audio contains invalid samples")
    if float(np.abs(audio).max()) < 1e-5:
        raise AudioError("audio is completely silent")
    return audio, info
