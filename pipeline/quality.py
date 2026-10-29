"""Technical audio quality: DNSMOS P.835 (ONNX) plus clipping, level, bandwidth and dropout checks."""
import os
import urllib.request

import numpy as np

from . import config

_session = None
DNSMOS_SECONDS = 9.01
P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort

        path = os.path.join(config.MODEL_DIR, "dnsmos_sig_bak_ovr.onnx")
        if not os.path.exists(path):
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            urllib.request.urlretrieve(config.DNSMOS_URL, path + ".part")
            os.replace(path + ".part", path)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = config.NUM_THREADS
        _session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    return _session


def dnsmos(audio: np.ndarray, speech_regions) -> dict:
    sr = config.SAMPLE_RATE
    if speech_regions:
        x = np.concatenate([audio[int(s * sr): int(e * sr)] for s, e in speech_regions])
    else:
        x = audio
    need = int(DNSMOS_SECONDS * sr)
    if len(x) < need:
        x = np.tile(x, int(np.ceil(need / max(len(x), 1))))
    sess = _get_session()
    name = sess.get_inputs()[0].name
    hop = sr * 3
    scores = []
    for start in range(0, max(1, len(x) - need + 1), hop):
        seg = x[start: start + need].astype(np.float32)[None, :]
        sig, bak, ovr = sess.run(None, {name: seg})[0][0]
        scores.append((P_SIG(sig), P_BAK(bak), P_OVR(ovr)))
        if len(scores) >= 20:
            break
    s = np.array(scores)
    return {"sig": round(float(s[:, 0].mean()), 2), "bak": round(float(s[:, 1].mean()), 2),
            "ovr": round(float(s[:, 2].mean()), 2), "sig_p10": round(float(np.percentile(s[:, 0], 10)), 2)}


def signal_checks(audio: np.ndarray, info: dict, speech_regions) -> dict:
    sr = config.SAMPLE_RATE
    if speech_regions:
        sp = np.concatenate([audio[int(s * sr): int(e * sr)] for s, e in speech_regions])
    else:
        sp = audio
    clip_frac = float(np.mean(np.abs(audio) >= 0.98))
    rms_db = float(20 * np.log10(np.sqrt(np.mean(sp ** 2)) + 1e-9))

    spec = np.abs(np.fft.rfft(sp[: sr * 60] * np.hanning(len(sp[: sr * 60]))))
    freqs = np.fft.rfftfreq(len(sp[: sr * 60]), 1 / sr)
    cum = np.cumsum(spec ** 2)
    rolloff = float(freqs[np.searchsorted(cum, 0.99 * cum[-1])]) if cum[-1] > 0 else 0.0

    # Dropouts: runs of digital silence inside speech regions (packet loss).
    dropouts = 0
    for s, e in speech_regions:
        seg = np.abs(audio[int(s * sr): int(e * sr)])
        if seg.size < 400:
            continue
        hop = int(0.02 * sr)
        n = len(seg) // hop
        frames = seg[: n * hop].reshape(n, hop).max(axis=1)
        dropouts += int(np.sum((frames[1:] < 1e-4) & (frames[:-1] > 1e-2)))
    return {
        "clip_fraction": round(clip_frac, 5),
        "speech_rms_db": round(rms_db, 1),
        "rolloff_99_hz": round(rolloff),
        "dropouts": dropouts,
        "source_peak": round(info.get("source_peak", 0.0), 3),
    }
