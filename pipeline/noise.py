"""Background-noise evidence from AudioSet tagging (AST) plus a speech-vs-floor SNR estimate."""
import numpy as np
import torch

from . import config

_fe = None
_model = None
CHUNK_SECONDS = 10.0
MAX_FULL_CHUNKS = 6
MAX_GAP_CHUNKS = 3

# Output phrase -> AudioSet labels. Speech-like labels are intentionally excluded,
# except for broadcast sources (TV/radio) which are a common call background.
NOISE_GROUPS = {
    "TV": ["Television", "Radio"],
    "music": ["Music", "Singing", "Musical instrument", "Pop music", "Background music", "Song", "Hip hop music",
              "Rock music", "Electronic music", "Christian music", "Theme music", "Jingle (music)"],
    # "Distortion" is deliberately absent: clipping is an audio-quality issue, not background noise.
    "static": ["Static", "White noise", "Pink noise", "Hiss", "Crackle", "Crunch", "Noise", "Buzz", "Hum",
               "Mains hum", "Environmental noise", "Sizzle"],
    "road noise": ["Vehicle", "Car", "Traffic noise, roadway noise", "Engine", "Motor vehicle (road)",
                   "Car passing by", "Truck", "Bus", "Idling", "Accelerating, revving, vroom", "Road noise"],
    "wind": ["Wind", "Wind noise (microphone)", "Rustle"],
    "background voices": ["Chatter", "Crowd", "Hubbub, speech noise, speech babble", "Children shouting",
                          "Babbling", "Conversation"],
    "keyboard typing": ["Typing", "Computer keyboard", "Typewriter"],
    "mechanical noise": ["Mechanisms", "Machine", "Air conditioning", "Mechanical fan", "Vacuum cleaner",
                         "Power tool", "Drill", "Engine starting", "Printer"],
    "dog barking": ["Dog", "Bark", "Bow-wow", "Yip"],
    "baby crying": ["Baby cry, infant cry", "Crying, sobbing"],
    "siren": ["Siren", "Emergency vehicle", "Police car (siren)", "Ambulance (siren)", "Fire engine, fire truck (siren)"],
    "kitchen noise": ["Dishes, pots, and pans", "Cutlery, silverware", "Water tap, faucet", "Sink (filling or washing)",
                      "Frying (food)", "Microwave oven", "Blender"],
}


def _get_model():
    global _fe, _model
    if _model is None:
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        _fe = ASTFeatureExtractor.from_pretrained(config.AUDIO_TAGGING_MODEL)
        model = ASTForAudioClassification.from_pretrained(config.AUDIO_TAGGING_MODEL).eval()
        _model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        _model.config = model.config
    return _fe, _model


def _chunks(x: np.ndarray, sr: int, min_seconds: float = 1.0, limit: int | None = None):
    n = int(CHUNK_SECONDS * sr)
    chunks = [x[i: i + n] for i in range(0, len(x), n)]
    chunks = [c for c in chunks if len(c) >= min_seconds * sr]
    if limit and len(chunks) > limit:
        idx = np.linspace(0, len(chunks) - 1, limit).round().astype(int)
        chunks = [chunks[i] for i in idx]
    return chunks


def _tag(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.zeros((0, 527), dtype=np.float32)
    fe, model = _get_model()
    out = []
    for i in range(0, len(chunks), 4):
        batch = fe(chunks[i: i + 4], sampling_rate=config.SAMPLE_RATE, return_tensors="pt")
        with torch.inference_mode():
            out.append(torch.sigmoid(model(**batch).logits).numpy())
    return np.concatenate(out)


def _group_scores(probs: np.ndarray) -> dict:
    _, model = _get_model()
    label2id = {v: int(k) for k, v in model.config.id2label.items()}
    if probs.shape[0] == 0:
        return {g: 0.0 for g in NOISE_GROUPS}
    scores = {}
    for g, labels in NOISE_GROUPS.items():
        ids = [label2id[l] for l in labels if l in label2id]
        per_chunk = probs[:, ids].max(axis=1)
        # Mean of the top-2 chunks: robust to a single spurious chunk, still catches intermittent noise.
        top = np.sort(per_chunk)[-2:]
        scores[g] = round(float(top.mean()), 3)
    return scores


def _top_labels(probs: np.ndarray, k: int = 8) -> list:
    _, model = _get_model()
    if probs.shape[0] == 0:
        return []
    m = probs.max(axis=0)
    idx = np.argsort(m)[::-1][:k]
    return [(model.config.id2label[int(i)], round(float(m[i]), 3)) for i in idx]


def snr_estimate(audio: np.ndarray, nonspeech: np.ndarray) -> dict:
    sr = config.SAMPLE_RATE
    hop = int(0.02 * sr)
    n = len(audio) // hop
    if n == 0:
        return {}
    frames = audio[: n * hop].reshape(n, hop)
    energy = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-10)
    ns = nonspeech[: n * hop].reshape(n, hop).all(axis=1)
    sp_e = energy[~ns]
    ns_e = energy[ns]
    speech_level = float(np.percentile(sp_e, 90)) if sp_e.size else float(np.percentile(energy, 95))
    floor_ns = float(np.percentile(ns_e, 50)) if ns_e.size > 25 else None
    floor_all = float(np.percentile(energy, 10))
    floor = floor_ns if floor_ns is not None else floor_all
    return {
        "speech_level_db": round(speech_level, 1),
        "noise_floor_db": round(floor, 1),
        "snr_db": round(speech_level - floor, 1),
        "nonspeech_seconds": round(float(ns.sum() * 0.02), 1),
    }


FLOOR_DB = -90.0
CLICK_JUMP_DB = 20.0
CLICK_MAX_FRAMES = 3
CLICK_MIN_REL_SPEECH_DB = -40.0


def gap_signal_stats(audio: np.ndarray, nonspeech: np.ndarray) -> dict:
    """Level, short-impulse rate and high-band level of the audio between speech regions.

    Levels are relative to the speech level so the stats do not depend on recording gain,
    and energies are floored so digital silence does not create spurious jumps.
    """
    from scipy.ndimage import median_filter
    from scipy.signal import butter, sosfilt

    sr = config.SAMPLE_RATE
    hop = int(0.005 * sr)
    n = len(audio) // hop
    if n == 0:
        return {}
    frames = audio[: n * hop].reshape(n, hop)
    e = np.maximum(10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12), FLOOR_DB)
    ns = nonspeech[: n * hop].reshape(n, hop).all(axis=1)
    gap_seconds = float(ns.sum() * 0.005)
    speech_level = float(np.percentile(e[~ns], 90)) if (~ns).sum() > 20 else float(np.percentile(e, 95))
    if gap_seconds < 1.0:
        return {"gap_seconds": round(gap_seconds, 1), "speech_level_db": round(speech_level, 1)}

    # Short impulses: loud 5 ms frames well above the local median that end within 15 ms.
    jump = e - median_filter(e, size=41)
    hot = ns & (jump > CLICK_JUMP_DB) & (e > speech_level + CLICK_MIN_REL_SPEECH_DB)
    padded = np.concatenate([[False], hot, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    runs = edges[1::2] - edges[::2]
    clicks = int(np.sum(runs <= CLICK_MAX_FRAMES))

    sos = butter(4, 3000, btype="highpass", fs=sr, output="sos")
    hf = sosfilt(sos, audio[nonspeech][: sr * 60])
    hf_db = max(10 * np.log10(np.mean(hf ** 2) + 1e-12), FLOOR_DB)
    ge = e[ns]
    floor = float(np.percentile(ge, 50))
    return {
        "gap_seconds": round(gap_seconds, 1),
        "speech_level_db": round(speech_level, 1),
        "gap_floor_db": round(floor, 1),
        "gap_p99_db": round(float(np.percentile(ge, 99)), 1),
        "floor_snr_db": round(speech_level - floor, 1),
        "events_rel_speech_db": round(float(np.percentile(ge, 99)) - speech_level, 1),
        "clicks_per_min": round(clicks / (gap_seconds / 60), 1),
        "gap_hf_rel_speech_db": round(hf_db - speech_level, 1),
    }


def analyze_noise(audio: np.ndarray, nonspeech: np.ndarray) -> dict:
    sr = config.SAMPLE_RATE
    gap_audio = audio[nonspeech]
    gap_probs = _tag(_chunks(gap_audio, sr, limit=MAX_GAP_CHUNKS))
    full_probs = _tag(_chunks(audio, sr, min_seconds=2.0, limit=MAX_FULL_CHUNKS))
    _, model = _get_model()
    label2id = {v: int(k) for k, v in model.config.id2label.items()}
    tv_label = float(full_probs[:, label2id["Television"]].max()) if full_probs.shape[0] else 0.0
    return {
        "gap_groups": _group_scores(gap_probs),
        "full_groups": _group_scores(full_probs),
        "tv_label_max": round(tv_label, 3),
        "gap_top": _top_labels(gap_probs),
        "full_top": _top_labels(full_probs),
        "snr": snr_estimate(audio, nonspeech),
        "gap_stats": gap_signal_stats(audio, nonspeech),
    }
