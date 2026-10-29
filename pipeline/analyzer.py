import time

import torch

from . import config
from .audio import load_audio

torch.set_num_threads(config.NUM_THREADS)


class _Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, name):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.t0 = time.perf_counter()

            def __exit__(self, *exc):
                timer.t[name] = round(time.perf_counter() - self.t0, 2)

        return _Ctx()


def extract_features(path: str) -> dict:
    from . import emotion, noise, quality, speakers, speech
    from .asr import transcribe

    tm = _Timer()
    with tm("decode"):
        audio, info = load_audio(path)
    with tm("speech"):
        sp = speech.analyze_speech(audio)
    with tm("asr"):
        utts, lang, lang_prob = transcribe(audio)
    with tm("speakers"):
        spk = speakers.assign_speakers(audio, utts)
    customer = [u for u in utts if u.speaker == "customer"]
    with tm("emotion_acoustic"):
        ser = emotion.acoustic_emotion(audio, customer)
    with tm("emotion_text"):
        txt = emotion.text_emotion(customer)
    with tm("prosody"):
        pros = emotion.prosody(audio, utts)
    with tm("noise"):
        ns_mask = speech.non_speech_mask(len(audio), sp.speech_regions)
        nz = noise.analyze_noise(audio, ns_mask)
    with tm("quality"):
        q = {"dnsmos": quality.dnsmos(audio, sp.speech_regions), **quality.signal_checks(audio, info, sp.speech_regions)}

    return {
        "duration": round(info["duration"], 2),
        "audio_info": info,
        "speech": {
            "speech_ratio": round(sp.speech_ratio, 3),
            "overlap_seconds": round(sp.overlap_seconds, 2),
            "max_overlap_region": round(sp.max_overlap_region, 2),
            "n_overlap_regions": len(sp.overlap_regions),
            "max_silence": round(sp.max_silence, 2),
            "max_internal_silence": round(sp.max_internal_silence, 2),
            "lead_silence": round(sp.lead_silence, 2),
            "trail_silence": round(sp.trail_silence, 2),
        },
        "language": lang,
        "language_prob": round(lang_prob, 2),
        "utterances": [
            {"start": round(u.start, 1), "end": round(u.end, 1), "speaker": u.speaker, "text": u.text} for u in utts
        ],
        "speaker_assignment": spk,
        "emotion_acoustic": ser,
        "emotion_text": txt,
        "prosody": pros,
        "noise": nz,
        "quality": q,
        "timings": tm.t,
    }


def warmup():
    """Load every model once so the first uploaded file is not slowed by model loading."""
    from . import asr, emotion, noise, quality, speakers, speech

    speech._get_inference()
    asr._get_model()
    speakers._get_embedder()
    emotion._get_ser()
    emotion._get_sentiment()
    emotion._get_heads()
    noise._get_model()
    quality._get_session()


def analyze(path: str) -> dict:
    """Full analysis of one file: {'result': schema dict, 'reasons': [...], 'features': {...}}."""
    from .rules import decide

    t0 = time.perf_counter()
    feats = extract_features(path)
    result, reasons = decide(feats)
    feats["timings"]["total"] = round(time.perf_counter() - t0, 2)
    return {"result": result.model_dump(), "reasons": reasons, "features": feats}
