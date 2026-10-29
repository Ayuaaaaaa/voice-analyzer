"""Turn extracted features into the required output schema.

Every threshold lives here so it can be documented and tuned in one place.
Each decision also records a short human-readable reason.
"""
from . import config
from .schema import AnalysisResult

# Background noise
TV_LABEL_MIN = 0.12
GROUP_GAP_MIN = 0.15
GROUP_FULL_MIN = 0.30
GROUP_HINT_MIN = 0.05
STATIC_CLICKS_PER_MIN = 15.0
STATIC_HF_REL_DB = -43.0
STATIC_GROUP_MIN = 0.15
STATIONARY_SNR_DB = 35.0
STATIONARY_MIN_FLOOR_DB = -75.0
TAG_MIN_GAP_DB = -65.0
TAG_MIN_REL_SPEECH_DB = -35.0
SEVERITY_EVENTS_LOW_DB = -10.0
SEVERITY_SNR_MEDIUM = 20.0
SEVERITY_SNR_HIGH = 10.0
SPEECH_LIKE_GROUPS = {"TV", "background voices"}

# Emotion: probability the voice head needs to override "neutral" on its own.
# Dev CV is flat between 0.5 and 0.7 (macro-F1 within 0.01); the stricter end is used because
# acted speech is far more expressive than real callers.
ACOUSTIC_MIN = 0.7
ACOUSTIC_SUPPORTED_MIN = 0.35
HIGH_INTENSITY_MIN = 0.7

# Audio quality (DNSMOS SIG on speech only; background noise is deliberately not used)
SIG_SEVERE = 2.2
SIG_SLIGHT = 2.7
CLIP_SEVERE = 0.02
CLIP_SLIGHT = 0.0008
RMS_SEVERE = -60.0
RMS_SLIGHT = -42.0
ROLLOFF_SEVERE_HZ = 900
ROLLOFF_SLIGHT_HZ = 1600
DROPOUTS_SEVERE_PER_MIN = 20.0
DROPOUTS_SLIGHT_PER_MIN = 1.0


_SEV_ORDER = ["none", "low", "medium", "high"]


def _noise(f: dict, reasons: list) -> tuple[bool, str, str, float]:
    nz = f["noise"]
    gs = nz.get("gap_stats", {})
    gap, full = nz["gap_groups"], nz["full_groups"]

    # 1) Sound-event tags (AST), excluding speech-like groups except an explicit TV tag.
    #    Tags are ignored when gap sound is near digital silence or far below speech
    #    ("barely perceptible artifacts should not automatically count").
    tags = {}
    audible_gaps = (gs.get("gap_p99_db", 0.0) >= TAG_MIN_GAP_DB
                    and gs.get("events_rel_speech_db", 0.0) >= TAG_MIN_REL_SPEECH_DB)
    if audible_gaps and nz["tv_label_max"] >= TV_LABEL_MIN:
        tags["TV"] = nz["tv_label_max"] / TV_LABEL_MIN
    for g in gap if audible_gaps else []:
        if g in SPEECH_LIKE_GROUPS or g == "static":
            continue
        ev = max(gap[g] / GROUP_GAP_MIN, full[g] / GROUP_FULL_MIN)
        if ev >= 1.0:
            tags[g] = ev
    # 2) Static/line noise from the signal itself.
    clicks = gs.get("clicks_per_min", 0.0)
    hf_rel = gs.get("gap_hf_rel_speech_db", -90.0)
    click_ev = clicks / STATIC_CLICKS_PER_MIN
    hf_ev = 1.0 + (hf_rel - STATIC_HF_REL_DB) / 6.0 if (audible_gaps and hf_rel >= STATIC_HF_REL_DB) else 0.0
    static_ev = max(click_ev, hf_ev, gap["static"] / STATIC_GROUP_MIN if audible_gaps else 0.0)
    # 3) Raised stationary floor (hum, fan, engine, wind...) from speech-to-floor SNR.
    floor_snr = gs.get("floor_snr_db", 99.0)
    stationary = floor_snr < STATIONARY_SNR_DB and gs.get("gap_floor_db", -120.0) >= STATIONARY_MIN_FLOOR_DB

    if not tags and static_ev < 1.0 and not stationary:
        reasons.append(f"noise: none (best tag x{max(tags.values(), default=0):.1f}, static x{static_ev:.1f}, floor SNR {floor_snr} dB)")
        margin = min(1.0 - static_ev, (floor_snr - STATIONARY_SNR_DB) / 20.0)
        return False, "", "none", max(0.3, min(1.0, 0.6 + margin))

    hint = {g: max(gap[g], full[g] * 0.5) for g in gap if g not in SPEECH_LIKE_GROUPS and g != "static"}
    if tags:
        kind = max(tags, key=tags.get)
        strength = tags[kind]
    elif stationary and hint and max(hint.values()) >= GROUP_HINT_MIN:
        kind = max(hint, key=hint.get)
        strength = 1.0 + (STATIONARY_SNR_DB - floor_snr) / 10.0
    elif static_ev >= 1.0:
        kind = "sharp static" if click_ev >= 1.0 else "static"
        strength = static_ev
    else:
        kind = "background hum"
        strength = 1.0 + (STATIONARY_SNR_DB - floor_snr) / 10.0

    # Severity: the worse of transient events (loudest gap events vs speech) and stationary floor.
    rel = gs.get("events_rel_speech_db", -90.0)
    sev_events = "low" if rel < SEVERITY_EVENTS_LOW_DB else "high" if (strength >= 3.0 and rel >= 0) else "medium"
    if floor_snr < SEVERITY_SNR_HIGH:
        sev_floor = "high"
    elif floor_snr < SEVERITY_SNR_MEDIUM:
        sev_floor = "medium"
    elif stationary:
        sev_floor = "low"
    else:
        sev_floor = "none"
    sev = max(sev_events if (tags or static_ev >= 1.0) else "none", sev_floor, key=_SEV_ORDER.index)
    sev = "low" if sev == "none" else sev
    reasons.append(
        f"noise: {kind} (evidence x{strength:.1f}; gap events {rel:+.0f} dB vs speech; floor SNR {floor_snr} dB) -> {sev}"
    )
    return True, kind, sev, max(0.3, min(1.0, 0.5 + (strength - 1.0) / 4.0))


def _quality(f: dict, reasons: list) -> tuple[str, float]:
    q = f["quality"]
    sig = q["dnsmos"]["sig"]
    dur_min = max(f["duration"] / 60, 1e-6)
    drop_rate = q["dropouts"] / dur_min
    roll = q["rolloff_99_hz"]
    severe = (sig < SIG_SEVERE or q["clip_fraction"] > CLIP_SEVERE or q["speech_rms_db"] < RMS_SEVERE
              or drop_rate >= DROPOUTS_SEVERE_PER_MIN or roll < ROLLOFF_SEVERE_HZ)
    slight = (sig < SIG_SLIGHT or q["clip_fraction"] > CLIP_SLIGHT or q["speech_rms_db"] < RMS_SLIGHT
              or drop_rate >= DROPOUTS_SLIGHT_PER_MIN or roll < ROLLOFF_SLIGHT_HZ)
    label = "severely_impaired" if severe else "slightly_impaired" if slight else "clear"
    reasons.append(
        f"quality: DNSMOS SIG {sig}, clipping {q['clip_fraction']:.4f}, speech level {q['speech_rms_db']} dB, "
        f"bandwidth {roll} Hz, dropouts/min {drop_rate:.1f} -> {label}"
    )
    margin = min(abs(sig - SIG_SLIGHT), abs(sig - SIG_SEVERE))
    return label, min(1.0, 0.5 + margin)


def _emotion(f: dict, reasons: list) -> tuple[str, str, float]:
    ser = f["emotion_acoustic"]
    txt = f["emotion_text"]
    mean = ser.get("mean", {})
    peak = ser.get("peak", {})
    sent = txt.get("sentiment", {})
    pros = f.get("prosody", {})

    esc, pos, distress = 0.0, 0.0, 0.0
    why = []
    if txt.get("hello_repeats", 0) >= 3:
        esc += 2.0
        why.append(f"repeated hello x{txt['hello_repeats']}")
    if txt.get("robot_check", 0):
        esc += 1.0
        why.append("asked for a real person")
    if txt.get("profanity", 0):
        esc += 0.5
        why.append("profanity")
    if sent.get("negative", 0) >= 0.6:
        esc += 1.0
        why.append("negative wording")
    if mean.get("angry", 0) + mean.get("disgusted", 0) >= 0.25:
        esc += 2.0
        why.append("angry voice")
    elif peak.get("angry", 0) >= 0.6:
        esc += 1.0
        why.append("angry moments")
    c, a = pros.get("customer"), pros.get("agent")
    if c and a and a["f0_iqr"] > 0 and c["f0_iqr"] / a["f0_iqr"] >= 2.0 and esc > 0:
        esc += 1.0
        why.append("agitated pitch")
    if txt.get("distress_words", 0):
        distress += 2.0
        why.append("distress words")
    if mean.get("fearful", 0) + mean.get("sad", 0) >= 0.5 and esc >= 1:
        distress += 1.0
        why.append("fearful/sad voice")
    if txt.get("thanks", 0):
        pos += 1.0
        why.append("thanks")
    if sent and sent.get("positive", 0) > max(sent.get("neutral", 0), sent.get("negative", 0)) and sent["positive"] >= 0.4:
        pos += 1.0
        why.append("positive wording")
    if mean.get("happy", 0) >= 0.25:
        pos += 1.0
        why.append("happy voice")

    probs = ser.get("tone_probs", {})
    p_high = ser.get("p_high")
    acoustic = max(probs, key=probs.get) if probs else "neutral"
    p_acoustic = probs.get(acoustic, 0.0)
    if probs:
        why.append(f"voice model {acoustic} {p_acoustic:.2f}")

    # Behavioural/text cues win; the acoustic head only overrides neutral when it is confident.
    if distress >= 2.0:
        tone = "distressed"
    elif esc >= 3.0:
        tone = "upset"
    elif esc >= 1.5:
        tone = "upset" if acoustic == "upset" and p_acoustic >= ACOUSTIC_MIN else "frustrated"
    elif pos >= 2.0 and pos > esc:
        tone = "satisfied"
    elif acoustic != "neutral" and p_acoustic >= ACOUSTIC_MIN:
        tone = acoustic
    elif acoustic == "satisfied" and p_acoustic >= ACOUSTIC_SUPPORTED_MIN and pos >= 1.0:
        tone = "satisfied"
    else:
        tone = "neutral"

    if tone in ("upset", "distressed") and esc + distress >= 4.0:
        intensity = "high"
    elif tone != "neutral" and p_high is not None and p_high >= HIGH_INTENSITY_MIN:
        intensity = "high"
    elif ser.get("n", 0) == 0 and txt.get("chars", 0) < 15:
        intensity = "low"
    else:
        intensity = "medium"

    cue = {"distressed": distress, "upset": esc, "frustrated": esc, "satisfied": pos, "neutral": 0.0}[tone]
    agree = probs.get(tone, 0.0) if probs else 0.3
    conf = 0.35 + 0.1 * min(cue, 3.0) + 0.35 * agree
    reasons.append(f"emotion: escalation {esc}, positive {pos}, distress {distress} [{', '.join(why) or 'no cues'}] -> {tone}/{intensity}")
    return tone, intensity, max(0.2, min(0.95, conf))


def decide(f: dict) -> tuple[AnalysisResult, list[str]]:
    from .emotion import calibrate_confidence

    reasons: list[str] = []
    sp = f["speech"]
    tone, intensity, c_emo = _emotion(f, reasons)
    c_emo = calibrate_confidence(c_emo)
    present, kind, sev, c_noise = _noise(f, reasons)
    quality, c_q = _quality(f, reasons)

    overlap = sp["max_overlap_region"] >= config.OVERLAP_MIN_REGION_SECONDS
    reasons.append(f"overlap: longest overlapped region {sp['max_overlap_region']}s -> {overlap}")
    silence = max(sp["max_internal_silence"], sp["lead_silence"]) >= config.LONG_SILENCE_SECONDS
    reasons.append(f"silence: longest gap {max(sp['max_internal_silence'], sp['lead_silence'])}s -> {silence}")

    c_overlap = min(1.0, 0.5 + abs(sp["max_overlap_region"] - config.OVERLAP_MIN_REGION_SECONDS))
    confidence = 0.4 * c_emo + 0.25 * c_noise + 0.15 * c_q + 0.2 * c_overlap
    result = AnalysisResult(
        emotional_tone=tone,
        emotional_intensity=intensity,
        background_noise_present=present,
        background_noise_type=kind,
        background_noise_severity=sev,
        audio_quality=quality,
        speaker_overlap_present=overlap,
        long_silence_present=silence,
        confidence=max(0.05, min(0.99, confidence)),
    )
    return result, reasons
