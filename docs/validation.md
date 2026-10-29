# Validation Results

Only three labeled production calls were provided, and all rule thresholds were
inspected against them, so they cannot measure generalization. Each output field is
therefore validated on public proxy data with a strict dev/test split. Every reported
number below comes from the **held-out test split**, which was run once, after all
tuning was finished.

| Split | Emotion (RAVDESS) | Emotion (CREMA-D, added in v2) | Noise / quality / overlap / silence (synthetic calls) |
|---|---|---|---|
| dev (tuning allowed) | actors 1-12 | odd actor ids (46 speakers) | RAVDESS actors 1-12 + ESC-50 folds 1-3 |
| test (report only) | actors 13-24 | even actor ids (45 speakers) | RAVDESS actors 13-24 + ESC-50 folds 4-5 |

One exception to "run once": the RAVDESS emotion test actors were scored three times in
total. They were scored once for v1, and twice for the v2 emotion model, once at each of
two acoustic thresholds. Section 1b explains why and reports all three. The CREMA-D test
speakers were never used for any choice.

All proxy audio is converted to the telephone channel first: 8 kHz, 300-3400 Hz,
mu-law, then upsampled to 16 kHz. Splits share no speaker and no noise recording.

Reproduce with `python eval/eval_emotion.py` and `python eval/eval_signal.py test`.
Confusion matrices and the reliability plot are in `eval/results/`.

## 1. Emotional tone (5 classes) and intensity

Approaches compared on the same test actors (n = 576 clips):

| Approach | Accuracy | Macro F1 |
|---|---|---|
| A. Handcrafted acoustic features (pitch, energy, spectral, MFCC) + logistic regression | 0.52 | 0.52 |
| B. emotion2vec+ class scores + cue rules (first version) | 0.47 | 0.26 |
| C. emotion2vec+ utterance embedding + logistic regression | 0.82 | 0.80 |
| **B2. Final: C as acoustic evidence + behavioural/text cue rules** | **0.78** | **0.77** |

Emotional intensity (`medium` vs `high`, RAVDESS normal vs strong, n = 432):
the first rules version scored 0.44 accuracy, the embedding head 0.74, and the
final system 0.67. The final system trails the embedding head on purpose: it only
says `high` when the head is at least 70% confident or the behavioural cues are strong.

Proxy label mapping: neutral/calm → neutral; happy → satisfied; disgust and
normal-intensity anger → frustrated; strong anger → upset; fear and strong sadness →
distressed. Surprise and normal-intensity sadness have no matching class and are excluded.

Why the final system (B2) is slightly below C on this proxy: on real calls, the
acoustic head only overrides `neutral` when it is at least 60% confident. The
provided labels are conservative (`call_002` contains an insult but is labelled
neutral), and acted emotion is far more pronounced than real customer speech.
Behavioural cues (repeated "hello?", asking for a real person, profanity, thanks,
distress words) carry the decision on real calls and cannot be tested on RAVDESS,
because every RAVDESS clip reads the same neutral sentences.

Confidence calibration: an isotonic map from raw rule confidence to P(correct),
fitted on grouped 4-fold cross-validation over the dev actors. On the test actors,
the expected calibration error (ECE) drops from **0.110 to 0.054**. Mean calibrated
confidence is 0.80 against an observed accuracy of 0.78.

## 1b. Emotion v2: adding CREMA-D (the production model)

RAVDESS has 24 speakers who all read the same two sentences, and its labels are the
emotions the actors intended. It is a weak test of generalization. v2 adds
[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D): 91 speakers (48 male,
43 female, ages 20-74, several ethnicities) and 12 sentences. Its licence is ODbL.

**Labels come from perception, not intention.** The labels use what crowd raters heard
from the voice alone (`VoiceVote` and `VoiceLevel` in `summaryTable.csv`). Raters heard
"neutral" in more than half of the clips that actors intended as emotional. The trial's
labels are also human judgements of how a caller sounds, so this proxy matches the task
better. The mapping is the same as for RAVDESS:

- neutral → neutral; happy → satisfied; disgust → frustrated; fear → distressed;
- anger with perceived intensity ≥ 65 → upset; other anger → frustrated;
- sadness with perceived intensity ≥ 65 → distressed; other sadness is excluded;
- clips where raters tied are excluded.

The 65 cut-off lies between the mean perceived level of medium-anger clips (60) and
high-anger clips (76). After exclusions, 6,525 clips remain.

**Model selection (dev speakers only, grouped 5-fold CV, production rule path, macro F1):**

| Training data for the embedding heads | RAVDESS dev | CREMA-D dev | Mean |
|---|---|---|---|
| RAVDESS only (v1), C = 0.1 | 0.824 | 0.435 | 0.629 |
| RAVDESS + CREMA-D, C = 0.5 | 0.785 | 0.508 | 0.647 |
| **RAVDESS + CREMA-D, C = 0.5, corpora weighted equally (chosen)** | **0.807** | **0.494** | **0.651** |

On dev CV, the acoustic override threshold (`ACOUSTIC_MIN`) has almost no effect
between 0.4 and 0.7: macro F1 stays within 0.01 on both corpora.

**Held-out test speakers:**

| Test set | v1 heads (RAVDESS only) | v2, threshold 0.6 | v2, threshold 0.7 (production) |
|---|---|---|---|
| RAVDESS actors 13-24 (n = 576), accuracy / macro F1 | 0.78 / 0.77 | 0.78 / 0.77 | 0.77 / 0.76 |
| CREMA-D even actors (n = 3,198), accuracy / macro F1 | 0.54 / 0.41 | 0.62 / 0.47 | **0.63 / 0.47** |
| Both (n = 3,774), accuracy / macro F1 | – | 0.65 / 0.55 | 0.65 / 0.55 |
| Intensity, both, non-neutral (n = 1,775), accuracy | – | 0.69 | 0.68 |
| Confidence ECE, raw → calibrated | – | 0.107 → 0.038 | 0.107 → 0.042 |

What this shows:

- **v1 overstated generalization.** With RAVDESS-only heads, accuracy drops from 0.78 on
  RAVDESS to 0.54 on new speakers with perceptual labels. Adding CREMA-D recovers about
  9 points there and costs at most 1 point on RAVDESS.
- **v2 fixes the "agitated or neutral voice reads as satisfied" error.** On the CREMA-D test
  set (threshold 0.6 run), v1 heads called 415 neutral clips "satisfied" and v2 heads call
  109. This is the same error seen on `call_001`.
- **What the remaining errors are.** On CREMA-D they are mostly neutral ↔ frustrated and
  distressed → neutral. Raters disagree on exactly these clips: audio-only human
  recognition of the intended CREMA-D emotion is reported at about 41%.
- **Intensity is the weakest emotion output.** The system rarely says `high` from the
  voice alone. On real calls, strong behavioural cues drive `high`.

**Why the threshold is 0.7.** The test set was first scored at the original threshold of
0.6. With the v2 heads at that setting, `call_002` became `distressed`: the voice head gave
distressed 0.63, while the label is neutral. Dev CV cannot tell 0.6 and 0.7 apart, so the
stricter end was chosen, which needs more acoustic evidence before overriding neutral. This
decision used one of the three labelled real calls, so it is disclosed here, and the test
numbers are reported at both settings.

## 2. Background noise

The test set has 72 noisy clips (8 ESC-50 noise classes × 3 recordings × SNR 20/10/0 dB)
and 24 clean clips. Clean clips carry a barely perceptible line floor 45 dB below speech,
which per the brief must not count as noise.

| Metric | Result |
|---|---|
| `background_noise_present` accuracy | **0.98** (no false positives; 2 misses, both at 0 dB SNR) |
| `background_noise_severity` accuracy (none/low/medium/high) | **0.90** |
| `background_noise_type` accuracy when detected | 0.70 |
| False "noise" on noise-free but degraded audio (clipping, packet loss, low volume, muffled) | **0 / 16** |

Type accuracy by class: baby crying, dog and keyboard 1.00; mechanical 0.65;
road/engine 0.56; siren 0.67; wind 0.00. Stationary broadband noise (wind, fans)
loses its character through the 300-3400 Hz phone band and is usually reported as
"static". This is a known limitation, not something tuned away.

The 0/16 result on degraded audio checks the brief's rule "do not infer background
noise solely from poor audio quality". Two guards make it hold. First, sound-event
tags are only trusted when gap audio is audible: at least -65 dBFS and no more than
35 dB below speech. Second, "Distortion" was removed from the noise vocabulary,
because clipping is a quality issue, not background noise.

## 3. Audio quality

Test set: 6 base calls, each clean and with 4 degradations at 2 levels (n = 54).

| Metric | Result |
|---|---|
| `audio_quality` accuracy (clear / slightly / severely impaired) | **0.93** (macro F1 0.90) |

Clean, low-volume and muffled audio are classified perfectly at every level. The
errors are mild clipping (0.67) and packet loss (0.83 at each level): with mild loss,
some clips show no detectable dropout. No clean clip was ever marked impaired.

## 4. Speaker overlap and long silence

The second speaker starts 0 or 0.25 s before the first ends (no overlap expected)
or 1 or 2 s before (overlap expected). The long-silence test inserts a gap of
2/5/8 s (no long silence expected) or 12/20 s (long silence expected).

| Field | Accuracy | Detail |
|---|---|---|
| `speaker_overlap_present` | **0.91** | detected 0% at 0 s, 12% at 0.25 s, 88% at 1 s, 88% at 2 s |
| `long_silence_present` | **1.00** | detected 0% at ≤8 s, 100% at ≥12 s |

The thresholds (longest overlapped region ≥ 0.5 s; silence ≥ 10 s) are documented in
`pipeline/config.py`. They are consistent with all three labelled calls, whose
longest overlaps are 0.34 s (labelled false), 0.74 s and 1.52 s (labelled true), and
whose longest silences, 3.9-8.9 s, are all labelled false.

## 5. The three labelled production calls

All 8 scored fields match the labels on all three calls under the production settings
(v2 heads, `ACOUSTIC_MIN = 0.7`, positive cues override a weaker escalation cue).
Because the rules were designed while looking at these calls, and because one
threshold choice used `call_002` (section 1b), this is a sanity check, not an
accuracy estimate.

## Caveats

* Proxy data is not production data: acted emotion, synthetic mixtures and simulated
  phone channels. These numbers show the components work as designed. They do not
  predict hidden-set accuracy. CREMA-D with perceptual labels is the more honest of
  the two emotion proxies.
* emotion2vec+ may have seen RAVDESS-like corpora during pre-training, which could
  make the RAVDESS numbers optimistic. CREMA-D is a weaker candidate for leakage.
* RAVDESS is licensed CC BY-NC-SA. CREMA-D is ODbL. The production heads are trained
  on both; a commercial deployment should drop RAVDESS or replace it with
  AutoAce-labelled calls.
* TV/radio background, the noise in `call_002`, has no ESC-50 equivalent and is only
  checked on the labelled call.
