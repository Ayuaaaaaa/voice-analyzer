# Technical memo — AutoAce Voice Tone and Background Noise trial

Hosted dashboard: **https://136-119-243-232.sslip.io**
Username `autoace` / password `3z0pQ81O8CcuzKA2`
Source: **https://github.com/Ayuaaaaaa/voice-analyzer**

Predictions for the three provided calls: [`outputs/predictions_provided_calls.csv`](../outputs/predictions_provided_calls.csv).
Validation numbers, splits and confusion matrices: [`docs/validation.md`](validation.md) and [`eval/results/`](../eval/results).

This memo covers architecture, approaches that were tried and dropped, cost, latency,
failure modes, and how confidential audio is handled.

## 1. What the system does

A CPU-only pipeline turns one audio file into the nine-field JSON the brief specifies.
Nothing is sent to a paid or public inference API. Models are open-source weights
downloaded at image build time; at run time the container is offline.

```
decode (soundfile / ffmpeg) → 16 kHz mono
        │
        ├─ pyannote segmentation-3.0  → VAD, overlap regions, silence
        ├─ faster-whisper small (int8) → utterances + language
        ├─ WeSpeaker embeddings        → agent vs customer (opener regex as the anchor)
        ├─ emotion2vec+ + logistic heads + text/behaviour cues → tone, intensity
        ├─ AST (int8) on a few chunks + gap-signal stats → noise present / type / severity
        └─ DNSMOS P.835 + clipping / level / bandwidth / dropouts → audio quality
                │
                └─ pipeline/rules.py  → AnalysisResult JSON + human-readable reasons
```

A FastAPI dashboard sits in front of a one-worker job queue. Upload is a ZIP or a
folder plus a CSV manifest (`name`, `result_json`). Validation reports missing files,
unlisted files, duplicates, unsupported types and nested paths before analysis starts.
A single bad file fails on its own; the rest of the batch continues. Results download
as CSV (exact `name,result_json` schema, filenames preserved) and JSON (with reasons
and per-file errors). Uploaded audio is deleted as soon as that file has been analysed.
Finished jobs expire after 24 hours.

## 2. Approaches tried

### 2.1 Speaker overlap and roles

The first attempt was the full `pyannote/speaker-diarization-3.1` pipeline. It tried
to download a gated community-1 PLDA (403) and, even patched, was too slow at a 0.1 s
step (about 5 minutes for `call_003`). A 0.5 s step was still ~55 s of diarization
alone.

**Kept:** one `Inference` pass of `segmentation-3.0` at a 2.5 s step. The max class
score is VAD; the second-max is overlap. Speaker identity is a cheaper second stage:
embed each ASR utterance with `wespeaker-voxceleb-resnet34-LM`, anchor on the
scripted agent opener (English and Spanish), and label the rest by cosine similarity.
The three provided calls are effectively mono (identical stereo channels), so channel
separation is not available.

### 2.2 Emotion

On telephone-simulated RAVDESS, held-out actors 13–24:

| | Accuracy | Macro F1 |
|---|---|---|
| A. Handcrafted pitch/energy/MFCC + logistic regression | 0.52 | 0.52 |
| B. emotion2vec+ class scores + cue rules | 0.47 | 0.26 |
| C. emotion2vec+ embedding + logistic regression | 0.82 | 0.80 |
| B2 (v1 production): C as acoustic evidence inside the cue rules | 0.78 | 0.77 |

C looks strongest on RAVDESS because every clip is a strongly acted isolated sentence.
On real calls the labels are conservative (`call_002` contains an insult and is still
`neutral`), so the production path keeps the behavioural/text cues in charge and only
lets the acoustic head override `neutral` when it is confident.

v1 heads were trained on RAVDESS only. That number does not travel: the same heads
score 0.54 accuracy / 0.41 macro F1 on held-out CREMA-D speakers with *perceptual*
(audio-only crowd) labels. v2 trains the heads on RAVDESS + CREMA-D, chosen by
grouped 5-fold CV on *dev speakers only*. On the CREMA-D test speakers this raises
accuracy from 0.54 to **0.63** and cuts false "satisfied" on true-neutral clips from
415 to 109. RAVDESS test accuracy stays at 0.77. Details and the one disclosed
threshold look at `call_002` are in `docs/validation.md` §1b.

Text cues (repeated "hello?", asking for a human, profanity, thanks, distress words,
multilingual sentiment) cannot be tested on RAVDESS/CREMA-D, because every clip uses
the same neutral sentences. They are the reason `call_001` is `upset` rather than
`satisfied` even though the voice head still leans happy.

### 2.3 Noise

AST on AudioSet, dynamically quantized to int8, capped at 6 full-file chunks and 3
gap chunks. Sound-event tags are not trusted unless the gap is audible (level and
relative to speech). "Distortion" was removed from the vocabulary: clipping is a
quality issue, not background noise. Stationary noise uses gap-floor SNR, not tags,
because wind and fans lose their identity in the 300–3400 Hz phone band.

A first noise rule flagged 21 of 24 *clean* clips as noisy (digital-silence gaps
looked like clicks). Guards: energy floor at −90 dB, audible-gap tests, tag priority
over the click detector. After that, false "noise" on degraded-but-clean audio is
0/16 on the held-out set.

### 2.4 Quality

DNSMOS SIG on speech regions only (`bak` is ignored, so background noise does not
make the recording look impaired). Plus clipping, speech RMS, 99% spectral rolloff
(muffled) and dropout rate. Dev accuracy went from 0.52 to 0.94 after the rolloff
check and threshold pass; the held-out test is 0.93.

### 2.5 Speed

The first whole-pipeline run was ~0.8× real time on a laptop. Caps (AST chunks, 60 s
of speech-emotion input, quantized AST, segmentation step 2.5 s) brought
`call_003` to ~0.5× on that machine. On the 4-vCPU hosted VM the same three calls
run at about 1.0× overall (short calls slower because of per-file overhead).

### 2.6 Hosting

Hugging Face Docker Spaces on free CPU now require a PRO subscription (HTTP 402).
The dashboard is therefore a Google Cloud `e2-standard-4` VM (4 vCPU, 16 GB) in
`us-central1`, Docker Compose, Caddy with automatic HTTPS on a static IP. Image
build bakes every weight in; `HF_HUB_OFFLINE=1` at run time.

## 3. Cost

The brief's cap is **$0.003 per audio minute**. The system never calls a paid API, so
the only cost is CPU time.

`cost per audio minute = real-time factor × instance $/hour / 60`

Measured on the hosted `e2-standard-4` (the deployment reviewers will use), one pass
over the three provided calls (`eval/results/benchmark_gce_e2-standard-4.json`):

| File | Audio | Processing | RTF |
|---|---|---|---|
| call_001.ogg | 31 s | 31 s | 1.01 |
| call_002.ogg | 35 s | 30 s | 0.85 |
| call_003.ogg | 172 s | 106 s | 0.62 |
| **overall** | **238 s** | **167 s** | **0.70** |

On-demand list price for `e2-standard-4` in `us-central1` is about **$0.134 / hour**.
That is **$0.00157 per audio minute**, under the cap. A cheaper 4-vCPU (Hetzner CX32
at $0.0113 / hour) would be about **$0.00013 / min**. Hugging Face's former free CPU
tier would have been $0 / min; it is no longer available without PRO.

Idle cost of leaving the trial VM up 24/7 is about $100 / month plus a 40 GB disk
(~$4). That is paid from the Google Cloud $300 trial credit, not per-minute
inference. After the trial the instance should be deleted (`gcloud compute instances
delete va-server --zone=us-central1-a`).

Batching does not change the per-minute figure: one worker, files in sequence, CPU
bound. Two workers on this size of machine would oversubscribe RAM (models are ~6 GB
resident) and would not cut the per-file wall time.

No audio, embeddings or transcripts are sent to OpenAI, AssemblyAI, or similar. A
whisper-large or LLM-judge path was not used, because it would either miss the cost
cap or send confidential audio off-box.

## 4. Latency

Same hosted VM, models already loaded in the worker (the dashboard warms up at
start):

- **Cold start** (process boot + load every model): 33 s on this VM, ~60 s on the
  development laptop. The UI shows "loading models" until that finishes; uploads
  queue behind it.
- **Per file:** 30 s, 30 s and 106 s on the three provided calls. Short files sit
  near 1× real time because ASR, AST and DNSMOS have per-file overhead; a ~3 min
  call is 0.62×.
- **ETA** on the dashboard is `mean(elapsed so far) × remaining files`.
- **Upload** is limited to 2 GB and 5,000 files. Decode rejects clips shorter than
  0.5 s or longer than 30 min.

Stage share on this VM, over the three calls: ASR 42%, noise tagging 23%, speech
emotion 13%, speaker embeddings 7%, quality 5%, VAD/overlap 5%, the rest under 4%.

## 5. Validation summary (held-out only)

Full tables in `docs/validation.md`. Headline test-split numbers:

| Field | Proxy | Accuracy |
|---|---|---|
| emotional_tone, 5-way | CREMA-D even actors (n = 3,198) | 0.63 |
| emotional_tone, 5-way | RAVDESS actors 13–24 (n = 576) | 0.77 |
| emotional_intensity | both, non-neutral | 0.68 |
| background_noise_present | ESC-50 mixes (n = 96) | 0.98 (0 false positives) |
| background_noise_severity | same | 0.90 |
| background_noise_type | when detected | 0.70 |
| audio_quality | 6 calls × degradations (n = 54) | 0.93 |
| speaker_overlap_present | synthetic two-talker | 0.91 |
| long_silence_present | synthetic gaps | 1.00 |
| noise inferred from poor quality | clipped / muffled / lossy / quiet (n = 16) | 0 / 16 |

The three provided calls match on all 8 scored fields. That is a sanity check.

## 6. Failure modes and next steps

**Emotion.** 0.63 on CREMA-D with perceptual labels is the honest number. Remaining
errors are mostly neutral ↔ frustrated and distressed → neutral, which is where
human audio-only raters also disagree (reported ~41% on intended CREMA-D emotion).
The voice head still sometimes reads agitation as `satisfied`; the cue rules and a
0.7 acoustic override are the guard. Intensity is weak from acoustics alone (`high`
is driven by behavioural cues on real calls).

Next: fine-tune a small multilingual speech-emotion model on AutoAce-labelled calls
once they exist. Do not chase 95% on RAVDESS.

**Noise type.** Wind is 0.00, road/engine 0.56, mechanical 0.65 after the telephone
band. Stationary broadband noise is reported as `static` / `background hum`. Presence
and severity stay usable. Next: a small supervised classifier on gap embeddings,
trained on telephone-filtered ESC-50, rather than more AST thresholds.

**Overlap.** 0.25 s overlaps are mostly missed; 1–2 s are usually caught. The 0.5 s
region threshold matches the labelled calls. Next: keep the threshold unless AutoAce
wants finer "talked over the agent" flags.

**Language.** ASR and sentiment are multilingual; emotion2vec+ is too. `call_002` is
Spanish and is handled. Very short or heavily overlapped Spanish may still mis-assign
the agent if the opener regex misses.

**Operational.** One worker; a second concurrent batch waits in the queue. A 30-minute
file is rejected rather than OOMing the 16 GB VM. Corrupt / silent / empty / wrong-type
files fail per row. After a VM reboot, Caddy and the app restart (`restart: always`);
the first minute is cold start.

**Hosting.** The Google Cloud trial stops the VM if the $300 credit runs out or 90
days pass, unless the account is upgraded. Delete the instance after the review.

## 7. Data handling

- Audio never leaves the VM. There is no cloud ASR, no LLM, no webhook.
- `HF_TOKEN` is a build-time secret, used once to fetch `pyannote/segmentation-3.0`
  and the WeSpeaker embedder (gated models, terms accepted). Runtime is
  `HF_HUB_OFFLINE=1`.
- Each uploaded file is deleted from disk when its analysis finishes; the upload
  directory is removed when the job ends. Job records (JSON results only) last 24 h.
- Login is a signed HttpOnly cookie, rate-limited. Credentials are environment
  variables, not in the image.
- `robots.txt` disallows crawling. The dashboard is a login wall.

## 8. Reproduce

```bash
pip install -r requirements.txt
export HF_TOKEN=... VA_USERNAME=... VA_PASSWORD=... VA_SECRET=...
python scripts/download_models.py
python cli.py path/to/folder --out results
uvicorn app.main:app --port 7860
```

Docker: `deploy/docker-compose.yml` (needs `deploy/.env` and `deploy/hf_token`).
Validation: `docs/validation.md` and `eval/`.
Dashboard end-to-end test: `python scripts/test_dashboard.py <calls_folder> --url https://136-119-243-232.sslip.io`.
Latency/cost: `python scripts/benchmark.py <files> --label mymachine`.
