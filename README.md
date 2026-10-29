# AutoAce Voice Analyzer

**Live dashboard:** [https://136-119-243-232.sslip.io](https://136-119-243-232.sslip.io)  
Sign-in: username `autoace`, password `3z0pQ81O8CcuzKA2`.  
**Source:** [https://github.com/Ayuaaaaaa/voice-analyzer](https://github.com/Ayuaaaaaa/voice-analyzer)

Analyses customer phone calls for emotional tone, background noise, audio quality, speaker overlap and
long silences. Every model is open source and runs on CPU inside the deployment. No audio is sent to any
external API.

- **Dashboard:** login, batch upload (folder or ZIP + CSV manifest), validation report, progress,
  per-file errors, label comparison, CSV/JSON download.
- **Technical memo:** [`docs/memo.md`](docs/memo.md)
- **Validation report:** [`docs/validation.md`](docs/validation.md). Raw metrics and confusion matrices
  are in [`eval/results/`](eval/results).
- **Predictions for the three provided calls:** [`outputs/predictions_provided_calls.csv`](outputs/predictions_provided_calls.csv)

## Output

One JSON object per file:

```json
{"emotional_tone": "upset", "emotional_intensity": "high", "background_noise_present": false,
 "background_noise_type": "", "background_noise_severity": "none", "audio_quality": "clear",
 "speaker_overlap_present": false, "long_silence_present": false, "confidence": 0.84}
```

## Batch format

Upload a ZIP file or a folder. Audio files go at the root, together with one CSV file that has the columns
`name` (the exact filename, including the extension) and `result_json` (may be empty). If `result_json`
holds labels, the dashboard compares them field by field and shows a confusion matrix for emotional tone.

The upload is validated before analysis starts. These cases are reported, not fatal:

- rows whose audio is missing;
- audio files not listed in the CSV (still analysed);
- duplicate rows;
- unsupported file types;
- files in sub-folders;
- invalid `result_json`.

A file that cannot be decoded, or is silent, too short or too long, fails on its own; the rest of the
batch continues. The results CSV keeps the input filenames and order.

Supported formats: wav, mp3, ogg/opus, flac, m4a/aac, webm, wma, amr, mp4. Files must be between 0.5 s
and 30 min long.

## Run locally

Requires Python 3.11 and about 6 GB of disk space for the models.

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export HF_TOKEN=hf_...   # read token; accept the terms of pyannote/segmentation-3.0 on huggingface.co first
python scripts/download_models.py                 # one-off, about 2.5 GB

# command line: writes predictions.csv (name,result_json) and predictions.json with reasons
python cli.py path/to/folder --out results

# dashboard on http://127.0.0.1:7860
export VA_USERNAME=admin VA_PASSWORD=choose-one VA_SECRET=$(python -c "import secrets;print(secrets.token_hex(32))")
uvicorn app.main:app --port 7860
```

Optional environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `VA_THREADS` | CPU count | torch threads |
| `VA_WHISPER_MODEL` | `small` | faster-whisper model size |
| `VA_MODEL_DIR` | `models/` | emotion heads and DNSMOS |
| `VA_JOBS_DIR` | `jobs/` | temporary upload storage |
| `VA_JOB_TTL_HOURS` | `24` | how long finished batch results stay available |

## Docker

The hosted dashboard is a Google Cloud `e2-standard-4` VM running `deploy/docker-compose.yml`
(Caddy terminates HTTPS). To run the same stack elsewhere:

```bash
# write deploy/.env (VA_USERNAME, VA_PASSWORD, VA_SECRET, DOMAIN) and deploy/hf_token
# (a Hugging Face read token; accept the pyannote/segmentation-3.0 terms first)
docker compose -f deploy/docker-compose.yml up -d --build
```

Or a single image:

```bash
docker build --secret id=HF_TOKEN,env=HF_TOKEN -t voice-analyzer .
docker run -p 7860:7860 -e VA_USERNAME=admin -e VA_PASSWORD=... -e VA_SECRET=... voice-analyzer
```

All model weights are downloaded at build time. At run time the container works offline.

## Reproducing the validation

The validation datasets are not included. Download them into `VA_DATA_DIR` (default `D:\va_env\data`):

- RAVDESS speech: `ravdess/Actor_*/`
- CREMA-D: `cremad/data/AudioWAV/` plus `cremad/summaryTable.csv`
- ESC-50: `ESC-50-master/`

Then run:

```bash
pip install -r requirements-eval.txt
cd eval
python eval_emotion.py                  # RAVDESS approaches A/B/C (v1)
python extract_cremad.py                # CREMA-D embeddings (~80 min on CPU)
python eval_emotion_combined.py select  # dev-only model selection
python eval_emotion_combined.py test    # single held-out test run + final heads
python eval_signal.py dev noise         # and: quality, overlap, summary; then the same with "test"
```

Latency and cost: `python scripts/benchmark.py path/to/folder --label mymachine`.

## Repository layout

```
pipeline/   analysis: decoding, speech/overlap, ASR, speaker roles, emotion, noise, quality, rules
app/        FastAPI dashboard, batch validation, job queue, static UI
eval/       validation scripts and results
scripts/    model download, benchmark, dashboard end-to-end test
docs/       memo and validation report
models/     trained emotion heads (DNSMOS is downloaded at build time)
```
