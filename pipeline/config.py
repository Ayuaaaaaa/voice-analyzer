import os

SAMPLE_RATE = 16000

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_DIR = os.environ.get("VA_MODEL_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "models"))

SEGMENTATION_MODEL = "pyannote/segmentation-3.0"
SPEAKER_EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
WHISPER_MODEL = os.environ.get("VA_WHISPER_MODEL", "small")
EMOTION_MODEL = "emotion2vec/emotion2vec_plus_base"
TEXT_SENTIMENT_MODEL = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
AUDIO_TAGGING_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
DNSMOS_URL = "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"

NUM_THREADS = int(os.environ.get("VA_THREADS", os.cpu_count() or 4))

MAX_FILE_SECONDS = 30 * 60
MIN_FILE_SECONDS = 0.5

# Speech activity / overlap (pyannote segmentation-3.0)
SEG_STEP_SECONDS = 2.5
VAD_ONSET = 0.5
OSD_ONSET = 0.5
OVERLAP_MIN_REGION_SECONDS = 0.5
LONG_SILENCE_SECONDS = 10.0
