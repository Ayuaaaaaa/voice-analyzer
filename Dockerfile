FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    MODELSCOPE_CACHE=/home/user/.cache/modelscope \
    VA_JOBS_DIR=/tmp/va_jobs \
    VA_THREADS=2 \
    OMP_NUM_THREADS=2 \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Bake all model weights into the image so a cold start needs no downloads.
# HF_TOKEN (read access, pyannote terms accepted) is passed as a build secret, never stored in a layer.
RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=true \
    HF_TOKEN=$(cat /run/secrets/HF_TOKEN) python scripts/download_models.py
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
