# fake-subsai-asr-gateway
#
# Lightweight FastAPI gateway — no GPU, no ML models.
# All heavy work (Whisper ASR, NLLB translation) is delegated to
# asr-transcription-lv:8101 and back-translator-lv:8104 via HTTP.
#
# Build:  docker compose build
# Run:    docker compose up -d
# Logs:   docker logs -f fake-subsai-asr-gateway

FROM python:3.11-slim

# Install curl for Docker healthcheck; libsndfile1 for audio processing.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ .

EXPOSE 9001

# Healthcheck: hit /health every 30s
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=15s \
    CMD curl -sf http://localhost:9001/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9001", "--workers", "1"]
