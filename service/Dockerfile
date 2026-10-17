# Build context MUST be ~/workspace/ai-builders-cup/ (so prototype/cloak.py is included):
#   docker build -f service/Dockerfile -t face-cloak .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/app/.torch_cache \
    PORT=8080

WORKDIR /app

# 1. CPU-only torch FIRST: a plain `pip install facenet-pytorch` would resolve
#    the ~755MB CUDA torch wheel and can exhaust disk on small builders.
COPY service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.14.1+cpu torchvision==0.29.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps facenet-pytorch==2.6.0

# 2. Code: prototype cloak.py stays the single source of truth (copied, never edited here).
COPY prototype/cloak.py ./prototype/cloak.py
COPY service/app.py ./service/app.py
COPY service/firestore_store.py ./service/firestore_store.py

# 3. Pre-download model weights at build time so cold start is ~seconds, not minutes.
RUN python -c "import sys; sys.path.insert(0, 'prototype'); from cloak import load_models; load_models('cpu')"

WORKDIR /app/service
EXPOSE 8080
# Cloud Run injects $PORT; single worker (torch CPU + 60-step PGD is memory-heavy per request).
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
