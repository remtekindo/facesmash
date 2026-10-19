# Build context MUST be the repo root (so prototype/cloak_v2.py is included). Cloud Build: pass --timeout=1800
# (default 10 min can be too short for torch + 2 model downloads):
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

# 2. ArcFace-family ONNX weights (insightface model zoo, GitHub release v0.7).
#    mbf  = MobileFaceNet, optimised against (cheap);  r50 = ResNet50, HOLD-OUT (verification only).
#    NOTE: insightface pretrained models are research / non-commercial licensed.
COPY service/fetch_models.py ./fetch_models.py
RUN python fetch_models.py /app/models
ENV FACESMASH_MBF_ONNX=/app/models/w600k_mbf.onnx \
    FACESMASH_ARCFACE_ONNX=/app/models/w600k_r50.onnx

# 3. Code: prototype/cloak_v2.py is the engine (cloak.py = legacy v1, kept in the repo only).
COPY prototype/cloak_v2.py ./prototype/cloak_v2.py
COPY prototype/cloak_advanced.py ./prototype/cloak_advanced.py
COPY service/app.py ./service/app.py
COPY service/firestore_store.py ./service/firestore_store.py

# 4. Pre-download model weights at build time so cold start is ~seconds, not minutes.
RUN python -c "import sys; sys.path.insert(0, 'prototype'); import cloak_v2 as C; from facenet_pytorch import MTCNN; MTCNN(); print(C.Ensemble(('vggface2','casia','mbf','arcface')).names)"

WORKDIR /app/service
EXPOSE 8080
# Cloud Run injects $PORT; single worker. Deploy with --concurrency=1 --cpu=2 --memory=4Gi --timeout=300.
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
