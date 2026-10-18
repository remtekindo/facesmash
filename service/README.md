# Face Cloak Service

> **v0.2 (2026-10-18):** the engine is now `prototype/cloak_v2.py` (ensemble + DSSIM budget + EOT), metrics are
> measured on the final photo, and `/report` reports honestly whether Gemini answered. The sections below describe
> the original v0.1 flow where not superseded — see `../docs/api-contract.md` (v0.3 section) and
> `../prototype/RESULTS_v2.md` for current behaviour and numbers.


REST microservice wrapping the validated cloaking prototype (`../prototype/cloak.py`).
It adds an adversarial "cloak" to a face photo so AI facial-recognition models
fail to match the identity — while the photo still looks normal to humans.

**Team API contract:** [`../api-contract.md`](../api-contract.md) (v0.1) —
the Flutter app and Laravel gateway are already building against its mock.

## How it works

1. `POST /protect` receives a photo (multipart).
2. MTCNN detects the face; a LowKey-style PGD attack maximizes the
   InceptionResnetV1 (VGGFace2) embedding distance inside a small L_inf ball
   (8/255), confined to the face region. Every 5 PGD steps the perturbation is
   round-tripped through real JPEG Q75 so it survives social-media recompression.
3. Returns the cloaked PNG (base64) + metrics: detection confidence
   before/after, embedding L2 distance, cosine similarity, JPEG-Q75 survival,
   PSNR.

**Attribution (required in README + pitch deck):** cloaking method based on
[LowKey](https://arxiv.org/abs/2101.07922) (Cherepanova et al., ICLR 2021);
models via `facenet-pytorch`. The team's original work is the mobile app, the
JPEG-robust social-media pipeline, per-photo perturbation randomization, the
before/after evaluation dashboard, and the physical sticker pattern.

## API

### `GET /health`
```json
{ "status": "ok", "models_loaded": true, "version": "0.1" }
```

### `POST /protect`
Multipart form fields:

| Field | Type | Default | Notes |
|---|---|---|---|
| `image` | file (required) | — | jpg/png/webp, ≤ 10 MB, auto-downscaled if > 2048 px |
| `strength` | `low` \| `standard` \| `max` | `standard` | PGD steps 30 / 60 / 90 |
| `seed` | int | random | per-photo randomization: a different seed → a different perturbation pattern (the "randomizer") |
| `jpeg_simulate` | bool | `true` | also report whether protection survives a JPEG Q75 round-trip |

Response `200`:
```json
{
  "protected_image": "<base64 PNG>",
  "metrics": {
    "detection_before": 1.0,
    "detection_after": 0.9997,
    "embedding_distance": 1.5184,
    "cosine_similarity": -0.1527,
    "jpeg_q75_survived": true,
    "jpeg_q75_distance": 1.5149,
    "psnr_db": 32.7
  },
  "meta": {
    "engine": "cloak-py 0.1 (LowKey-style PGD)",
    "surrogate_model": "InceptionResnetV1 / VGGFace2",
    "strength": "standard",
    "seed": 12345,
    "processing_time_ms": 22500,
    "downscaled": false
  }
}
```

Errors (JSON `{"error": "CODE", "message": "..."}`):

| HTTP | `error` | Meaning |
|---|---|---|
| 400 | `BAD_IMAGE` | unreadable / unsupported format |
| 400 | `BAD_STRENGTH` | invalid `strength` value |
| 413 | `TOO_LARGE` | > 10 MB |
| 422 | `NO_FACE` | no face detected |

**Timing:** model load ~2.3 s once at startup; warm request ≈ 16–25 s on CPU
(`standard`). Clients must use a **60 s timeout** and honest progress copy
("Protecting photo… ~20 s").

## Run locally

```bash
cd service
python3 -m venv venv && source venv/bin/activate
# Staged install: torch CPU first (a plain `pip install -r requirements.txt`
# would make pip try to resolve facenet-pytorch's numpy<2.0 constraint against
# the pinned numpy 2.5.2 — the validated prototype runs fine with --no-deps).
pip install torch==2.14.1+cpu torchvision==0.29.1+cpu \
  --index-url https://download.pytorch.org/whl/cpu
pip install --no-deps facenet-pytorch==2.6.0
pip install pillow==12.3.0 numpy==2.5.2 requests==2.34.2 urllib3==2.8.0 \
  certifi==2026.7.22 charset-normalizer==3.5.2 idna==3.20 tqdm==4.70.1
pip install fastapi uvicorn python-multipart
uvicorn app:app --host 0.0.0.0 --port 8080
# health check
curl localhost:8080/health
# cloak a photo
curl -X POST localhost:8080/protect \
  -F "image=@photo.jpg" -F "strength=standard" -o result.json
```

## Build & deploy to Cloud Run

Build context must be `~/workspace/ai-builders-cup/` (the image copies
`prototype/cloak.py` in):

```bash
cd ~/workspace/ai-builders-cup
export PROJECT_ID=<gcp-project> REGION=asia-southeast1

# build + push
gcloud builds submit --tag gcr.io/$PROJECT_ID/face-cloak \
  -f service/Dockerfile .

# deploy (2 vCPU / 4 GB; 300 s timeout covers the ~25 s worst case comfortably)
gcloud run deploy face-cloak \
  --image gcr.io/$PROJECT_ID/face-cloak \
  --platform managed --region $REGION \
  --allow-unauthenticated \
  --memory 4Gi --cpu 2 --timeout 300 --port 8080 \
  --min-instances 0 --max-instances 4
```

Model weights are baked into the image at build time (`TORCH_HOME=/app/.torch_cache`),
so cold starts only pay the ~2.3 s model-load cost.

## Firestore job history (Firebase/Firestore — required by the submission brief)

Every completed `POST /protect` writes one document to the Firestore
`protection_jobs` collection: parameters, metrics and verdict only —
photos and personal data are never stored. Uses the Cloud Run default
service account (no key files); the service account needs the
`datastore.user` IAM role. Fail-safe by design: if Firestore is
unavailable the request still succeeds (warning logged). Successful
writes return `meta.firestore_job_id` in the `/protect` response.

Setup (one-time, ~5 min): [`FIRESTORE_SETUP.md`](FIRESTORE_SETUP.md).
Disable: env `FIRESTORE_ENABLED=0`.

## Honest limitations

- **Single surrogate model.** Cloaking is white-box vs InceptionResnetV1/VGGFace2.
  Transferability to unknown commercial models (Meta, TikTok, etc.) is untested —
  do not claim it defeats them. What *is* proven: identity matching breaks
  (embedding distance 1.52 vs ~1.0–1.2 matcher threshold) and survives JPEG Q75.
- **Breaks recognition, not detection.** Face *detection* still finds the face
  (~0.9997 confidence); the product story is anti-recognition/anti-training,
  not invisibility.
- **~16–25 s CPU latency per image** (synchronous). Fine for the prototype and
  demo; production path is a job queue (Cloud Tasks + callback/webhook) and/or
  GPU instances (Cloud Run now supports GPUs) or a distilled faster model.
- One face per image (most confident detection); group photos cloak only the
  primary face — multi-face support is future work.
