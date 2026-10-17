# Face Smash

Anti-AI-facial-recognition for everyday photos. Upload a photo, get back a visually near-identical
image whose face embedding is pushed far away from your identity — so AI scrapers and
face-search engines can't match it back to you.

Built for the **Google Cloud AI Builders Cup 2026** (JAPAC) by team HelloWorld.

## How it works

1. **PGD cloaking engine** (`service/`) — a FastAPI microservice running 60 steps of
   Projected Gradient Descent (eps 8/255) against a face-recognition surrogate
   (InceptionResnetV1 / VGGFace2). The perturbation is invisible (PSNR 32.7 dB)
   but moves the embedding **1.52** away from the original (identity threshold ~1.0–1.2),
   and survives JPEG recompression (Q75: 1.51).
2. **Web app** (`nuxt-app/`) — a Nuxt 3 frontend where users upload photos, tune
   strength, and download cloaked results. Proxies to the engine; persists job
   metadata to Firestore.
3. **AI report** — Gemini generates a plain-language protection report per photo.
4. **Physical sticker** (`prototype/` research) — a 6 cm universal adversarial patch
   for in-person protection (5/6 placements break matching).

## Validated numbers

| Metric | Value |
|---|---|
| Embedding distance (original → cloaked) | 1.5184 (threshold ~1.0–1.2) |
| After JPEG Q75 | 1.5149 |
| Cosine similarity | −0.1527 |
| Face detection confidence | 1.0 → 0.9997 (still detectable — by design) |
| PSNR | 32.7 dB |
| Latency | ~16–25 s/request on 2 vCPU |

## Honest limits

- Breaks **identity matching**, not face detection. A detector still sees a face;
  a matcher can't tell whose it is.
- Validated against one surrogate model and one test face — no claim against
  commercial black-box systems.

## Architecture on Google Cloud

- **Cloud Run** (`asia-southeast1`): `face-smash` engine (2 vCPU / 4 GiB) + `web/` (Nuxt)
- **Firestore**: `protection_jobs` collection (job metadata, fail-safe — the API never breaks if Firestore is down)
- **Artifact Registry**: engine container image
- **Gemini**: `/report` plain-language summaries
- **Firebase**: web hosting/auth surface via the Nuxt app

## Run it

```bash
# Engine (Docker)
docker build -f service/Dockerfile -t face-smash .
docker run -p 8080:8080 face-smash

# Web app
cd nuxt-app && npm install && npm run build
ENGINE_URL=http://localhost:8080 npm run preview
```

See `service/FIRESTORE_SETUP.md` for the Firestore IAM setup and `docs/api-contract.md`
for the API contract.
