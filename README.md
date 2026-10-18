# Face Smash

Anti-AI-facial-recognition for everyday photos. Upload a photo, get back a visually near-identical
image whose face embedding is pushed far away from your identity — so AI scrapers and
face-search engines can't match it back to you.

Built for the **Google Cloud AI Builders Cup 2026** (JAPAC) by team HelloWorld.
Cloaking method based on LowKey (Cherepanova et al., ICLR 2021) with ensemble/EOT/DSSIM-budget ideas; models via `facenet-pytorch` and insightface.

## How it works

1. **Cloaking engine v2** (`service/`, `prototype/cloak_v2.py`) — FastAPI microservice. Optimises a
   perturbation against an **ensemble of 3 face-recognition models** (VGGFace2, CASIA-WebFace, MobileFaceNet)
   with a margin loss, **EOT** (crop jitter, resampling, real JPEG) and a **DSSIM perceptual budget** so the
   noise hides where it is cheapest. Up to 2 faces per photo by default (`MAX_FACES`).
2. **Verification on the final photo** — the output is re-detected and re-embedded under PNG / JPEG-75 /
   downscale+JPEG-75, on all ensemble models **plus ArcFace-R50 as an independent hold-out**. The verdict
   (`broken` / `partial` / `weak`) uses the **worst case**.
3. **Web app** (`web/`) — Nuxt 4 frontend: upload, strength, before/after slider, per-model results.
   Proxies to the engine; job metadata (never the photo) goes to Firestore.
4. **Gemini report** — `/report` sends the *verified* numbers to Gemini (Gemini API key, or Vertex AI via the
   Cloud Run service account) and returns structured JSON. If Gemini is unavailable the API says so
   (`source: "template"`, `fallback_reason`) and the UI shows an "offline summary" badge — never a silent mock.
   `GET /report/selftest` makes a live call: run it before recording the demo.
5. **Physical sticker** (`prototype/` research) — a 6 cm universal adversarial patch (v1 research, not re-tested).

## Measured results (8 faces, final photo, worst case over upload transforms)

| | face SSIM | ArcFace-R50 (held-out) | GlintR100 (held-out) | all 5 models ≤ 0.30 |
|---|---|---|---|---|
| v1 (old) | 0.929 | 0.73 | 0.75 | 0/8 |
| v2 low | 0.987 | 0.37 | 0.40 | 0/8 |
| v2 standard | 0.978 | 0.23 | 0.26 | 6/8 |
| v2 max | 0.970 | 0.11 | 0.15 | 8/8 |

Mean cosine similarity to the original (lower = better, ≤ 0.30 = no match). Full method, per-face logs and
limits: [`prototype/RESULTS_v2.md`](prototype/RESULTS_v2.md), harness in `prototype/eval_v2/`.

## Honest limits

- Breaks **identity matching**, not face detection.
- Tested on **open-source** recognisers only (3 attacked + 2 held-out) and **8 test faces** — **no claim** against
  commercial systems (Meta, Google, Clearview, PimEyes).
- Stronger levels are visible (slight "retouched" look). Always compare with the slider.
- insightface pretrained weights are research / non-commercial licensed.

## Architecture on Google Cloud

- **Cloud Run** (`asia-southeast1`): `face-smash` engine (2 vCPU / 4 GiB, `--concurrency=1`, `--timeout=300`) + `web/` (Nuxt)
- **Firestore**: `protection_jobs` collection (job metadata, fail-safe — the API never breaks if Firestore is down)
- **Artifact Registry**: engine container image
- **Gemini** (Gemini API key or Vertex AI via service account): `/report` structured plain-language summaries
- **Firebase**: web hosting/auth surface via the Nuxt app

## Run it

```bash
# Engine (Docker)
docker build -f service/Dockerfile -t face-smash .
docker run -p 8080:8080 face-smash

# Web app
cd web && pnpm install && pnpm build
ENGINE_URL=http://localhost:8080 node .output/server/index.mjs
```

See `service/FIRESTORE_SETUP.md` for the Firestore IAM setup and `docs/api-contract.md`
for the API contract.
