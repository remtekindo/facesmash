# HANDOVER — Face Smash engine v2 (save point FS2-20261018-A4)

Written 2026-10-18 for the next AI agent / human. Repo `remtekindo/facesmash`, branch **`claude`** (3 feature commits
above `main`; **`main` is untouched and is the safe baseline**). Read this file first, then `prototype/RESULTS_v2.md`.

## 0. Context in 6 lines
- Owner: "Re" (Remtekindo, Bali), solo. Competition: **Google Cloud AI Builders Cup 2026 (APAC)**, category *Media, Content &
  Digital Experiences*. Rules: working prototype on Cloud Run/GCP/Firebase, proposal deck PDF, 3-min public demo video,
  all in English, **"meaningful use of Gen AI" is a judged criterion**, no students on the team.
- Product: Face Smash = adversarial "cloak" so AI face recognition can't match a posted photo to the person.
- Stack: Nuxt 4 web (`web/`, Cloud Run) → FastAPI engine (`service/`, Cloud Run 2 vCPU/4 GiB) → Firestore job log, Gemini report.
- There is a **separate, already-live "GPT version"** (built earlier with another AI; not in this repo's `claude` branch).
  Another assistant (Muse) reviewed v2, reported **Cloud Build failed**, and recommended submitting the live version.
  The owner had not decided at the time of writing; ~5 h to deadline was mentioned.
- My (Claude's) position, told to the owner: lock the live version for safety, but **do not repeat v1's claims**
  (see §3); deploy v2 as a *separate* Cloud Run service and switch `ENGINE_URL` only if it passes §5 checks.
- Owner preferences: reply in **Indonesian**, direct and critical (no flattery), short, linear non-technical steps, folder
  trees over command lists, give every file a unique ID so versions can be compared, never fabricate results,
  never put secrets in code/logs, don't bring up the "LLM for Public Administration" idea unprompted.

## 1. What exists on branch `claude`
```
facesmash/
├─ HANDOVER.md                      this file
├─ README.md                        results table, honest limits, run commands
├─ docs/api-contract.md             v0.1/v0.2 + new "v0.3 changes" section (additive)
├─ prototype/
│  ├─ cloak.py                      LEGACY v1 engine (unchanged)
│  ├─ cloak_v2.py                   NEW engine (ensemble + DSSIM budget + EOT + verification)
│  ├─ RESULTS_v2.md                 measured results, dead ends, limits
│  └─ eval_v2/                      evallib.py, exp.py (research harness), results/*.log, baseline.json
├─ service/
│  ├─ app.py                        v0.2: /protect, /report (Gemini), /report/selftest, /health
│  ├─ fetch_models.py               downloads ArcFace ONNX weights (replaces a Dockerfile heredoc)
│  ├─ Dockerfile, requirements.txt  updated for v2
│  └─ firestore_store.py            unchanged
└─ web/app/app.vue, web/server/api/report.post.ts   new result UI, compare-slider fix, Gemini badge
```
Commits: `0c7dc50` feat(engine) [FS2-20261018-A1] · `808d04d` fix(docker) [A2] · `5b892f0` fix(service) MAX_FACES=2 [A3] · this save point [A4].

## 2. How v2 works (so you can defend it in Q&A)
1. MTCNN finds faces + landmarks (default max **2** faces, env `MAX_FACES`).
2. Per face: Adam optimisation of a perturbation against an **ensemble** (VGGFace2, CASIA-WebFace via facenet-pytorch;
   MobileFaceNet via ONNX→torch `onnx2torch`). Loss = hinge on cosine similarity to the original embedding
   (`target_cos`) + `ssim_w` × DSSIM (perceptual budget) inside a soft face ellipse; luma-dominant (chroma ×0.12).
3. **EOT** each step: crop/landmark jitter, down-up resampling, real JPEG with straight-through gradient.
4. Strength table in `cloak_v2.STRENGTH` (eps/255, steps, target_cos, ssim_w): low (20,70,0.10,35),
   standard (24,100,−0.05,10), max (28,150,−0.20,5).
5. `verify()` re-detects the face on the FINAL photo, re-embeds under PNG / JPEG-75 / downscale-50%+JPEG-75 and reports the
   **worst case**, also on **ArcFace-R50 as an independent hold-out** (never attacked). Verdict: cos ≤ 0.30 broken, ≤ 0.50
   partial, else weak. `/protect` returns it in `verification`; UI renders it.

## 3. Key findings (measured; do not contradict without re-measuring)
- v1 reported the L2 distance of the 160×160 crop it had just optimised, on the model it attacked (≈1.9). On the real output
  photo (8 faces) v1's worst-case cosine was VGGFace2 0.50, CASIA 0.58, MobileFaceNet 0.67, ArcFace-R50 0.73, GlintR100 0.75
  (≥0.30 = still matched) with face SSIM 0.929 and visible blotches. So v1's "invisible" and "breaks matching" claims do not hold
  up under honest measurement; PSNR 33–35 dB did not mean invisible.
- v2 (mean of 8 faces): face SSIM low 0.987 / standard 0.978 / max 0.970; held-out ArcFace-R50 cos 0.37 / 0.23 / 0.11,
  GlintR100 0.40 / 0.26 / 0.15; faces with all 5 models ≤0.30: **0/8, 6/8, 8/8**.
- Biggest win: DSSIM penalty instead of saturating ±eps. Chroma freedom turned irises green → chroma ×0.12.
- Dead ends: high-pass filtering the perturbation (protection collapsed), texture-weighted caps (visible contour patterns),
  plain larger L∞ budgets (visible).
- Timing (2 CPU threads, sandbox): one face low 22 s / standard 32 s / max 49 s; service end-to-end standard 39 s (659×439),
  59 s (1665×2048); synthetic 3-face composite at max 186 s, peak RSS 1.7 GB → hence `MAX_FACES=2`.

## 4. Verified vs NOT verified
Verified (actually run): the full evaluation (8 faces × 3 strengths + v1); `service/app.py` via TestClient and real uvicorn
(small, large and 3-face photos; NO_FACE and BAD_STRENGTH errors); `/report` template fallback and the Gemini HTTP/parse path
against a **local fake server**; `pnpm build` of `web/` and a Chromium screenshot of the new result card against the live local
service; `pip install --dry-run` of the requirements (no conflicts); `fetch_models.py` downloads both ONNX files.

NOT verified — treat as open risks:
1. **Docker build / Cloud Build.** No Docker daemon in the sandbox; the owner's Cloud Build run reportedly failed, cause unknown
   (no log seen). Suspects: heredoc needing BuildKit (fixed in A2, untested), default 10-min Cloud Build timeout (use
   `--timeout=1800`), `onnx`/`onnx2torch` resolution. **First action: get the last ~30 lines of the failing step.**
2. **A real Gemini call.** No key/credentials here. Backends: `GEMINI_API_KEY`, or Vertex AI via the Cloud Run service account
   (needs `roles/aiplatform.user`, `VERTEX_LOCATION` default `us-central1`). Models tried in order: `GEMINI_MODEL`,
   `gemini-2.5-flash`, `gemini-flash-latest` (a docs page listed 2.5-flash as stable and 3.8-flash as recommended — re-check).
3. Cloud Run speed/cold start; Firestore writes with the new payload; any commercial recogniser; non-frontal / dark / child /
   masked faces (test set was 8 mostly frontal adult faces).

## 5. Next steps, in order
1. Build: `gcloud builds submit --timeout=1800 --tag <artifact-registry-image> -f service/Dockerfile .` from the repo root (build
   context must be the repo root). Fix whatever the log shows.
2. Deploy to a NEW Cloud Run service (e.g. `face-smash-v2`): `--cpu=2 --memory=4Gi --concurrency=1 --timeout=300`; set
   `GEMINI_API_KEY` as a secret or grant Vertex access.
3. Gate checks: `GET /health` lists 4 models + `holdout:["arcface"]`; `GET /report/selftest` returns `source:"gemini"`;
   one real photo at `standard` returns verdict `broken` in well under 300 s.
4. Only then point the web app's `ENGINE_URL` at v2. Otherwise keep the live version and use v2 as "future work".
5. Pitch deck / video: use numbers from `RESULTS_v2.md` only; **never reuse v1's 1.52 / 32.7 dB / "invisible" claims**; cite LowKey
   (Cherepanova et al., ICLR 2021), facenet-pytorch, insightface; state limits (open-source models, 8 faces, no commercial claim)
   and the insightface research-only licence. Consider defaulting the UI to `max` (8/8 broken, ~60 s).
6. Optional: more/diverse test faces; opt-in Gemini multimodal "privacy audit" (conflicts with the privacy promise unless the
   user explicitly consents to sending the photo).

## 6. Rebuild the sandbox (if needed)
```
python3 -m venv venv && . venv/bin/activate
pip install torch torchvision                      # CPU works; the +cpu index may be blocked in some sandboxes
pip install --no-deps facenet-pytorch==2.6.0
pip install pillow numpy requests tqdm scikit-image onnxruntime onnx onnx2torch fastapi uvicorn python-multipart httpx
python service/fetch_models.py models              # w600k_mbf.onnx + w600k_r50.onnx
# held-out eval models: insightface release v0.7 antelopev2.zip (glintr100.onnx) -> models/heldout/antelopev2/
# run the service:  cd service && FIRESTORE_ENABLED=0 FACESMASH_MBF_ONNX=../models/w600k_mbf.onnx \
#                   FACESMASH_ARCFACE_ONNX=../models/w600k_r50.onnx uvicorn app:app --port 8080
# eval: cd prototype/eval_v2 && FACESMASH_MODELS=models FACESMASH_FACES=faces python exp.py --name run --strength standard --faces a,b
```
Test photos are **not** committed (they include the owner's screenshots). Public alternatives used: `skimage.data.astronaut()`
and the facenet-pytorch repo's `data/test_images/*/1.jpg`.

## 7. Conventions
Commit format `type(scope): description [batchID]` with batch IDs `FS2-YYYYMMDD-An` and the attribution trailers from the
session. Push only to `claude` unless the owner says otherwise. Never claim "passes/tested/no regression" without evidence.
