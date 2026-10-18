# Handover — Face Smash engine v2 (batch FS2-20261018-A1)

Save point written 2026-10-18 by Claude (session handed over to another account). Branch: `claude`.

## What this batch did
Problem reported: prototype output "still lacking" (visible blotches/patterns in the cloaked photo, UI metrics
looked great). Root causes found and fixed:

1. **Metrics were self-flattering.** v1 reported the distance of the 160×160 crop it had just optimised, on the model
   it attacked. On the real output photo v1 only reached cos 0.50–0.75 (still matched). → v2 verifies on the final
   photo, worst case over upload transforms, with an independent hold-out model (ArcFace-R50).
2. **Visible noise.** v1 saturated ±8/255 everywhere at 160 px, then upsampled. → v2 uses a DSSIM perceptual penalty,
   luma-dominant noise, soft face ellipse; face SSIM 0.93 → 0.97–0.99.
3. **Weak transfer.** → ensemble (3 models) + EOT.
4. **Gemini rules risk.** `/report` silently fell back to a mock. → structured Gemini call (API key or Vertex AI),
   explicit `source`/`fallback_reason`, UI badge, `/report/selftest`.
5. **UI bugs.** Compare slider showed black bars for portrait photos; hero stats were hard-coded v1 numbers.

Results table and limits: `prototype/RESULTS_v2.md`. Harness: `prototype/eval_v2/` (logs in `results/`).

## Changed / added files
```
facesmash/
├─ HANDOVER.md                      (new)
├─ README.md                        (results table, honest limits, run commands)
├─ docs/api-contract.md             (+ v0.3 section)
├─ prototype/
│  ├─ cloak_v2.py                   (NEW engine; cloak.py = legacy v1, untouched)
│  ├─ RESULTS_v2.md                 (new)
│  └─ eval_v2/                      (new: evallib.py, exp.py, results/*.log)
├─ service/
│  ├─ app.py                        (rewritten: v2 engine, honest verification, Gemini report/selftest)
│  ├─ Dockerfile                    (ONNX weights download, cloak_v2.py, concurrency note)
│  ├─ requirements.txt              (+ onnx, onnx2torch, google-auth)
│  └─ README.md                     (v0.2 note)
└─ web/
   ├─ app/app.vue                   (new metrics, per-model table, compare fix, Gemini badge)
   └─ server/api/report.post.ts     (forwards verification)
```

## Verified vs NOT verified
Verified (run in the sandbox, CPU, 2 threads):
- Engine v2 on 8 faces × 3 strengths + v1 baseline (`prototype/eval_v2/results/`).
- `service/app.py` end-to-end via FastAPI TestClient: `/protect` (659×439 and 1665×2048 photos), NO_FACE and
  BAD_STRENGTH errors, `/report` template fallback and the Gemini HTTP/parse path **against a local fake server**.
- `web/` builds (`pnpm build`) and renders the new result card in Chromium against the live local service.

NOT verified:
- **A real Gemini call** (no API key / GCP credentials here). Before the demo: deploy, then `GET /report/selftest`
  must return `source: "gemini"`. Vertex needs `roles/aiplatform.user` on the Cloud Run service account.
  Default model is `gemini-2.5-flash` (fallback `gemini-flash-latest`); check the current model list.
- The **Docker build** (ONNX download step, image size, cold start) and Cloud Run latency (expect ~30–45 s standard
  on 2 vCPU for one face; 3 faces ≈ 3×).
- Multi-face photos (code path exists: up to 3 faces; not tested on a real group photo).
- Firestore writes with the new `metrics`/`verification` payload (schema is a superset of v1).
- Any commercial recogniser.

## Update — Cloud Build failure reported (cause NOT confirmed)
The user's Cloud Build run of v2 failed; no log was available to me and no Docker daemon exists in my sandbox, so
the cause is unconfirmed. Suspects, in order: (1) the Dockerfile heredoc (`RUN python - <<'PY'`) needs BuildKit —
now replaced by `service/fetch_models.py`; (2) Cloud Build's default 10-minute timeout (use `--timeout=1800`);
(3) pip resolution of `onnx`/`onnx2torch`. `fetch_models.py` was run here and downloads both ONNX files correctly.
To diagnose: send the failing step's last ~30 log lines. The live v1 service is untouched (branch `main` unchanged).

## Next steps (suggested order)
1. Deploy engine with `--concurrency=1 --cpu=2 --memory=4Gi --timeout=300`; run `/health` and `/report/selftest`.
2. Decide the default strength: `standard` = 6/8 faces fully broken; `max` = 8/8 with slightly more visible change.
   Consider defaulting the UI to `max` (~60 s).
3. Update the pitch deck/demo numbers from `RESULTS_v2.md` (do **not** reuse v1's 1.52 / 32.7 dB claims) and cite
   LowKey, facenet-pytorch, insightface in the deck (attribution + licence note).
4. Record the 3-minute demo with a photo, the slider, the per-model table and the Gemini report badge showing "Gemini".
5. Optional: add a held-out check on more faces/ethnicities/lighting; a Gemini multimodal "privacy audit" (opt-in,
   because sending the photo to Gemini conflicts with the privacy promise unless the user consents).

## Commit message
`feat(engine): cloak v2 ensemble+DSSIM, honest final-photo verification, Gemini report [FS2-20261018-A1]`
