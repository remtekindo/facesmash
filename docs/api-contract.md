# Cloak Engine — API Contract v0.1 (draft)

**Status:** draft for parallel team development, 2026-10-09.
Lets the Nuxt monolith (Rey + friend) be built **this weekend with mock responses** — no need to wait for the engine microservice. (Stack revised 2026-10-09: Nuxt monolith + cloaking engine, both on Cloud Run asia-southeast1. Flutter deferred — only if the main stack is green before ~16 Oct.)

**Engine status:** proven on CPU. LowKey-style PGD cloaking vs InceptionResnetV1 (VGGFace2):
embedding distance original→cloaked **1.518** (same-identity baseline 0.200; identity-matcher threshold ~1.0–1.2 → matcher breaks),
**survives JPEG Q75 recompression (1.515)**, PSNR 32.7 dB (visually subtle).
Honest limits: breaks *identity matching*, not face detection; transferability to unknown commercial models untested.
Full numbers: `prototype/RESULTS.md`. Code: `prototype/cloak.py`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/protect` | Cloak one photo, return protected image + metrics |
| GET | `/health` | Readiness probe |

## POST /protect

**Request** — multipart/form-data (recommended) or JSON with base64:

| Field | Type | Required | Notes |
|---|---|---|---|
| `image` | file / base64 string | yes | jpg, png, webp. Needs a detectable face (≥ ~160px face crop works) |
| `strength` | `low` \| `standard` \| `max` | no (default `standard`) | Maps to PGD steps 30 / 60 / 90. Higher = stronger shift, slower |
| `seed` | int | no | **Per-photo randomization** — a different seed produces a different perturbation pattern each time. This is the "randomizer": adversarial, not naive noise |
| `jpeg_simulate` | bool | no (default `true`) | Also return whether protection survives a JPEG Q75 round-trip (simulated social-media upload) |

**Response 200** (JSON) — see `mock-protect-response.json` for a byte-faithful example:

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
    "processing_time_ms": 22500
  }
}
```

**Errors:**

| Code | `error` | Meaning | Client action |
|---|---|---|---|
| 400 | `BAD_IMAGE` | unreadable / unsupported format | ask user for another photo |
| 413 | `TOO_LARGE` | > 10 MB | downscale client-side first |
| 422 | `NO_FACE` | MTCNN found no face | ask for a clearer front-facing photo |
| 504 | (timeout) | > 60 s | retry once with `strength=low` |

**Timing (measured on reference CPU, 2026-10-09):** model load 2.3 s (one-time, cached);
PGD 60 steps 15.7 s → warm request ≈ 16–25 s including encode/decode.
**Client timeout: 60 s.** Show honest progress copy ("Protecting photo… ~20 s"), never a fake instant result.

## GET /health

```json
{ "status": "ok", "models_loaded": true, "version": "0.1" }
```

## Client snippets (drop-in with mock URL)

**Flutter (dio):**
```dart
final form = FormData.fromMap({
  'image': await MultipartFile.fromFile(photoPath, filename: 'photo.jpg'),
  'seed': Random().nextInt(1 << 31), // different pattern every photo
  'strength': 'standard',
});
final res = await dio.post('$engineUrl/protect',
    data: form,
    options: Options(receiveTimeout: const Duration(seconds: 60)));
final b64 = res.data['protected_image'] as String;
```

**Nuxt server route (gateway):**
```ts
// server/api/protect.post.ts — proxies to the engine so the engine URL stays server-side
const form = new FormData()
form.append('image', file) // File/Blob from the multipart upload
form.append('seed', String(Math.floor(Math.random() * 2 ** 31))) // per-photo randomization
form.append('strength', 'standard')
const res = await $fetch(`${engineUrl}/protect`, {
  method: 'POST',
  body: form,
  timeout: 60000, // engine takes ~16–25 s on CPU; never fake an instant result
})
const metrics = res.metrics // detection_before/after, embedding_distance, ...
```

## Deployment path (friend's lane, after the contract is stable)

Thin FastAPI wrapper around `prototype/cloak.py` → Docker (`python:3.12-slim`, CPU torch from `prototype/requirements.txt`) → Cloud Run (2 vCPU / 4 GB, request timeout 300 s). Model weights download on cold start, then stay cached in memory.

## Attribution (README + pitch deck — required)

- Cloaking method based on **LowKey** (Cherepanova et al., *"Leveraging Adversarial Attacks to Protect Social Media Users from Facial Recognition"*, ICLR 2021).
- Surrogate models via `facenet-pytorch` (InceptionResnetV1 pretrained on VGGFace2; MTCNN).
- **Team's original work** (what the jury judges): the mobile app, the JPEG-robust pipeline tuned for social media, per-photo perturbation randomization, the before/after evaluation dashboard, and the physical sticker pattern.

## POST /report — Gemini evaluation report (v0.2, implemented 2026-10-13)

**Purpose (option B, decided):** turn `/protect` metrics into a plain-language before/after evaluation report for the user (Indonesian default, English optional). This is the product's genuine Gemini usage — it also strengthens the "built with Google AI" story for the jury.

**Endpoint (engine service, implemented 2026-10-13):** `POST /report` on the FastAPI service — implemented server-side so the Nuxt app just POSTs here; **no separate Nuxt server route / Gemini integration needed on the Nuxt side** (this supersedes the earlier "Nuxt server route" plan). API key **never** leaves the server (env var).

**Env vars:** `GEMINI_API_KEY` (preferred) or `GOOGLE_GEMINI_API_KEY` — grab a free key from AI Studio; `GEMINI_MODEL` — any current `gemini-*` text model (default `gemini-2.5-flash`). **No key → the endpoint serves the mock below**, so the UI contract is identical and the demo works either way. If the key is set but the Gemini call fails, it falls back to the mock with `"note": "gemini_unavailable_fallback"`.

**Request** (JSON):
```json
{
  "metrics": { "detection_before": 1.0, "detection_after": 0.9997, "embedding_distance": 1.5184, "cosine_similarity": -0.1527, "jpeg_q75_survived": true, "psnr_db": 32.7 },
  "meta": { "strength": "standard", "processing_time_ms": 22500 },
  "lang": "id"
}
```

**Prompt template** (system):
> Kamu adalah asisten privasi dalam aplikasi pelindung foto. Jelaskan hasil proteksi di bawah ini dengan bahasa Indonesia yang sederhana dan menenangkan, untuk pengguna awam. Struktur: (1) satu kalimat kesimpulan — apakah foto ini sekarang aman dari pengenalan wajah AI; (2) jelaskan tiap angka dalam 1 kalimat bahasa sehari-hari; (3) satu paragraf jujur tentang batasan — proteksi ini teruji pada model tertentu, bukan jaminan mutlak terhadap semua AI; (4) satu tips praktis. Maksimal 150 kata. Jangan melebih-lebihkan klaim.

**Prompt template** (user) — fill placeholders from the request:
> Hasil proteksi foto: jarak embedding {embedding_distance} (batas identitas ~1.0 — semakin jauh semakin tidak dikenali), kemiripan kosinus {cosine_similarity}, deteksi wajah sebelum {detection_before} / sesudah {detection_after}, tahan kompresi media sosial: {jpeg_q75_survived}, kualitas visual {psnr_db} dB (di atas 30 = nyaris tak terlihat). Kekuatan: {strength}.

**Response 200** (JSON):
```json
{ "report": "<teks laporan>", "lang": "id", "model": "gemini-2.x" }
```

**Mock for parallel dev** (until the key is wired):
```json
{ "report": "Foto kamu sekarang jauh lebih aman. Sistem pengenal wajah AI tidak lagi bisa mengaitkan foto ini dengan identitasmu (skor kemiripan turun jauh di bawah batas). Proteksinya tetap bertahan walau foto dikompres saat diupload ke media sosial, dan secara kasat mata fotomu tidak berubah. Batasan jujur: pengujian dilakukan terhadap model tertentu — ini perlindungan kuat, bukan jaminan mutlak terhadap semua AI di masa depan. Tips: gunakan kekuatan 'max' untuk foto yang sangat sensitif.", "lang": "id", "model": "mock" }
```

**Setup note for the team:** enable Vertex AI API (or grab a key from AI Studio) on the competition's Google Cloud project; store as server env var. Model choice: any current `gemini-*` text model. If the key isn't ready, ship with the mock above — the UI contract is identical.

---

## v0.3 changes (engine v2, 2026-10-18) — additive, v0.1/v0.2 clients keep working

**`POST /protect`** (same request; new optional field `max_faces`, default 2 (env `MAX_FACES`; time scales linearly per face); `jpeg_simulate` is ignored —
verification always runs). Response keeps `protected_image`, `metrics`, `meta` and adds `verification`.

`metrics` semantics changed (field names kept for compatibility) — **measured on the final photo, worst case over
all checked models and upload transforms (PNG, JPEG-75, 50% downscale + JPEG-75)**:

| field | meaning now |
|---|---|
| `embedding_distance` | L2 distance = sqrt(2 − 2·worst_cos). Match threshold ≈ **1.18** (cos 0.30), was "~1.0" |
| `cosine_similarity` | worst-case cosine to the original (≤ 0.30 = no match) |
| `jpeg_q75_survived` / `jpeg_q75_distance` | worst case over the two lossy transforms (JPEG-75, downscale+JPEG-75) |
| `psnr_db` | **face region** PSNR (was whole image); `psnr_full_db` = whole image |
| `ssim_face`, `visibility` | new: face SSIM and `near-invisible` / `subtle` / `visible on close look` / `visible` |

`verification` (new):
```json
{ "verdict": "broken | partial | weak", "identity_match_broken": true, "match_threshold_cos": 0.3,
  "models": { "vggface2": {"worst_cos": -0.06, "role": "optimised against"},
              "arcface":  {"worst_cos": 0.16,  "role": "independent check (not optimised against)"} },
  "transforms": ["png", "jpeg75", "down50_jpeg75"] }
```
`meta` adds `faces_protected`, `independent_check`; `surrogate_model` now lists the attacked ensemble.

**`POST /report`** — request may include `verification`; `lang` default stays `id` (the web app sends `en`).
Response: `{ report, sections:{headline,summary,numbers[],limits,tips[]}, lang, source, model, backend }`.
`source` is `"gemini"` or `"template"`; on `template` there is also `fallback_reason`
(`no_credentials`, `http_403`, `bad_output:…`, …). The old silent `mock` is gone.
Backends: Gemini API key (`GEMINI_API_KEY`) or Vertex AI through the Cloud Run service account
(`VERTEX_LOCATION`, default `us-central1`; grant the service account `roles/aiplatform.user`).
`GEMINI_MODEL` overrides the model; fallbacks: `gemini-2.5-flash`, `gemini-flash-latest`.

**`GET /report/selftest`** — live Gemini round-trip; 200 `{"ok":true,"source":"gemini",…}` or 503 with the reason.
`GET /health` now lists `ensemble`, `holdout` and `gemini` status.
