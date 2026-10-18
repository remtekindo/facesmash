"""
Face Cloak microservice — FastAPI wrapper around prototype/cloak_v2.py.

v0.2 (2026-10-18): engine v2 (ensemble + DSSIM perceptual budget + EOT) and
metrics measured on the FINAL output image (re-detected, re-cropped, re-embedded
under PNG / JPEG-75 / downscale+JPEG-75) — not on the 160x160 crop that was
optimised, which is what v0.1 reported.

Endpoints: POST /protect, POST /report (Gemini), GET /report/selftest, GET /health.
"""
import base64
import io
import json
import logging
import math
import os
import random
import sys
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
import requests
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

# Firestore job-history persistence (rules-mandated Firebase/Firestore usage).
# The module is fail-safe by design: it never raises, so persistence can
# never break /protect (see firestore_store.py).
from firestore_store import save_protection_job

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prototype"))
import cloak_v2 as C  # noqa: E402

log = logging.getLogger("face-cloak")

VERSION = "0.2"
ENGINE_NAME = f"cloak-v2 {VERSION} (ensemble PGD + DSSIM budget + EOT)"
MAX_BYTES = 10 * 1024 * 1024          # 10 MB
MAX_DIM = 2048                        # downscale if larger
MAX_FACES = int(os.environ.get("MAX_FACES", "2"))
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
# cos(original, protected) at/below which a recogniser no longer matches the identity.
MATCH_COS = C.MATCH_COS_THRESHOLD
PARTIAL_COS = 0.50

_models: dict = {}
_gpu_lock = threading.Lock()   # one heavy job at a time (CPU + RAM bound)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup, not per request.
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", os.cpu_count() or 2)))
    from facenet_pytorch import MTCNN
    _models["mtcnn"] = MTCNN(image_size=160, margin=0, keep_all=True, device="cpu")
    # vggface2 + casia + mbf are optimised against; arcface (R50) is HOLD-OUT:
    # only used to verify transfer, never attacked. Missing ONNX files are skipped.
    _models["ens"] = C.Ensemble(("vggface2", "casia", "mbf", "arcface"), holdout=("arcface",))
    log.info("ensemble: %s (holdout: %s)", _models["ens"].names, sorted(_models["ens"].holdout))
    yield
    _models.clear()


app = FastAPI(title="Face Cloak Service", version=VERSION, lifespan=lifespan)

# CORS: the Nuxt web app normally calls this service server-to-server; restrict via
# CORS_ALLOW_ORIGINS (comma-separated) if browsers call it directly.
_cors_origins = [
    o.strip()
    for o in os.environ.get("CORS_ALLOW_ORIGINS", "*").split(",")
    if o.strip()
] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "face-cloak",
        "version": VERSION,
        "health": "/health",
        "docs": "/docs",
        "endpoints": ["POST /protect", "POST /report", "GET /report/selftest"],
    }


@app.exception_handler(HTTPException)
async def _http_exc_handler(request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content=detail)


@app.get("/health")
def health():
    ens = _models.get("ens")
    return {
        "status": "ok",
        "models_loaded": "mtcnn" in _models,
        "version": VERSION,
        "ensemble": ens.names if ens else [],
        "holdout": sorted(ens.holdout & set(ens.names)) if ens else [],
        "gemini": _gemini_status(),
    }


# --------------------------------------------------------------------------- #
# Visual-quality metrics on the REAL output (face regions only)
# --------------------------------------------------------------------------- #
def _visual(orig: Image.Image, out: Image.Image, jobs) -> dict:
    a, b = C.pil_to_t(orig), C.pil_to_t(out)
    psnrs, ssims = [], []
    for j in jobs:
        x1, y1, x2, y2 = [int(v) for v in j.box]
        x1, y1 = max(0, x1), max(0, y1)
        fa, fb = a[:, y1:y2, x1:x2], b[:, y1:y2, x1:x2]
        if fa.shape[-1] < 8 or fa.shape[-2] < 8:
            continue
        mse = float(((fa - fb) ** 2).mean())
        psnrs.append(99.0 if mse == 0 else 10 * math.log10(1.0 / mse))
        ssims.append(float(C._ssim_map(fa, fb).mean()))
    mse_full = float(((a - b) ** 2).mean())
    return {
        "psnr_face": float(np.mean(psnrs)) if psnrs else 99.0,
        "ssim_face": float(np.mean(ssims)) if ssims else 1.0,
        "psnr_full": 99.0 if mse_full == 0 else 10 * math.log10(1.0 / mse_full),
    }


def _visibility_label(ssim: float) -> str:
    if ssim >= 0.975:
        return "near-invisible"
    if ssim >= 0.95:
        return "subtle"
    if ssim >= 0.92:
        return "visible on close look"
    return "visible"


def _l2_from_cos(c: float) -> float:
    return math.sqrt(max(0.0, 2.0 - 2.0 * c))


def _protect_sync(raw: bytes, content_type: str, strength: str, seed, max_faces: int) -> dict:
    t_start = time.time()
    if strength not in C.STRENGTH:
        raise HTTPException(400, {"error": "BAD_STRENGTH",
                                  "message": f"strength must be one of {sorted(C.STRENGTH)}"})
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, {"error": "TOO_LARGE",
                                  "message": f"image exceeds {MAX_BYTES // 1024 // 1024} MB; downscale client-side first"})
    if (content_type or "") not in ALLOWED_TYPES:
        raise HTTPException(400, {"error": "BAD_IMAGE",
                                  "message": f"unsupported content type: {content_type}"})
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()  # force full decode to catch truncated files
        img = img.convert("RGB")   # also drops EXIF/GPS metadata: the output carries none
    except Exception:
        raise HTTPException(400, {"error": "BAD_IMAGE", "message": "unreadable image file"})

    downscaled = False
    if max(img.size) > MAX_DIM:
        img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
        downscaled = True

    mtcnn, ens = _models["mtcnn"], _models["ens"]
    all_jobs = C.detect_faces(mtcnn, img, max_faces=max(1, min(max_faces, MAX_FACES)), min_prob=0.9)
    if not all_jobs:
        # fall back to the single best face at a lower confidence bar
        all_jobs = C.detect_faces(mtcnn, img, max_faces=1, min_prob=0.0)
    if not all_jobs:
        raise HTTPException(422, {"error": "NO_FACE",
                                  "message": "no face detected; use a clearer front-facing photo"})
    jobs = all_jobs
    prob_before = max(j.prob for j in jobs)
    run_seed = seed if seed is not None else random.randint(0, 2**31 - 1)

    with _gpu_lock:
        refs = C.reference_embeddings(img, mtcnn, ens, jobs)
        out, _results = C.cloak_image(img, jobs, ens, C.cfg_for(strength, seed=run_seed))
        ver = C.verify(img, out, mtcnn, ens, jobs, refs)

    boxes, probs = mtcnn.detect(out)
    prob_after = float(np.max(probs)) if boxes is not None and len(boxes) else 0.0
    vis = _visual(img, out, jobs)

    worst_all = ver["worst_cos_all"]
    # "survives upload compression": worst case over the two lossy transforms only
    lossy = [max((v for v in pf.values() if v is not None), default=1.0)
             for t in ("jpeg75", "down50_jpeg75") for pf in ver["per_transform"][t]]
    worst_lossy = max(lossy) if lossy else 1.0
    verdict = "broken" if worst_all <= MATCH_COS else "partial" if worst_all <= PARTIAL_COS else "weak"

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    elapsed_ms = int((time.time() - t_start) * 1000)

    metrics = {
        "detection_before": round(prob_before, 4),
        "detection_after": round(prob_after, 4),
        # worst case over every checked model AND every upload transform, on the final photo
        "embedding_distance": round(_l2_from_cos(worst_all), 4),
        "cosine_similarity": round(worst_all, 4),
        "jpeg_q75_survived": bool(worst_lossy <= MATCH_COS),
        "jpeg_q75_distance": round(_l2_from_cos(worst_lossy), 4),
        "psnr_db": round(vis["psnr_face"], 1),          # face region (was: whole image)
        "psnr_full_db": round(vis["psnr_full"], 1),
        "ssim_face": round(vis["ssim_face"], 4),
        "visibility": _visibility_label(vis["ssim_face"]),
    }
    verification = {
        "verdict": verdict,                                  # broken | partial | weak
        "identity_match_broken": bool(worst_all <= MATCH_COS),
        "match_threshold_cos": MATCH_COS,
        "models": {n: {"worst_cos": round(c, 4),
                       "role": "independent check (not optimised against)" if n in ens.holdout else "optimised against"}
                   for n, c in ver["worst_cos"].items()},
        "transforms": list(ver["per_transform"].keys()),
    }
    meta = {
        "engine": ENGINE_NAME,
        "surrogate_model": " + ".join(n for n in ens.names if n not in ens.holdout),
        "independent_check": sorted(ens.holdout & set(ens.names)),
        "strength": strength,
        "seed": run_seed,
        "processing_time_ms": elapsed_ms,
        "downscaled": downscaled,
        "faces_protected": len(jobs),
    }

    # Firestore job history: params, metrics and verdict only — never the photo.
    firestore_job_id = save_protection_job({
        "strength": strength, "seed": run_seed, "downscaled": downscaled,
        "processing_time_ms": elapsed_ms, "engine": ENGINE_NAME,
        "faces_protected": len(jobs), "metrics": metrics, "verification": verification,
        "version": VERSION,
    })
    if firestore_job_id:
        meta["firestore_job_id"] = firestore_job_id

    return {"protected_image": b64, "metrics": metrics, "verification": verification, "meta": meta}


@app.post("/protect")
def protect(
    image: UploadFile = File(...),
    strength: str = Form("standard"),
    seed: int | None = Form(None),
    jpeg_simulate: bool = Form(True),     # kept for contract compatibility; verification always runs
    max_faces: int = Form(MAX_FACES),
):
    # sync endpoint -> FastAPI runs it in a worker thread, /health stays responsive
    return _protect_sync(image.file.read(), image.content_type, strength, seed, max_faces)


# --------------------------------------------------------------------------- #
# POST /report — Gemini plain-language report.
# Backends: (1) Gemini Developer API key, (2) Vertex AI via the Cloud Run service
# account (ADC). The response says exactly what happened: source = "gemini" or
# "template" (deterministic, metrics-based, clearly labelled) — never a silent mock.
# --------------------------------------------------------------------------- #
class ReportMetrics(BaseModel):
    detection_before: float = 1.0
    detection_after: float = 1.0
    embedding_distance: float = Field(..., ge=0)
    cosine_similarity: float = 0.0
    jpeg_q75_survived: bool = False
    psnr_db: float = 0.0
    ssim_face: float | None = None
    visibility: str | None = None


class ReportMeta(BaseModel):
    strength: str = "standard"
    processing_time_ms: int = 0
    faces_protected: int = 1


class ReportVerification(BaseModel):
    verdict: str | None = None
    models: dict = {}


class ReportRequest(BaseModel):
    metrics: ReportMetrics
    meta: ReportMeta = ReportMeta()
    verification: ReportVerification = ReportVerification()
    lang: str = "id"  # "id" (default) or "en"


GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
VERTEX_URL = ("https://{host}/v1/projects/{project}/locations/{loc}/publishers/google/models/{model}:generateContent")
GEMINI_MODELS = [m for m in [os.environ.get("GEMINI_MODEL"), "gemini-2.5-flash", "gemini-flash-latest"] if m]

_SYSTEM = {
    "en": (
        "You are the privacy assistant inside a photo-protection app (Face Smash). Explain the VERIFIED "
        "results you are given to a non-technical user, in plain English. Rules: (1) never contradict the "
        "given verdict (broken = identity match broken on every checked model; partial = some models may "
        "still match; weak = likely still recognisable) and never invent numbers; (2) the checks were run on "
        "a few open-source face-recognition models in simulation — say plainly this is not a guarantee against "
        "commercial systems or future AI, and never claim it defeats any named company; (3) the protection "
        "stops identity matching, not face detection; (4) if visibility is not 'near-invisible', say the photo "
        "may look slightly retouched. Keep summary under 90 words, 3 short bullets for the numbers, "
        "one honest-limits sentence, and 2 practical tips."),
    "id": (
        "Kamu asisten privasi di aplikasi pelindung foto (Face Smash). Jelaskan hasil TERVERIFIKASI di bawah "
        "kepada pengguna awam dalam bahasa Indonesia sederhana. Aturan: (1) jangan bertentangan dengan verdict "
        "(broken = pencocokan identitas patah di semua model yang diuji; partial = sebagian model mungkin masih "
        "mengenali; weak = kemungkinan masih dikenali) dan jangan mengarang angka; (2) pengujian dilakukan pada "
        "beberapa model open-source dalam simulasi — katakan jelas ini bukan jaminan terhadap sistem komersial "
        "atau AI masa depan, dan jangan klaim mengalahkan perusahaan tertentu; (3) proteksi menghentikan "
        "pencocokan identitas, bukan deteksi wajah; (4) jika visibility bukan 'near-invisible', sebutkan foto "
        "mungkin tampak sedikit diretus. Ringkasan di bawah 90 kata, 3 poin singkat untuk angka, satu kalimat "
        "batasan jujur, dan 2 tips praktis."),
}

_REPORT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "headline": {"type": "STRING"},
        "summary": {"type": "STRING"},
        "numbers": {"type": "ARRAY", "items": {"type": "STRING"}},
        "limits": {"type": "STRING"},
        "tips": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["headline", "summary", "numbers", "limits", "tips"],
}


def _facts(body: ReportRequest) -> dict:
    m = body.metrics
    verdict = body.verification.verdict or (
        "broken" if m.cosine_similarity <= MATCH_COS else "partial" if m.cosine_similarity <= PARTIAL_COS else "weak")
    return {
        "verdict": verdict,
        "worst_case_cosine_similarity_on_final_photo": m.cosine_similarity,
        "match_threshold_cosine": MATCH_COS,
        "embedding_distance_l2": m.embedding_distance,
        "survives_jpeg_and_resize": m.jpeg_q75_survived,
        "face_detection_before_after": [m.detection_before, m.detection_after],
        "visual_quality_psnr_face_db": m.psnr_db,
        "ssim_face": m.ssim_face,
        "visibility": m.visibility,
        "strength": body.meta.strength,
        "faces_protected": body.meta.faces_protected,
        "per_model_worst_cosine": {k: v.get("worst_cos") for k, v in body.verification.models.items()},
    }


def _gemini_status() -> dict:
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_GEMINI_API_KEY"):
        return {"configured": True, "backend": "api_key", "models": GEMINI_MODELS}
    if os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT") or os.environ.get("K_SERVICE"):
        return {"configured": True, "backend": "vertex_adc", "models": GEMINI_MODELS}
    return {"configured": False, "backend": None, "models": GEMINI_MODELS}


def _vertex_token_and_project():
    import google.auth
    import google.auth.transport.requests
    creds, project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(google.auth.transport.requests.Request())
    return creds.token, (os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT") or project)


def _gemini_generate(system: str, user_text: str, schema: dict | None = None) -> tuple[str, str, str]:
    """Returns (text, model, backend). Raises RuntimeError(code) if every attempt fails."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_GEMINI_API_KEY")
    gen_cfg = {"temperature": 0.4, "maxOutputTokens": 900}
    if schema:
        gen_cfg.update({"responseMimeType": "application/json", "responseSchema": schema})
    payload = {"systemInstruction": {"parts": [{"text": system}]},
               "contents": [{"role": "user", "parts": [{"text": user_text}]}],
               "generationConfig": gen_cfg}
    if not _gemini_status()["configured"]:
        raise RuntimeError("no_credentials")      # no API key and not running on GCP
    last = "no_credentials"
    for model in GEMINI_MODELS:
        try:
            if key:
                backend = "api_key"
                r = requests.post(GEMINI_API_URL.format(model=model), json=payload, timeout=30,
                                  headers={"x-goog-api-key": key, "Content-Type": "application/json"})
            else:
                backend = "vertex_adc"
                token, project = _vertex_token_and_project()
                loc = os.environ.get("VERTEX_LOCATION", "us-central1")
                host = "aiplatform.googleapis.com" if loc == "global" else f"{loc}-aiplatform.googleapis.com"
                r = requests.post(VERTEX_URL.format(host=host, project=project, loc=loc, model=model),
                                  json=payload, timeout=30,
                                  headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
            if r.status_code != 200:
                last = f"http_{r.status_code}"
                log.warning("gemini %s via %s -> HTTP %s: %s", model, backend, r.status_code, r.text[:200])
                continue
            parts = r.json()["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts).strip()
            if text:
                return text, model, backend
            last = "empty_response"
        except Exception as exc:  # noqa: BLE001 — credentials, timeout, parse...
            last = type(exc).__name__
            log.warning("gemini %s failed: %s", model, exc)
    raise RuntimeError(last)


def _template_report(body: ReportRequest, lang: str) -> dict:
    """Deterministic, metrics-based fallback. Honest about weak results; no gen-AI claim."""
    f = _facts(body)
    v, cos, vis = f["verdict"], f["worst_case_cosine_similarity_on_final_photo"], f["visibility"] or "n/a"
    if lang == "id":
        head = {"broken": "Pencocokan identitas terpatahkan pada model yang diuji.",
                "partial": "Proteksi sebagian: beberapa model mungkin masih mengenali.",
                "weak": "Proteksi lemah: foto kemungkinan masih dikenali."}[v]
        nums = [f"Kemiripan terburuk (semua model & kompresi): {cos:+.2f} (ambang cocok ≈ {MATCH_COS:.2f}).",
                f"Tahan kompresi & resize: {'ya' if f['survives_jpeg_and_resize'] else 'tidak'}.",
                f"Tampilan: {vis} (SSIM wajah {f['ssim_face']})."]
        limits = ("Diuji pada beberapa model open-source dalam simulasi — bukan jaminan terhadap sistem komersial "
                  "atau AI masa depan. Proteksi menghentikan pencocokan identitas, bukan deteksi wajah.")
        tips = ["Coba kekuatan 'max' jika hasilnya partial/weak.", "Unggah foto yang sudah diproteksi, bukan aslinya."]
        summ = head
    else:
        head = {"broken": "Identity matching is broken on every model we checked.",
                "partial": "Partial protection: some models may still recognise the face.",
                "weak": "Weak protection: the face is likely still recognisable."}[v]
        nums = [f"Worst-case similarity (all models, all upload transforms): {cos:+.2f} (match threshold ≈ {MATCH_COS:.2f}).",
                f"Survives compression and resizing: {'yes' if f['survives_jpeg_and_resize'] else 'no'}.",
                f"Appearance: {vis} (face SSIM {f['ssim_face']})."]
        limits = ("Checked on a few open-source models in simulation — not a guarantee against commercial systems "
                  "or future AI. It stops identity matching, not face detection.")
        tips = ["Try 'max' strength if the result is partial or weak.", "Post the protected file, never the original."]
        summ = head
    return {"headline": head, "summary": summ, "numbers": nums, "limits": limits, "tips": tips}


def _flatten(s: dict) -> str:
    return "\n".join([s["headline"], "", s["summary"], "", *[f"• {n}" for n in s["numbers"]], "", s["limits"], "",
                      *[f"Tip: {t}" for t in s["tips"]]])


@app.post("/report")
def report(body: ReportRequest):
    lang = body.lang if body.lang in ("id", "en") else "id"
    facts = _facts(body)
    user_text = "Verified protection results (JSON):\n" + json.dumps(facts, ensure_ascii=False)
    try:
        text, model, backend = _gemini_generate(_SYSTEM[lang], user_text, _REPORT_SCHEMA)
        sections = json.loads(text)
        for k in ("headline", "summary", "numbers", "limits", "tips"):
            if k not in sections:
                raise ValueError(f"missing {k}")
        return {"report": _flatten(sections), "sections": sections, "lang": lang,
                "source": "gemini", "model": model, "backend": backend}
    except Exception as exc:  # noqa: BLE001
        code = str(exc) if isinstance(exc, RuntimeError) else f"bad_output:{type(exc).__name__}"
        sections = _template_report(body, lang)
        return {"report": _flatten(sections), "sections": sections, "lang": lang,
                "source": "template", "model": "none", "fallback_reason": code}


@app.get("/report/selftest")
def report_selftest():
    """Live Gemini round-trip. Run this BEFORE recording the demo: source must be 'gemini'."""
    try:
        text, model, backend = _gemini_generate("Reply with the single word: ok", "ping")
        return {"ok": True, "source": "gemini", "model": model, "backend": backend, "reply": text[:40]}
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=503, content={"ok": False, "source": "none", "reason": str(exc),
                                                      "status": _gemini_status()})
