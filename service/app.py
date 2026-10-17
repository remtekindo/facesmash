"""
Face Cloak microservice — thin FastAPI wrapper around prototype/cloak.py.

Implements the team API contract v0.1 (see ../api-contract.md), which the
Flutter app and Laravel gateway are already building against with mocks.

The heavy ML (LowKey-style PGD cloaking vs InceptionResnetV1) is reused
as-is from prototype/cloak.py; this file only handles HTTP, validation,
metrics, and response shaping.
"""
import base64
import io
import os
import random
import sys
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

# Reuse the validated prototype code without copying or modifying it.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prototype"))
from cloak import (  # noqa: E402
    cloak_face,
    detect_face_box,
    jpeg_roundtrip,
    load_models,
    make_embed_fn,
    pil_to_tensor,
    tensor_to_pil,
)

VERSION = "0.1"
MAX_BYTES = 10 * 1024 * 1024          # 10 MB
MAX_DIM = 2048                        # downscale if larger
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
STRENGTH_STEPS = {"low": 30, "standard": 60, "max": 90}
# Embedding L2 distance above which identity matching breaks (per prototype/RESULTS.md)
MATCH_BREAK_THRESHOLD = 1.0

_models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup (~2.3 s on CPU), not per request.
    mtcnn, resnet = load_models("cpu")
    _models["mtcnn"] = mtcnn
    _models["embed_fn"] = make_embed_fn(resnet, "cpu")
    yield
    _models.clear()


app = FastAPI(title="Face Cloak Service", version=VERSION, lifespan=lifespan)

# CORS: per api-contract.md v0.2 the Nuxt web app POSTs /report directly to
# this service from the browser. A cross-origin JSON POST triggers a CORS
# preflight (OPTIONS); without this middleware FastAPI answers 405 and the
# browser blocks the call. Server-to-server calls (Nuxt server route ->
# /protect) are unaffected by this. Restrict via CORS_ALLOW_ORIGINS
# (comma-separated) once the web app's URL is known; default "*" is fine
# for the competition demo since the service is --allow-unauthenticated.
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
        "endpoints": ["POST /protect", "POST /report"],
    }


@app.exception_handler(HTTPException)
async def _http_exc_handler(request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content=detail)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": "mtcnn" in _models, "version": VERSION}


def _psnr_db(a: Image.Image, b: Image.Image) -> float:
    arr_a = np.asarray(a).astype(np.float64)
    arr_b = np.asarray(b).astype(np.float64)
    mse = float(np.mean((arr_a - arr_b) ** 2))
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(255.0**2 / mse))


@app.post("/protect")
async def protect(
    image: UploadFile = File(...),
    strength: str = Form("standard"),
    seed: int | None = Form(None),
    jpeg_simulate: bool = Form(True),
):
    t_start = time.time()

    if strength not in STRENGTH_STEPS:
        raise HTTPException(400, {"error": "BAD_STRENGTH",
                                  "message": f"strength must be one of {sorted(STRENGTH_STEPS)}"})

    raw = await image.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, {"error": "TOO_LARGE",
                                  "message": f"image exceeds {MAX_BYTES // 1024 // 1024} MB; downscale client-side first"})
    if (image.content_type or "") not in ALLOWED_TYPES:
        raise HTTPException(400, {"error": "BAD_IMAGE",
                                  "message": f"unsupported content type: {image.content_type}"})
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()  # force full decode to catch truncated files
        img = img.convert("RGB")
    except Exception:
        raise HTTPException(400, {"error": "BAD_IMAGE", "message": "unreadable image file"})

    downscaled = False
    if max(img.size) > MAX_DIM:
        img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
        downscaled = True

    mtcnn, embed_fn = _models["mtcnn"], _models["embed_fn"]
    box, prob_before = detect_face_box(mtcnn, img)
    if box is None:
        raise HTTPException(422, {"error": "NO_FACE",
                                  "message": "no face detected; use a clearer front-facing photo"})

    run_seed = seed if seed is not None else random.randint(0, 2**31 - 1)

    crop = img.crop(box).resize((160, 160), Image.BILINEAR)
    face_tensor = pil_to_tensor(crop)
    cloaked_crop, _dists, _pgd_s = cloak_face(
        face_tensor, embed_fn, steps=STRENGTH_STEPS[strength], seed=run_seed, verbose=False)

    out = img.copy()
    out.paste(tensor_to_pil(cloaked_crop).resize((box[2] - box[0], box[3] - box[1]), Image.BILINEAR), box)

    with torch.no_grad():
        e0 = embed_fn(face_tensor)
        e1 = embed_fn(cloaked_crop)
        emb_dist = float(torch.norm(e1 - e0, p=2))
        cos_sim = float(torch.nn.functional.cosine_similarity(e0, e1, dim=0))

    jpeg_q75_distance = None
    jpeg_q75_survived = None
    if jpeg_simulate:
        # Simulated social-media upload: does protection survive JPEG Q75?
        q75 = jpeg_roundtrip(cloaked_crop, quality=75)
        with torch.no_grad():
            e2 = embed_fn(q75)
            jpeg_q75_distance = float(torch.norm(e2 - e0, p=2))
        jpeg_q75_survived = bool(jpeg_q75_distance >= MATCH_BREAK_THRESHOLD)

    _box2, prob_after = detect_face_box(mtcnn, out)
    psnr = _psnr_db(img, out)

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    elapsed_ms = int((time.time() - t_start) * 1000)

    # Firestore job history: one doc per completed /protect — params,
    # metrics and verdict only, never the photo. Returns None (and logs a
    # warning) if persistence is disabled or the write fails.
    firestore_job_id = save_protection_job({
        "strength": strength,
        "seed": run_seed,
        "downscaled": downscaled,
        "processing_time_ms": elapsed_ms,
        "engine": f"cloak-py {VERSION} (LowKey-style PGD)",
        "surrogate_model": "InceptionResnetV1 / VGGFace2",
        "match_break_threshold": MATCH_BREAK_THRESHOLD,
        "metrics": {
            "detection_before": round(prob_before, 4),
            "detection_after": round(prob_after, 4),
            "embedding_distance": round(emb_dist, 4),
            "cosine_similarity": round(cos_sim, 4),
            "jpeg_q75_survived": jpeg_q75_survived,
            "jpeg_q75_distance": round(jpeg_q75_distance, 4) if jpeg_q75_distance is not None else None,
            "psnr_db": round(psnr, 1),
        },
        "verdict": {
            "identity_match_broken": bool(emb_dist >= MATCH_BREAK_THRESHOLD),
            "jpeg_q75_survived": jpeg_q75_survived,
        },
        "version": VERSION,
    })

    meta = {
        "engine": f"cloak-py {VERSION} (LowKey-style PGD)",
        "surrogate_model": "InceptionResnetV1 / VGGFace2",
        "strength": strength,
        "seed": run_seed,
        "processing_time_ms": elapsed_ms,
        "downscaled": downscaled,
    }
    if firestore_job_id:
        meta["firestore_job_id"] = firestore_job_id

    return {
        "protected_image": b64,
        "metrics": {
            "detection_before": round(prob_before, 4),
            "detection_after": round(prob_after, 4),
            "embedding_distance": round(emb_dist, 4),
            "cosine_similarity": round(cos_sim, 4),
            "jpeg_q75_survived": jpeg_q75_survived,
            "jpeg_q75_distance": round(jpeg_q75_distance, 4) if jpeg_q75_distance is not None else None,
            "psnr_db": round(psnr, 1),
        },
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# POST /report — Gemini plain-language evaluation report (contract v0.2,
# implemented 2026-10-13). Moved server-side into this service so the Nuxt app
# just POSTs here; no separate Nuxt server route / Gemini integration needed.
# The API key never leaves the server (env var). Without a key the endpoint
# serves the contract's mock response, so the UI contract is identical.
# ---------------------------------------------------------------------------

class ReportMetrics(BaseModel):
    detection_before: float = 1.0
    detection_after: float = 0.9997
    embedding_distance: float = Field(..., ge=0)
    cosine_similarity: float = 0.0
    jpeg_q75_survived: bool = True
    psnr_db: float = 32.7


class ReportMeta(BaseModel):
    strength: str = "standard"
    processing_time_ms: int = 0


class ReportRequest(BaseModel):
    metrics: ReportMetrics
    meta: ReportMeta = ReportMeta()
    lang: str = "id"  # "id" (default) or "en"


GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Honest-limits guardrails (must survive into every generated report):
# the cloak breaks *identity matching*, not face detection; it was validated
# against ONE surrogate model (InceptionResnetV1/VGGFace2), one test face, and
# a simulated social-media pipeline — never claim it defeats Meta/Google/Apple
# models or guarantees absolute protection.
_REPORT_SYSTEM_ID = (
    "Kamu adalah asisten privasi dalam aplikasi pelindung foto. Jelaskan hasil proteksi "
    "di bawah ini dengan bahasa Indonesia yang sederhana dan menenangkan, untuk pengguna awam. "
    "Struktur: (1) satu kalimat kesimpulan — apakah foto ini sekarang aman dari pengenalan wajah AI; "
    "(2) jelaskan tiap angka dalam 1 kalimat bahasa sehari-hari; "
    "(3) satu paragraf jujur tentang batasan — proteksi ini mematahkan pencocokan identitas "
    "(bukan deteksi wajah) dan teruji pada satu model tertentu dalam simulasi, bukan jaminan mutlak "
    "terhadap semua AI dan bukan klaim bisa mengalahkan model perusahaan besar; "
    "(4) satu tips praktis. Maksimal 150 kata. Jangan melebih-lebihkan klaim."
)
_REPORT_SYSTEM_EN = (
    "You are a privacy assistant inside a photo-protection app. Explain the protection results "
    "below in simple, reassuring plain English for a non-technical user. "
    "Structure: (1) one concluding sentence — whether this photo is now safe from AI face recognition; "
    "(2) one everyday-language sentence per number; "
    "(3) one honest paragraph on limits — this breaks identity matching (not face detection) and was "
    "validated against a single surrogate model in simulation, not a guarantee against all AI and not "
    "a claim to defeat big-tech models; "
    "(4) one practical tip. Max 150 words. Never overstate claims."
)

_REPORT_MOCK_ID = (
    "Foto kamu sekarang jauh lebih aman. Sistem pengenal wajah AI tidak lagi bisa mengaitkan foto ini "
    "dengan identitasmu (skor kemiripan turun jauh di bawah batas). Proteksinya tetap bertahan walau foto "
    "dikompres saat diupload ke media sosial, dan secara kasat mata fotomu tidak berubah. Batasan jujur: "
    "pengujian dilakukan terhadap model tertentu — ini perlindungan kuat, bukan jaminan mutlak terhadap "
    "semua AI di masa depan. Tips: gunakan kekuatan 'max' untuk foto yang sangat sensitif."
)
_REPORT_MOCK_EN = (
    "Your photo is now much safer. AI face-recognition systems can no longer link this photo to your "
    "identity (the similarity score dropped far below the threshold). The protection survives compression "
    "when uploaded to social media, and your photo looks unchanged to the naked eye. Honest limits: "
    "testing was done against a specific model — this is strong protection, not an absolute guarantee "
    "against all future AI. Tip: use 'max' strength for highly sensitive photos."
)


def _report_user_prompt(m: ReportMetrics, meta: ReportMeta, lang: str) -> str:
    if lang == "en":
        return (
            f"Photo protection results: embedding distance {m.embedding_distance} (identity threshold ~1.0 — "
            f"farther means less recognizable), cosine similarity {m.cosine_similarity}, face detection before "
            f"{m.detection_before} / after {m.detection_after}, survives social-media compression: "
            f"{m.jpeg_q75_survived}, visual quality {m.psnr_db} dB (above 30 = nearly invisible). "
            f"Strength: {meta.strength}."
        )
    return (
        f"Hasil proteksi foto: jarak embedding {m.embedding_distance} (batas identitas ~1.0 — semakin jauh "
        f"semakin tidak dikenali), kemiripan kosinus {m.cosine_similarity}, deteksi wajah sebelum "
        f"{m.detection_before} / sesudah {m.detection_after}, tahan kompresi media sosial: "
        f"{m.jpeg_q75_survived}, kualitas visual {m.psnr_db} dB (di atas 30 = nyaris tak terlihat). "
        f"Kekuatan: {meta.strength}."
    )


def _gemini_report(system: str, user_prompt: str) -> tuple[str, str] | None:
    """Call Gemini generateContent REST API. Returns (report text, model), or None on any failure."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_GEMINI_API_KEY")
    if not api_key:
        return None
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    try:
        r = requests.post(
            GEMINI_API_URL.format(model=model),
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json={
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [{"parts": [{"text": user_prompt}]}],
                "generationConfig": {"temperature": 0.7, "maxOutputTokens": 400},
            },
            timeout=30,
        )
        r.raise_for_status()
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        return (text, model)
    except Exception:
        return None


@app.post("/report")
def report(body: ReportRequest):
    lang = body.lang if body.lang in ("id", "en") else "id"
    system = _REPORT_SYSTEM_ID if lang == "id" else _REPORT_SYSTEM_EN
    user_prompt = _report_user_prompt(body.metrics, body.meta, lang)

    result = _gemini_report(system, user_prompt)
    if result is not None:
        text, model = result
        return {"report": text, "lang": lang, "model": model}

    # No key, or Gemini failed: serve the contract mock (demo-safe fallback).
    mock = _REPORT_MOCK_ID if lang == "id" else _REPORT_MOCK_EN
    resp = {"report": mock, "lang": lang, "model": "mock"}
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_GEMINI_API_KEY"):
        resp["note"] = "gemini_unavailable_fallback"
    return resp
