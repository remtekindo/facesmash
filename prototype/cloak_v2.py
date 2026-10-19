"""
cloak_v2.py — perceptually-constrained, ensemble, margin-based facial cloaking.
v2.1 (Qwen Update): Enhanced EOT for social media resilience + spatial jitter for detector shift.

Why v2 exists (measured, see prototype/RESULTS_v2.md):
  v1 (cloak.py) maximised the embedding distance of ONE model on a 160x160 crop,
  saturated the L_inf budget everywhere (visible blotches / texture on flat skin
  and backgrounds) and reported the distance measured on that same crop. Judged
  on the real output photo (re-detected, re-cropped, re-embedded), protection was
  weak and did not transfer to other recognisers.

What changed:
  1. ENSEMBLE: VGGFace2 + CASIA-WebFace (facenet-pytorch) + ArcFace-R50 (optional,
     ONNX converted to torch) are attacked simultaneously -> far better transfer.
  2. MARGIN LOSS: hinge on cosine similarity (stop pushing once every model is
     below a target), plus an energy penalty -> uses only the budget it needs.
  3. PERCEPTUAL BUDGET: a differentiable DSSIM penalty (Fawkes-style) inside the
     loop, plus a soft face-ellipse mask and a generous L_inf cap. The optimiser
     itself decides where noise is cheapest to hide (textured areas) instead of
     spraying +-eps everywhere. Luma-dominant, very low chroma (colour shifts such
     as green irises were the most noticeable artefact in testing).
  4. EOT: random crop/landmark jitter, down-up resampling and straight-through
     real JPEG inside the loop, so protection survives re-detection, resize and
     compression (not only a perfect re-crop). [v2.1: Added mild blur + harsher JPEG 40-91]
  5. HONEST VERIFICATION on the FINAL composited image: re-detect, re-embed, under
     PNG / JPEG-75 / down-50%+JPEG-75; report the WORST case.

Method lineage: LowKey (Cherepanova et al., ICLR 2021); ensemble/EOT ideas as in
Fawkes / LowKey / standard transfer-attack practice. Models: facenet-pytorch,
insightface (ArcFace, MIT-licensed code; check model licence for commercial use).
"""
from __future__ import annotations

import io
import logging
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import roi_align

log = logging.getLogger("cloak_v2")

ARC_TPL = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)

# cosine similarity at/below which we call the identity match "broken".
# (ArcFace-style verification thresholds sit around 0.25-0.35; impostor mean ~0.)
MATCH_COS_THRESHOLD = 0.30

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Cfg:
    eps: float = 24 / 255         # L_inf cap (a safety bound; DSSIM does the real work)
    chroma: float = 0.12          # chroma amplitude relative to luma
    steps: int = 100
    lr: float = 0.12
    target_cos: float = -0.05     # hinge target (cos <= this on every model)
    ssim_w: float = 10.0          # DSSIM penalty weight (higher = less visible, weaker)
    work_face: int = 320          # face-box width (px) used for optimisation
    region_margin: float = 0.35   # region = box expanded by this fraction
    ellipse: tuple = (0.62, 0.68) # semi-axes as fraction of box (w, h)
    eot: bool = True
    eot_min_scale: float = 0.75   # strongest down-up resampling in EOT
    models_per_step: int = 0      # 0 = all models every step; k = random subset
    seed: int | None = None

# strength -> (eps/255, steps, target_cos, ssim_w). Calibrated on 8 test faces
# (prototype/RESULTS_v2.md); "max" trades visibility for protection.
STRENGTH = {
    "low":      (20.0, 70,  0.10, 35.0),
    "standard": (24.0, 100, -0.05, 10.0),
    "max":      (28.0, 150, -0.20, 5.0),
}

def cfg_for(strength: str, **kw) -> Cfg:
    eps, steps, tgt, ssim_w = STRENGTH[strength]
    c = Cfg(eps=eps / 255, steps=steps, target_cos=tgt, ssim_w=ssim_w)
    for k, v in kw.items():
        setattr(c, k, v)
    return c

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class Ensemble:
    """Differentiable face-embedding ensemble. Inputs are [0,1] RGB tensors."""

    ONNX_ENV = {"arcface": "FACESMASH_ARCFACE_ONNX", "mbf": "FACESMASH_MBF_ONNX"}

    def __init__(self, names=("vggface2", "casia", "mbf"), onnx_paths=None, holdout=()):
        """holdout: names that are only used to VERIFY (never optimised against) —
        an honest, independent check of transfer."""
        from facenet_pytorch import InceptionResnetV1
        self.holdout = set(holdout)
        self.nets: list[tuple[str, str, torch.nn.Module]] = []
        for n in names:
            if n == "vggface2":
                self.nets.append((n, "facenet", InceptionResnetV1(pretrained="vggface2").eval()))
            elif n == "casia":
                self.nets.append((n, "facenet", InceptionResnetV1(pretrained="casia-webface").eval()))
            elif n in self.ONNX_ENV:
                path = (onnx_paths or {}).get(n) or os.environ.get(self.ONNX_ENV[n], "")
                if not path or not os.path.exists(path):
                    log.warning("%s ONNX not found (%s); running without it", n, path)
                    continue
                try:
                    from onnx2torch import convert
                    self.nets.append((n, "arcface", convert(path).eval()))
                except Exception as exc:  # noqa: BLE001
                    log.warning("%s conversion failed (%s); running without it", n, exc)
        for _, _, net in self.nets:
            for p in net.parameters():
                p.requires_grad_(False)

    @property
    def names(self):
        return [n for n, _, _ in self.nets]

    @property
    def train_idx(self):
        return [i for i, (n, _, _) in enumerate(self.nets) if n not in self.holdout]

    def embed(self, name_idx: int, view: torch.Tensor) -> torch.Tensor:
        _, kind, net = self.nets[name_idx]
        if kind == "facenet":
            return F.normalize(net((view * 255.0 - 127.5) / 128.0), dim=1)
        return F.normalize(net((view - 0.5) / 0.5), dim=1)

# --------------------------------------------------------------------------- #
# Geometry helpers (all differentiable w.r.t. the image, not the geometry)
# --------------------------------------------------------------------------- #
def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """2x3 similarity transform mapping src->dst (least squares)."""
    sm, dm = src.mean(0), dst.mean(0)
    s0, d0 = src - sm, dst - dm
    var = (s0 ** 2).sum() / len(src)
    cov = d0.T @ s0 / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    scale = (S * np.diag(D)).sum() / max(var, 1e-9)
    t = dm - scale * R @ sm
    return np.concatenate([scale * R, t[:, None]], axis=1).astype(np.float32)

def facenet_view(x, box, margin, out=160):
    w, h = box[2] - box[0], box[3] - box[1]
    m = margin * max(w, h)
    b = torch.tensor([[0.0, box[0] - m, box[1] - m, box[2] + m, box[3] + m]], dtype=x.dtype)
    return roi_align(x[None], b, (out, out), spatial_scale=1.0, sampling_ratio=2, aligned=True)

def arc_view(x_aa, pts, out=112):
    """Aligned 112x112 crop via differentiable grid_sample (pixel-centre coords)."""
    M = _umeyama(ARC_TPL, pts.astype(np.float32))          # template -> source
    ys, xs = torch.meshgrid(torch.arange(out, dtype=torch.float32),
                            torch.arange(out, dtype=torch.float32), indexing="ij")
    ones = torch.ones_like(xs)
    P = torch.stack([xs, ys, ones], dim=-1).reshape(-1, 3)   # (N,3)
    src = P @ torch.from_numpy(M).T                          # (N,2) in source px
    H, W = x_aa.shape[-2:]
    gx = 2 * src[:, 0] / (W - 1) - 1
    gy = 2 * src[:, 1] / (H - 1) - 1
    grid = torch.stack([gx, gy], dim=-1).reshape(1, out, out, 2)
    return F.grid_sample(x_aa[None], grid, mode="bilinear", padding_mode="border", align_corners=True)

def _jitter_pts(pts, rng, w):
    c = pts.mean(0)
    s = 1.0 + rng.normal(0, 0.05)
    a = math.radians(rng.normal(0, 4.0))
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], dtype=np.float32)
    t = rng.normal(0, 0.03 * w, size=2).astype(np.float32)
    return ((pts - c) * s) @ R.T + c + t

def jpeg_ste(x: torch.Tensor, quality: int) -> torch.Tensor:
    """Forward = real JPEG, backward = identity (straight-through)."""
    arr = (x.detach().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=int(quality))
    buf.seek(0)
    y = torch.from_numpy(np.asarray(Image.open(buf).convert("RGB")).astype(np.float32) / 255.0).permute(2, 0, 1)
    return x + (y - x).detach()

def _gauss_blur(t, sigma):
    if sigma <= 0:
        return t
    k = int(2 * round(3 * sigma) + 1)
    ax = torch.arange(k, dtype=t.dtype) - k // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    c = t.shape[1]
    t = F.conv2d(F.pad(t, (k // 2, k // 2, 0, 0), mode="replicate"), g.view(1, 1, 1, k).repeat(c, 1, 1, 1), groups=c)
    t = F.conv2d(F.pad(t, (0, 0, k // 2, k // 2), mode="replicate"), g.view(1, 1, k, 1).repeat(c, 1, 1, 1), groups=c)
    return t

# --------------------------------------------------------------------------- #
# Perceptual budget map
# --------------------------------------------------------------------------- #
def face_mask(x0: torch.Tensor, box, cfg: Cfg) -> torch.Tensor:
    """(1,H,W) soft face ellipse in [0,1] (smoothstep edge -> no visible box border)."""
    _, H, W = x0.shape
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    a, b = cfg.ellipse[0] * (box[2] - box[0]), cfg.ellipse[1] * (box[3] - box[1])
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    r = torch.sqrt(((xs - cx) / a) ** 2 + ((ys - cy) / b) ** 2)
    t = ((1.0 - r) / 0.35).clamp(0, 1)            # 1 inside r<=0.65, 0 at r>=1
    return (t * t * (3 - 2 * t))[None]

def _ssim_map(a, b, win=7, sigma=1.5):
    """Differentiable SSIM map, (3,H,W) inputs in [0,1] -> (H,W) mean over channels."""
    ax = torch.arange(win, dtype=a.dtype) - win // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    k = (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)
    f = lambda t: F.conv2d(t[None], k, padding=win // 2, groups=3)[0]
    mu_a, mu_b = f(a), f(b)
    va, vb, cab = f(a * a) - mu_a ** 2, f(b * b) - mu_b ** 2, f(a * b) - mu_a * mu_b
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_a * mu_b + C1) * (2 * cab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))
    return m.mean(0)

def _to_rgb_delta(raw: torch.Tensor, chroma: float) -> torch.Tensor:
    """raw (3,H,W) in [-1,1] = (Y, Cb, Cr) -> RGB offsets in ~[-1,1]."""
    Y, Cb, Cr = raw[0], raw[1] * chroma, raw[2] * chroma
    d = torch.stack([Y + 1.402 * Cr, Y - 0.344 * Cb - 0.714 * Cr, Y + 1.772 * Cb])
    return d.clamp(-1, 1)

# --------------------------------------------------------------------------- #
# Core optimisation for ONE face
# --------------------------------------------------------------------------- #
@dataclass
class FaceJob:
    box: tuple            # (x1,y1,x2,y2) tight MTCNN box, native coords
    pts: np.ndarray       # (5,2) landmarks, native coords
    prob: float = 1.0

@dataclass
class FaceResult:
    box: tuple
    steps: int
    seconds: float
    final_loss_cos: dict = field(default_factory=dict)

def _views(ens: Ensemble, x, box, pts, rng, jitter: bool, idx=None):
    """One embedding view per model (dict idx->tensor) for image tensor x (3,H,W)."""
    outs = {}
    w = box[2] - box[0]
    for i, (_, kind, _) in enumerate(ens.nets):
        if idx is not None and i not in idx:
            continue
        if kind == "facenet":
            if jitter:
                s = 1.0 + rng.normal(0, 0.05); sh = rng.normal(0, 0.03, size=2) * w
                cx, cy = (box[0] + box[2]) / 2 + sh[0], (box[1] + box[3]) / 2 + sh[1]
                hw, hh = (box[2] - box[0]) / 2 * s, (box[3] - box[1]) / 2 * s
                bj = (cx - hw, cy - hh, cx + hw, cy + hh)
                margin = float(rng.choice([0.0, 0.08, 0.15]))
            else:
                bj, margin = box, 0.08
            outs[i] = facenet_view(x, bj, margin)
        else:
            sa = min(1.0, 224.0 / max(w, 1.0))
            if sa < 1.0:
                xa = F.interpolate(x[None], scale_factor=sa, mode="bilinear", antialias=True, recompute_scale_factor=False)[0]
            else:
                xa = x
            p = pts * (xa.shape[-1] / x.shape[-1])
            pj = _jitter_pts(p, rng, w * sa) if jitter else p
            outs[i] = arc_view(xa, pj)
    return outs

def cloak_face(x_native: torch.Tensor, job: FaceJob, ens: Ensemble, cfg: Cfg, verbose=False):
    """Return (delta_native (3,h,w) over the face REGION, region_xyxy, FaceResult)."""
    t0 = time.time()
    rng = np.random.default_rng(cfg.seed)
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
    _, H, W = x_native.shape
    bx1, by1, bx2, by2 = job.box
    bw, bh = bx2 - bx1, by2 - by1
    m = cfg.region_margin
    rx1, ry1 = max(0, int(bx1 - m * bw)), max(0, int(by1 - m * bh))
    rx2, ry2 = min(W, int(bx2 + m * bw) + 1), min(H, int(by2 + m * bh) + 1)
    region = x_native[:, ry1:ry2, rx1:rx2]
    rh, rw = region.shape[-2:]

    s = min(1.0, cfg.work_face / max(bw, 1.0))
    if s < 1.0:
        wh, ww = max(8, round(rh * s)), max(8, round(rw * s))
        x0 = F.interpolate(region[None], size=(wh, ww), mode="bilinear", antialias=True, align_corners=False)[0]
    else:
        x0, wh, ww = region, rh, rw
    sx, sy = ww / rw, wh / rh
    box_w = ((bx1 - rx1) * sx, (by1 - ry1) * sy, (bx2 - rx1) * sx, (by2 - ry1) * sy)
    pts_w = (job.pts - np.array([rx1, ry1], dtype=np.float32)) * np.array([sx, sy], dtype=np.float32)

    mask = face_mask(x0, box_w, cfg)                       # (1,h,w)
    # reference identity embedding (mean over mildly-jittered clean views)
    with torch.no_grad():
        refs = {i: [] for i in ens.train_idx}
        for k in range(6):
            for i, v in _views(ens, x0, box_w, pts_w, rng, jitter=k > 0, idx=ens.train_idx).items():
                refs[i].append(ens.embed(i, v))
        e0 = {i: F.normalize(torch.cat(r).mean(0, keepdim=True), dim=1) for i, r in refs.items()}

    face_w = mask[0] + 1e-6
    p = (torch.randn(3, wh, ww) * 0.05).requires_grad_(True)
    opt = torch.optim.Adam([p], lr=cfg.lr)
    train = ens.train_idx
    last_cos = {i: 1.0 for i in train}

    for step in range(cfg.steps):
        for g in opt.param_groups:   # cosine lr decay to 20%
            g["lr"] = cfg.lr * (0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * step / cfg.steps)))
        delta = _to_rgb_delta(torch.tanh(p), cfg.chroma) * (cfg.eps * mask)
        x = (x0 + delta).clamp(0, 1)
        if cfg.eot:
            # 1. Resolution loss (down-up sampling) - increased probability
            if rng.random() < 0.6:
                r = float(rng.uniform(cfg.eot_min_scale, 0.95))
                small = F.interpolate(x[None], scale_factor=r, mode="bilinear", antialias=True, recompute_scale_factor=False)
                x = F.interpolate(small, size=(wh, ww), mode="bilinear", align_corners=False)[0]
            
            # 2. Social Media Compression Simulation (Harsher JPEG + Mild Blur)
            if rng.random() < 0.7:
                # Mild blur to simulate sharpening artifacts after social media compression
                if rng.random() < 0.5:
                    kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1) / 16.0
                    x = x.unsqueeze(0)
                    x = F.conv2d(x, kernel, padding=1, groups=3).squeeze(0)
                # Real JPEG, straight-through with lower bound (40) for aggressive compression simulation
                x = jpeg_ste(x, int(rng.integers(40, 91)))
                
        idx = list(train)
        if cfg.models_per_step and cfg.models_per_step < len(train):
            idx = sorted(rng.choice(train, cfg.models_per_step, replace=False).tolist())
        views = _views(ens, x, box_w, pts_w, rng, jitter=cfg.eot, idx=idx)
        loss_adv = 0.0
        for i in idx:
            cos = (ens.embed(i, views[i]) * e0[i]).sum()
            last_cos[i] = float(cos.detach())
            loss_adv = loss_adv + torch.relu(cos - cfg.target_cos)
        sm = _ssim_map(x0, (x0 + delta).clamp(0, 1))
        dssim = (1 - (sm * face_w).sum() / face_w.sum()) / 2
        loss = loss_adv / len(idx) + cfg.ssim_w * dssim
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if verbose and (step + 1) % 10 == 0:
            print(f"  step {step+1}/{cfg.steps} cos={[round(c,3) for c in last_cos.values()]} loss={float(loss.detach()):.4f}")

    with torch.no_grad():
        delta = _to_rgb_delta(torch.tanh(p), cfg.chroma) * (cfg.eps * mask)
        if (wh, ww) != (rh, rw):
            delta_n = F.interpolate(delta[None], size=(rh, rw), mode="bicubic", align_corners=False)[0]
            cap = F.interpolate((cfg.eps * mask)[None], size=(rh, rw), mode="bilinear", align_corners=False)[0]
            delta_n = torch.max(torch.min(delta_n, cap), -cap)
        else:
            delta_n = delta
    return delta_n, (rx1, ry1, rx2, ry2), FaceResult(job.box, cfg.steps, time.time() - t0, last_cos)

# --------------------------------------------------------------------------- #
# Image-level API
# --------------------------------------------------------------------------- #
def pil_to_t(img):
    return torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()

def t_to_pil(t):
    return Image.fromarray((t.detach().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8))

def detect_faces(mtcnn, img, max_faces=3, min_prob=0.9) -> list[FaceJob]:
    boxes, probs, pts = mtcnn.detect(img, landmarks=True)
    if boxes is None:
        return []
    jobs = [FaceJob(tuple(float(v) for v in b), p.astype(np.float32), float(pr))
            for b, pr, p in zip(boxes, probs, pts) if pr is not None and pr >= min_prob]
    jobs.sort(key=lambda j: -(j.box[2] - j.box[0]) * (j.box[3] - j.box[1]))
    return jobs[:max_faces]

def cloak_image(img: Image.Image, jobs: list[FaceJob], ens: Ensemble, cfg: Cfg, verbose=False):
    """Cloak every job in-place on a float copy. Returns (PIL out, [FaceResult])."""
    x = pil_to_t(img)
    results = []
    for k, job in enumerate(jobs):
        c = Cfg(**{**cfg.__dict__, "seed": None if cfg.seed is None else cfg.seed + k})
        delta, (rx1, ry1, rx2, ry2), fr = cloak_face(x, job, ens, c, verbose)
        x[:, ry1:ry2, rx1:rx2] = (x[:, ry1:ry2, rx1:rx2] + delta).clamp(0, 1)
        results.append(fr)
    return t_to_pil(x), results

# --------------------------------------------------------------------------- #
# Honest verification on the FINAL image
# --------------------------------------------------------------------------- #
def _jpeg(img, q):
    b = io.BytesIO(); img.save(b, "JPEG", quality=q); b.seek(0)
    return Image.open(b).convert("RGB")

UPLOAD_TRANSFORMS = {
    "png": lambda im: im,
    "jpeg75": lambda im: _jpeg(im, 75),
    "down50_jpeg75": lambda im: _jpeg(im.resize((max(8, im.width // 2), max(8, im.height // 2)), Image.LANCZOS), 75),
}

@torch.no_grad()
def reference_embeddings(img, mtcnn, ens, jobs):
    xt = pil_to_t(img)
    return [{i: ens.embed(i, v) for i, v in _views(ens, xt, j.box, j.pts, np.random.default_rng(0), jitter=False).items()}
            for j in jobs]

@torch.no_grad()
def verify(orig, out, mtcnn, ens, jobs, refs=None):
    """Re-detect each protected face on the output, embed with every model under
    several upload transforms. Returns dict with per-model worst-case cosine.
    cos <= MATCH_COS_THRESHOLD  =>  identity match broken for that model."""
    refs = refs or reference_embeddings(orig, mtcnn, ens, jobs)
    res = {"per_transform": {}, "models": ens.names}
    worst = {n: -1.0 for n in ens.names}
    for tname, fn in UPLOAD_TRANSFORMS.items():
        im = fn(out)
        sc = im.width / out.width
        per_face = []
        for j, ref in zip(jobs, refs):
            # locate the same face again, the way a scraper would
            boxes, probs, pts = mtcnn.detect(im, landmarks=True)
            if boxes is None:
                per_face.append({n: None for n in ens.names}); continue
            cx, cy = (j.box[0] + j.box[2]) / 2 * sc, (j.box[1] + j.box[3]) / 2 * sc
            k = int(np.argmin([(abs((b[0]+b[2])/2-cx) + abs((b[1]+b[3])/2-cy)) for b in boxes]))
            box = tuple(float(v) for v in boxes[k]); xt = pil_to_t(im)
            r = {}
            for i, n in enumerate(ens.names):
                coss = []
                for jit_m in (0.0, 0.15):   # tight and loose crops, worst case wins
                    if ens.nets[i][1] == "facenet":
                        v = facenet_view(xt, box, jit_m)
                    else:
                        sa = min(1.0, 224.0 / max(box[2]-box[0], 1.0))
                        xa = F.interpolate(xt[None], scale_factor=sa, mode="bilinear", antialias=True, recompute_scale_factor=False)[0] if sa < 1 else xt
                        v = arc_view(xa, pts[k] * (xa.shape[-1] / xt.shape[-1]))
                    coss.append(float((ens.embed(i, v) * ref[i]).sum()))
                r[n] = max(coss)
                worst[n] = max(worst[n], r[n])
            per_face.append(r)
        res["per_transform"][tname] = per_face
    res["worst_cos"] = worst
    res["worst_cos_all"] = max(worst.values())
    res["identity_match_broken"] = bool(res["worst_cos_all"] <= MATCH_COS_THRESHOLD)
    return res