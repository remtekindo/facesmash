"""
cloak_advanced.py — Advanced Robust Cloak (Gemini-inspired).
MI-FGSM + Adaptive Perceptual Masking on top of cloak_v2.
API matches cloak_v2.cloak_face().
"""
from __future__ import annotations
import math, time
import numpy as np
import torch
import torch.nn.functional as F
from cloak_v2 import (Cfg, Ensemble, FaceJob, FaceResult, _gauss_blur,
    _jitter_pts, _ssim_map, _to_rgb_delta, _umeyama, _views,
    arc_view, face_mask, facenet_view, jpeg_ste, pil_to_t, t_to_pil)


def _perceptual_mask(x0: torch.Tensor) -> torch.Tensor:
    gray = 0.2989 * x0[0:1] + 0.5870 * x0[1:2] + 0.1140 * x0[2:3]
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=x0.dtype, device=x0.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=x0.dtype, device=x0.device).view(1,1,3,3)
    gx = F.conv2d(gray.unsqueeze(0), kx, padding=1).squeeze(0)
    gy = F.conv2d(gray.unsqueeze(0), ky, padding=1).squeeze(0)
    edges = torch.sqrt(gx**2 + gy**2 + 1e-6)
    norm = edges / (edges.max() + 1e-5)
    return torch.clamp(norm * 1.2 + 0.4, 0.4, 1.6)  # (1,h,w)


def cloak_face_advanced(x_native, job, ens, cfg, verbose=False, momentum_decay=1.0):
    t0 = time.time()
    rng = np.random.default_rng(cfg.seed)
    if cfg.seed is not None: torch.manual_seed(cfg.seed)
    _, H, W = x_native.shape
    bx1, by1, bx2, by2 = job.box
    bw, bh = bx2 - bx1, by2 - by1
    m = cfg.region_margin
    rx1, ry1 = max(0, int(bx1 - m*bw)), max(0, int(by1 - m*bh))
    rx2, ry2 = min(W, int(bx2 + m*bw)+1), min(H, int(by2 + m*bh)+1)
    region = x_native[:, ry1:ry2, rx1:rx2]
    rh, rw = region.shape[-2:]
    s = min(1.0, cfg.work_face / max(bw, 1.0))
    if s < 1.0:
        wh, ww = max(8, round(rh*s)), max(8, round(rw*s))
        x0 = F.interpolate(region[None], size=(wh,ww), mode="bilinear", antialias=True, align_corners=False)[0]
    else:
        x0, wh, ww = region, rh, rw
    sx, sy = ww/rw, wh/rh
    box_w = ((bx1-rx1)*sx, (by1-ry1)*sy, (bx2-rx1)*sx, (by2-ry1)*sy)
    pts_w = (job.pts - np.array([rx1,ry1], dtype=np.float32)) * np.array([sx,sy], dtype=np.float32)
    base_mask = face_mask(x0, box_w, cfg)
    eff_mask = base_mask * _perceptual_mask(x0)
    with torch.no_grad():
        refs = {i: [] for i in ens.train_idx}
        for k in range(6):
            for i, v in _views(ens, x0, box_w, pts_w, rng, jitter=k>0, idx=ens.train_idx).items():
                refs[i].append(ens.embed(i, v))
        e0 = {i: F.normalize(torch.cat(r).mean(0, keepdim=True), dim=1) for i, r in refs.items()}
    x = x0.clone().detach()
    g = torch.zeros_like(x)
    train = ens.train_idx
    last_cos = {i: 1.0 for i in train}
    alpha = cfg.eps / cfg.steps * 2.0
    face_w = base_mask[0] + 1e-6
    for step in range(cfg.steps):
        x = x.detach().requires_grad_(True)
        xa = x
        if cfg.eot:
            if rng.random() < 0.5:
                r = float(rng.uniform(cfg.eot_min_scale, 0.95))
                small = F.interpolate(xa[None], scale_factor=r, mode="bilinear", antialias=True, recompute_scale_factor=False)
                xa = F.interpolate(small, size=(wh,ww), mode="bilinear", align_corners=False)[0]
            if rng.random() < 0.4:
                xa = jpeg_ste(xa, int(rng.integers(50, 91)))
        idx = list(train)
        if cfg.models_per_step and cfg.models_per_step < len(train):
            idx = sorted(rng.choice(train, cfg.models_per_step, replace=False).tolist())
        views = _views(ens, xa, box_w, pts_w, rng, jitter=cfg.eot, idx=idx)
        loss_adv = 0.0
        for i in idx:
            cos = (ens.embed(i, views[i]) * e0[i]).sum()
            last_cos[i] = float(cos.detach())
            loss_adv = loss_adv + torch.relu(cos - cfg.target_cos)
        sm = _ssim_map(x0, (x0 + (x - x0)).clamp(0, 1))
        dssim = (1 - (sm * face_w).sum() / face_w.sum()) / 2
        loss = loss_adv / len(idx) + cfg.ssim_w * dssim
        loss.backward()
        with torch.no_grad():
            grad = x.grad / (x.grad.abs().mean(dim=[1,2,3], keepdim=True) + 1e-12)
            g = momentum_decay * g + grad
            x = x + (alpha * eff_mask) * g.sign()
            cap = cfg.eps * eff_mask
            x = torch.max(torch.min(x, x0 + cap), x0 - cap).clamp(0, 1)
        if verbose and (step+1) % 20 == 0:
            print(f"  step {step+1}/{cfg.steps} cos={[round(c,3) for c in last_cos.values()]}")
    with torch.no_grad():
        delta = (x.detach() - x0).clamp(-cfg.eps*1.6, cfg.eps*1.6)
        delta = _to_rgb_delta(torch.tanh(delta/(cfg.eps+1e-8)*2.0), cfg.chroma) * (cfg.eps * eff_mask)
        delta_n = F.interpolate(delta[None], size=(rh,rw), mode="bicubic", align_corners=False)[0] if (wh,ww)!=(rh,rw) else delta
    return delta_n, (rx1,ry1,rx2,ry2), FaceResult(job.box, cfg.steps, time.time()-t0, last_cos)


def cloak_image(img, jobs, ens, cfg, verbose=False):
    x = pil_to_t(img); results = []
    for k, job in enumerate(jobs):
        c = Cfg(**{**cfg.__dict__, "seed": None if cfg.seed is None else cfg.seed+k})
        delta, (rx1,ry1,rx2,ry2), fr = cloak_face_advanced(x, job, ens, c, verbose)
        x[:, ry1:ry2, rx1:rx2] = (x[:, ry1:ry2, rx1:rx2] + delta).clamp(0, 1)
        results.append(fr)
    return t_to_pil(x), results
