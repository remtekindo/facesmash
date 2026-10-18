# Cloak engine v2 — measured results (2026-10-18, batch FS2-20261018-A1)

All numbers below are **measured on the final output photo**: the face is re-detected (MTCNN),
re-cropped/aligned like a consumer pipeline, re-embedded, and compared to the original's embedding —
under four upload transforms (PNG, JPEG-75, JPEG-50, 50% downscale + JPEG-75) and two crop margins.
The value reported is the **worst case** (highest cosine similarity). Cosine ≤ 0.30 ⇒ identity match broken.
Reproduce with `prototype/eval_v2/` (needs the model files; test photos are not committed).

## Why v1 looked good but was not
v1 reported the embedding distance of the 160×160 crop **it had just optimised**, on the **same model it
attacked**. Re-measured on the real output photo (8 faces, same harness):

| engine | face SSIM ↑ | VGGFace2 | CASIA | MobileFaceNet | ArcFace-R50 | GlintR100 |
|---|---|---|---|---|---|---|
| **v1 (old, "standard")** | 0.929 | **0.50** (attacked) | 0.58 | 0.67 | 0.73 | 0.75 |
| **v2 low** | 0.987 | 0.19 (attacked) | 0.20 (attacked) | 0.24 (attacked) | **0.37** held-out | **0.40** held-out |
| **v2 standard** | 0.978 | 0.00 (attacked) | 0.03 (attacked) | 0.03 (attacked) | **0.23** held-out | **0.26** held-out |
| **v2 max** | 0.970 | −0.16 (attacked) | −0.12 (attacked) | −0.13 (attacked) | **0.11** held-out | **0.15** held-out |

Mean worst-case cosine over 8 faces; lower = better protection; ≥ 0.30 = still matched. v1 attacked only VGGFace2.
**Held-out** = never used in optimisation (ArcFace-R50 and GlintR100 are different architectures/training sets
from the attacked ensemble; MobileFaceNet shares WebFace600K with R50, so R50 is a *partially* independent check).

Faces (of 8) with **every** model ≤ 0.30 at worst case: **low 0/8 · standard 6/8 · max 8/8**
(v1: 0/8 on every model but one face on VGGFace2).

Mean runtime, 1 face, ≤1280 px, 2 CPU threads: low 22 s · standard 32 s · max 49 s
(service end-to-end incl. verification: 39 s standard on a 659×439 photo; 59 s on a 1665×2048 photo).

## What changed technically (and what each change did)
| change | effect (measured) |
|---|---|
| Ensemble VGGFace2 + CASIA + MobileFaceNet instead of one model | transfer to held-out ArcFace/Glint |
| EOT: crop/landmark jitter, down-up resampling, real JPEG (straight-through) | survives re-detection and upload transforms |
| **DSSIM perceptual penalty** instead of saturating ±8/255 everywhere | biggest win: SSIM 0.89–0.94 → 0.96–0.99 *and* stronger |
| luma-dominant, chroma ×0.12 | chroma ×0.3 turned irises green / tinted noses |
| soft face ellipse mask (no box paste) | no rectangular seam, no pattern on backgrounds |
| verification on the final photo, worst case | the UI verdict can no longer be self-flattering |

Dead ends tried (kept out of the code): high-pass-filtering the perturbation (protection collapsed to ~0.6 cos),
texture-weighted amplitude caps (still produced visible "contour" patterns), larger plain L∞ budgets
(protected but looked clearly retouched).

## Honest limits
1. **Open-source models only.** GlintR100/ArcFace-R50 are strong modern recognisers but **not** the commercial
   systems (Meta, Google, Clearview, PimEyes). Transfer to those is **untested**; do not claim it.
2. **8 faces**, all adults, mostly frontal, mostly light-skinned or studio-lit; no children, masks, profiles,
   low light, group photos. Treat percentages as indicative, not statistics.
3. **Visible at higher strengths.** Max = SSIM 0.970 (p99 pixel change ≈ 26/255); it can look slightly retouched.
4. **Detection still works**; only identity matching is attacked. Purification/denoising or fine-tuning by an
   attacker would weaken it. This is a speed bump, not a vault.
5. Verdict thresholds (cos 0.30) are a convention (ArcFace-style); they were not calibrated on a labelled dataset.
6. Model licences: insightface pretrained models are research/non-commercial; VGGFace2/CASIA have dataset terms.
   Fine for a competition prototype; check before commercialising.
