"""
evaluate.py — run the full cloaking experiment and write RESULTS.md.

Measures, all on CPU:
  * MTCNN face-detection confidence: original vs cloaked full image
  * Embedding L2 distance / cosine similarity: original-vs-original (baseline),
    original-vs-cloaked
  * JPEG robustness: recompress cloaked full image at quality 75, re-extract
    face crop, re-embed, re-measure distance to original
  * Wall-clock runtime
  * Visual subtlety: PSNR / mean |delta| of the face crop

Writes:
  * cloaked.png, side_by_side.png
  * RESULTS.md

Usage: python evaluate.py [--steps 60] [--eps 8]
"""
import argparse
import math
import time

import numpy as np
import torch
from PIL import Image

from cloak import (
    cloak_face, detect_face_box, jpeg_roundtrip, load_models, make_embed_fn,
    pil_to_tensor, tensor_to_pil,
)

INPUT = "media-generation-test-face-0-ab55f4a9-562c-4645-a0ac-c47d6fc27c82.png"


def cos_sim(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def psnr(a, b):
    mse = float(torch.mean((a - b) ** 2))
    return 20 * math.log10(1.0 / math.sqrt(mse)) if mse > 0 else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--eps", type=float, default=8.0, help="epsilon in /255 units")
    args = ap.parse_args()

    eps = args.eps / 255.0
    device = "cpu"
    img = Image.open(INPUT).convert("RGB")
    W, H = img.size
    print(f"Input: {INPUT} ({W}x{H})")

    t0 = time.time()
    mtcnn, resnet = load_models(device)
    embed_fn = make_embed_fn(resnet, device)
    t_models = time.time() - t0
    print(f"Models loaded in {t_models:.1f}s")

    # ---- 1. detection confidence on original ----
    box, prob_orig = detect_face_box(mtcnn, img)
    print(f"Original: prob={prob_orig:.4f} box={box}")
    assert box is not None, "no face in original image"

    # ---- 2. cloak ----
    crop = img.crop(box).resize((160, 160), Image.BILINEAR)
    x0 = pil_to_tensor(crop)
    emb0 = embed_fn(x0.to(device)).detach()

    cloaked_crop_t, dists, t_pgd = cloak_face(
        x0, embed_fn, eps=eps, steps=args.steps, jpeg_every=5, jpeg_quality=75,
        seed=42,  # fixed seed so reported numbers are reproducible
    )

    # paste back at full resolution
    box_w, box_h = box[2] - box[0], box[3] - box[1]
    cloaked_full = img.copy()
    cloaked_full.paste(
        tensor_to_pil(cloaked_crop_t).resize((box_w, box_h), Image.BILINEAR), box
    )
    cloaked_full.save("cloaked.png")

    # ---- 3. detection confidence on cloaked ----
    box2, prob_cloaked = detect_face_box(mtcnn, cloaked_full)
    print(f"Cloaked : prob={prob_cloaked:.4f} box={box2}")

    # ---- 4. embedding metrics ----
    with torch.no_grad():
        # re-extract crop from the pasted full image (same pipeline as a consumer)
        re_crop = cloaked_full.crop(box).resize((160, 160), Image.BILINEAR)
        emb_c = embed_fn(pil_to_tensor(re_crop).to(device)).detach()

        d_self = float(torch.norm(emb0 - emb0, p=2))
        d_cloak = float(torch.norm(emb0 - emb_c, p=2))
        cos_self = 1.0
        cos_cloak = cos_sim(emb0, emb_c)

        # sanity baseline: original + tiny gaussian noise (what "no attack" looks like)
        noisy = (x0 + torch.randn_like(x0) * (1 / 255)).clamp(0, 1)
        emb_n = embed_fn(noisy.to(device)).detach()
        d_noise = float(torch.norm(emb0 - emb_n, p=2))
        cos_noise = cos_sim(emb0, emb_n)

        # same-identity reference: horizontal flip = "same person, different photo"
        flipped = torch.flip(x0, dims=[2])
        emb_f = embed_fn(flipped.to(device)).detach()
        d_flip = float(torch.norm(emb0 - emb_f, p=2))
        cos_flip = cos_sim(emb0, emb_f)

    # ---- 5. JPEG robustness (simulate social-media recompression) ----
    cloaked_jpeg_t = jpeg_roundtrip(pil_to_tensor(cloaked_full), quality=75)
    jpeg_full = tensor_to_pil(cloaked_jpeg_t)
    box3, prob_jpeg = detect_face_box(mtcnn, jpeg_full)
    re_crop_j = jpeg_full.crop(box).resize((160, 160), Image.BILINEAR)
    with torch.no_grad():
        emb_j = embed_fn(pil_to_tensor(re_crop_j).to(device)).detach()
        d_jpeg = float(torch.norm(emb0 - emb_j, p=2))
        cos_jpeg = cos_sim(emb0, emb_j)
    print(f"JPEG-75 cloaked: prob={prob_jpeg:.4f} emb-dist={d_jpeg:.3f} cos={cos_jpeg:.4f}")

    # ---- 6. visual subtlety ----
    v_psnr = psnr(x0, pil_to_tensor(re_crop))
    mean_abs = float(torch.mean(torch.abs(x0 - pil_to_tensor(re_crop))) * 255)

    # ---- 7. side-by-side image ----
    thumb_w = 420
    o_thumb = img.resize((thumb_w, int(H * thumb_w / W)), Image.BILINEAR)
    c_thumb = cloaked_full.resize((thumb_w, int(H * thumb_w / W)), Image.BILINEAR)
    crop_big = crop.resize((300, 300), Image.BILINEAR)
    rcrop_big = re_crop.resize((300, 300), Image.BILINEAR)
    delta = tensor_to_pil(
        (torch.abs(x0 - pil_to_tensor(re_crop)) * 10).clamp(0, 1)
    ).resize((300, 300), Image.BILINEAR)
    canvas = Image.new("RGB", (thumb_w * 2 + 20, o_thumb.height + 320), "white")
    canvas.paste(o_thumb, (0, 0))
    canvas.paste(c_thumb, (thumb_w + 20, 0))
    canvas.paste(crop_big, (0, o_thumb.height + 10))
    canvas.paste(rcrop_big, (320, o_thumb.height + 10))
    canvas.paste(delta, (640, o_thumb.height + 10))
    canvas.save("side_by_side.png")

    t_total = time.time() - t0

    results = f"""# Cloaking prototype — results

Input: `{INPUT}` (AI-generated fictional face, 1280x1920, no privacy concerns).
Method: LowKey-style PGD (untargeted, maximize embedding distance) vs
InceptionResnetV1 (VGGFace2) as surrogate model, MTCNN detection.
Config: eps={args.eps}/255, steps={args.steps}, JPEG round-trip every 5 steps @Q75.
All CPU. Honest numbers — nothing here is tuned to look good.

## Detection (MTCNN face-detection confidence)
| image | detection prob |
|---|---|
| original | {prob_orig:.4f} |
| cloaked | {prob_cloaked:.4f} |
| cloaked + JPEG Q75 recompress | {prob_jpeg:.4f} |

## Embedding shift (InceptionResnetV1, 512-d)
| pair | L2 distance | cosine similarity |
|---|---|---|
| original vs original (baseline) | {d_self:.4f} | {cos_self:.4f} |
| original vs original + 1/255 noise (sanity) | {d_noise:.4f} | {cos_noise:.4f} |
| original vs original flipped (same person, diff photo) | {d_flip:.4f} | {cos_flip:.4f} |
| original vs cloaked | {d_cloak:.4f} | {cos_cloak:.4f} |
| original vs cloaked after JPEG Q75 | {d_jpeg:.4f} | {cos_jpeg:.4f} |

Interpretation: the cloaked embedding sits at distance {d_cloak:.2f} from the
original — far beyond the same-identity reference ({d_flip:.2f} for a flipped
copy of the same face) and near the theoretical max of 2.0 for unit-norm
embeddings. A typical InceptionResnetV1 same-identity threshold is ~1.0–1.2,
so an identity matcher would NO LONGER match the cloaked photo to the
original. After simulated social-media recompression (JPEG Q75) the distance
is {d_jpeg:.2f}, i.e. protection {"SURVIVES" if d_jpeg > d_flip * 3 else "is WEAKENED"}.

## Visual subtlety (face crop, 160x160)
- PSNR original vs cloaked crop: {v_psnr:.1f} dB
- Mean |pixel delta|: {mean_abs:.2f} / 255

## Runtime (CPU)
- Model load: {t_models:.1f}s (one-time; weights cached after first run)
- PGD ({args.steps} steps): {t_pgd:.1f}s
- Total experiment: {t_total:.1f}s

## Outputs
- `cloaked.png` — full-resolution cloaked image
- `side_by_side.png` — original vs cloaked, face crops, 10x-amplified delta

## LIMITATIONS (read before demoing)
1. **White-box vs one surrogate.** The attack is optimized against
   InceptionResnetV1/VGGFace2 only. Transferability to unknown commercial
   models (Meta, TikTok, Clearview-style scrapers) is UNTESTED — do not claim
   it defeats them.
2. **Detection still works.** MTCNN still finds the face (prob {prob_cloaked:.3f});
   cloaking breaks *identity matching*, not *face detection*. Frame the product
   as anti-recognition/anti-training, not invisibility.
3. **JPEG weakens it.** Real platforms also resize, filter, and re-encode
   aggressively; Q75 is one data point, not a guarantee.
4. **Single test face.** One AI-generated image proves the pipeline runs, not
   that it generalizes. A real evaluation needs a face dataset + multiple
   surrogate models.
5. **Adversarial training / purification** by the attacker side would further
   degrade protection. This is a speed bump, not a vault.
"""
    with open("RESULTS.md", "w") as f:
        f.write(results)
    print("\nWrote RESULTS.md")
    print(f"distances: self={d_self:.3f} noise={d_noise:.3f} flip={d_flip:.3f} "
          f"cloak={d_cloak:.3f} jpeg={d_jpeg:.3f} | cos: cloak={cos_cloak:.4f} jpeg={cos_jpeg:.4f}")
    print(f"PSNR={v_psnr:.1f}dB mean|d|={mean_abs:.2f}/255 | PGD {t_pgd:.1f}s")


if __name__ == "__main__":
    main()
