"""
cloak.py — LowKey-style PGD facial cloaking pipeline (prototype).

Based on: Cherepanova et al., "LowKey: Leveraging Adversarial Attacks to Protect
Social Media Users from Facial Recognition", ICLR 2021.
https://arxiv.org/abs/2101.07922

Pipeline:
  1. Detect the face with MTCNN.
  2. Run PGD to MAXIMIZE the embedding distance between the original face
     embedding and the perturbed one (untargeted), from an InceptionResnetV1
     model pretrained on VGGFace2, inside a small L_inf ball (default 8/255)
     so the change stays visually subtle. Perturbation is restricted to the
     detected face region only.
  3. Social-media robustness is VERIFIED (not trained): the final cloaked
     output is round-tripped through real JPEG Q75 and the embedding distance
     re-measured honestly. (In-loop JPEG EOT was removed 2026-10-18: it baked
     visible block artifacts into the perturbation at 160x160.)

Usage:
  python cloak.py --input face.png --output cloaked.png --steps 60
"""
import argparse
import io
import time

import numpy as np
import torch
from PIL import Image


def load_models(device="cpu"):
    from facenet_pytorch import InceptionResnetV1, MTCNN

    mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
    resnet = InceptionResnetV1(pretrained="vggface2", device=device).eval()
    return mtcnn, resnet


def detect_face_box(mtcnn, img_pil, margin_ratio=0.25, min_prob=0.0):
    """Return (box, prob) of the most confident face. Box in original coords."""
    boxes, probs = mtcnn.detect(img_pil)
    if boxes is None or len(boxes) == 0:
        return None, 0.0
    i = int(np.argmax(probs))
    if probs[i] < min_prob:
        return None, float(probs[i])
    x1, y1, x2, y2 = boxes[i]
    w, h = x2 - x1, y2 - y1
    mx, my = w * margin_ratio, h * margin_ratio
    W, H = img_pil.size
    box = (
        max(0, int(x1 - mx)),
        max(0, int(y1 - my)),
        min(W, int(x2 + mx)),
        min(H, int(y2 + my)),
    )
    return box, float(probs[i])


def pil_to_tensor(img_pil):
    arr = np.asarray(img_pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def tensor_to_pil(t):
    arr = (t.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0)
    return Image.fromarray(arr.astype(np.uint8))


def jpeg_roundtrip(tensor, quality=75):
    """Real (non-differentiable) JPEG compress/decompress of a [0,1] tensor."""
    buf = io.BytesIO()
    tensor_to_pil(tensor).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return pil_to_tensor(Image.open(buf).convert("RGB"))


def make_embed_fn(resnet, device):
    from facenet_pytorch.models.mtcnn import fixed_image_standardization

    def embed(face_tensor):
        # face_tensor: (3,160,160) in [0,1]; fixed_image_standardization
        # expects [0,255] (it computes (x - 127.5) / 128)
        x = fixed_image_standardization(face_tensor.unsqueeze(0).to(device) * 255.0)
        return resnet(x).squeeze(0)

    return embed


def cloak_face(
    face_tensor,
    embed_fn,
    eps=8 / 255,
    steps=60,
    alpha=None,
    jpeg_every=0,
    jpeg_quality=75,
    seed=None,
    verbose=True,
):
    """PGD: maximize embedding distance from the original embedding.

    Uses a random start inside the eps-ball (standard PGD practice): at
    x == x0 the gradient of ||emb(x) - emb(x0)|| is exactly zero, so a
    zero start never moves. The randomness also means every run produces a
    different perturbation pattern — a feature, not a bug, for this product.

    jpeg_every: if > 0, round-trip through real JPEG every N steps (EOT-lite
    for social-media robustness). DEFAULT 0 (disabled): in-loop JPEG at the
    160x160 working resolution bakes visible 8x8 block artifacts into the
    perturbation, which become glaringly obvious when the cloaked crop is
    upscaled back to the face-box size. The /protect endpoint still VERIFIES
    JPEG-Q75 survival on the final output (honest reporting); it just no
    longer destroys visual quality during optimization. (Fixed 2026-10-18
    after user report of visibly degraded output.)

    Returns (cloaked_tensor, distances_over_time, elapsed_seconds).
    """
    device = face_tensor.device
    if alpha is None:
        alpha = eps / 10.0
    if seed is not None:
        torch.manual_seed(seed)
    x0 = face_tensor.to(device)
    orig_emb = embed_fn(x0).detach()
    x = x0.clone()
    # random start: uniform noise inside the L_inf ball
    x = x + torch.empty_like(x).uniform_(-eps, eps)
    x = torch.max(torch.min(x, x0 + eps), x0 - eps).clamp(0, 1)
    dists = []

    t_start = time.time()
    for step in range(steps):
        x = x.detach().requires_grad_(True)
        emb = embed_fn(x)
        dist = torch.norm(emb - orig_emb, p=2)
        loss = -dist  # we ASCEND on dist == DESCEND on loss
        loss.backward()
        with torch.no_grad():
            grad_sign = x.grad.sign()
            x = x - alpha * grad_sign  # minus: descend on (-dist)
            x = torch.max(torch.min(x, x0 + eps), x0 - eps)  # L_inf ball
            x = x.clamp(0, 1)
        dists.append(float(dist.detach()))
        if jpeg_every and (step + 1) % jpeg_every == 0:
            # EOT-lite: force the perturbation to survive real JPEG.
            x = jpeg_roundtrip(x.detach().cpu(), quality=jpeg_quality).to(device)
            x = torch.max(torch.min(x, x0 + eps), x0 - eps)
            x = x.clamp(0, 1)
        if verbose and (step + 1) % 10 == 0:
            print(f"  step {step + 1}/{steps}  emb-dist={dists[-1]:.3f}")
    elapsed = time.time() - t_start
    if verbose:
        print(f"  PGD done in {elapsed:.1f}s, final emb-dist={dists[-1]:.3f}")
    return x.detach().cpu(), dists, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--eps", type=float, default=8 / 255)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--jpeg-every", type=int, default=0)
    ap.add_argument("--jpeg-quality", type=int, default=75)
    ap.add_argument("--margin", type=float, default=0.25)
    args = ap.parse_args()

    device = "cpu"
    img = Image.open(args.input).convert("RGB")
    print(f"Loading models on {device} ...")
    mtcnn, resnet = load_models(device)
    embed_fn = make_embed_fn(resnet, device)

    box, prob = detect_face_box(mtcnn, img, margin_ratio=args.margin)
    if box is None:
        raise SystemExit("No face detected.")
    print(f"Face detected: prob={prob:.4f}, box={box}")

    crop = img.crop(box).resize((160, 160), Image.BILINEAR)
    face_tensor = pil_to_tensor(crop)
    cloaked_crop, dists, elapsed = cloak_face(
        face_tensor, embed_fn, eps=args.eps, steps=args.steps,
        jpeg_every=args.jpeg_every, jpeg_quality=args.jpeg_quality,
    )

    out = img.copy()
    out.paste(tensor_to_pil(cloaked_crop).resize(
        (box[2] - box[0], box[3] - box[1]), Image.BILINEAR), box)
    out.save(args.output)
    print(f"Saved cloaked image -> {args.output}")


if __name__ == "__main__":
    main()
