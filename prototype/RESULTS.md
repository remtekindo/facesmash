# Cloaking prototype — results

Input: `media-generation-test-face-0-ab55f4a9-562c-4645-a0ac-c47d6fc27c82.png` (AI-generated fictional face, 1280x1920, no privacy concerns).
Method: LowKey-style PGD (untargeted, maximize embedding distance) vs
InceptionResnetV1 (VGGFace2) as surrogate model, MTCNN detection.
Config: eps=8.0/255, steps=60, JPEG round-trip every 5 steps @Q75.
All CPU. Honest numbers — nothing here is tuned to look good.

## Detection (MTCNN face-detection confidence)
| image | detection prob |
|---|---|
| original | 1.0000 |
| cloaked | 0.9997 |
| cloaked + JPEG Q75 recompress | 0.9997 |

## Embedding shift (InceptionResnetV1, 512-d)
| pair | L2 distance | cosine similarity |
|---|---|---|
| original vs original (baseline) | 0.0000 | 1.0000 |
| original vs original + 1/255 noise (sanity) | 0.0182 | 0.9998 |
| original vs original flipped (same person, diff photo) | 0.1997 | 0.9801 |
| original vs cloaked | 1.5184 | -0.1527 |
| original vs cloaked after JPEG Q75 | 1.5149 | -0.1474 |

Interpretation: the cloaked embedding sits at distance 1.52 from the
original — far beyond the same-identity reference (0.20 for a flipped
copy of the same face) and near the theoretical max of 2.0 for unit-norm
embeddings. A typical InceptionResnetV1 same-identity threshold is ~1.0–1.2,
so an identity matcher would NO LONGER match the cloaked photo to the
original. After simulated social-media recompression (JPEG Q75) the distance
is 1.51, i.e. protection SURVIVES.

## Visual subtlety (face crop, 160x160)
- PSNR original vs cloaked crop: 32.7 dB
- Mean |pixel delta|: 4.79 / 255

## Runtime (CPU)
- Model load: 2.3s (one-time; weights cached after first run)
- PGD (60 steps): 15.7s
- Total experiment: 22.5s

## Outputs
- `cloaked.png` — full-resolution cloaked image
- `side_by_side.png` — original vs cloaked, face crops, 10x-amplified delta

## LIMITATIONS (read before demoing)
1. **White-box vs one surrogate.** The attack is optimized against
   InceptionResnetV1/VGGFace2 only. Transferability to unknown commercial
   models (Meta, TikTok, Clearview-style scrapers) is UNTESTED — do not claim
   it defeats them.
2. **Detection still works.** MTCNN still finds the face (prob 0.9997);
   cloaking breaks *identity matching*, not *face detection*. Frame the product
   as anti-recognition/anti-training, not invisibility.
3. **JPEG Q75 barely dents it (1.515 vs 1.518), but that's one data point.**
   Real platforms also resize, crop, filter, and re-encode in chains;
   robustness against the full Instagram/Facebook pipeline is untested.
4. **Single test face.** One AI-generated image proves the pipeline runs, not
   that it generalizes. A real evaluation needs a face dataset + multiple
   surrogate models.
5. **Adversarial training / purification** by the attacker side would further
   degrade protection. This is a speed bump, not a vault.
