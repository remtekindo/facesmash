# Usage: FACESMASH_MODELS=models FACESMASH_FACES=faces python exp.py --name run1 --strength standard --faces a,b,c
import sys, json, time, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from PIL import Image
torch.set_num_threads(2)
FACES = os.environ.get('FACESMASH_FACES', 'faces')   # dir with test photos (not committed)
import cloak_v2 as C, evallib
from facenet_pytorch import MTCNN

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--models", default="vggface2,casia,mbf")
ap.add_argument("--strength", default="standard")
ap.add_argument("--faces", default="user_selfie,user_portrait,astronaut,kate_siegel,angelina_jolie")
ap.add_argument("--cfg", default="{}")
a = ap.parse_args()
kw = json.loads(a.cfg)
if "eps255" in kw: kw["eps"] = kw.pop("eps255")/255
if "ellipse" in kw: kw["ellipse"] = tuple(kw["ellipse"])
mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, device="cpu")
ens = C.Ensemble(tuple(a.models.split(",")), onnx_paths={"arcface":f"{evallib.MODELS}/w600k_r50.onnx","mbf":f"{evallib.MODELS}/heldout/w600k_mbf.onnx"})
judge = evallib.Judge()
od = f"out_{a.name}"; os.makedirs(od, exist_ok=True)

def worst_cos(r):
    w = {}
    for tf, d in r["tf"].items():
        if d is None: continue
        for k, v in d.items():
            c = 1 - v*v/2 if k in ("vgg","casia") else v
            w[k.replace("_cos","")] = max(w.get(k.replace("_cos",""), -9), c)
    return w

rows = []
for f in a.faces.split(","):
    p = [x for x in os.listdir(FACES) if x.startswith(f)][0]
    img = Image.open(FACES+"/"+p).convert("RGB"); img.thumbnail((1280,1280), Image.LANCZOS)
    jobs = C.detect_faces(mtcnn, img, max_faces=1)
    cfg = C.cfg_for(a.strength, seed=7, **kw)
    t = time.time(); out, frs = C.cloak_image(img, jobs, ens, cfg); el = time.time()-t
    out.save(f"{od}/{f}.png")
    r = evallib.evaluate(judge, img, out)
    wc = worst_cos(r); v = r["visual"]
    rows.append((f, el, v, wc))
    print(f"{f:16s} {el:5.1f}s ssim={v['ssim_face']:.3f} psnr={v['psnr_face']:.1f} |d|={v['mean_abs_face']:.2f} p99={v['p99_abs_face']:.0f}  worst-cos " +
          " ".join(f"{k}={wc[k]:+.2f}" for k in ("vgg","casia","arc","glint","mbf")), flush=True)
m = lambda k: np.mean([r[3][k] for r in rows])
print(f"MEAN[{a.name}] ssim={np.mean([r[2]['ssim_face'] for r in rows]):.3f} psnr={np.mean([r[2]['psnr_face'] for r in rows]):.1f} t={np.mean([r[1] for r in rows]):.1f}s  " +
      " ".join(f"{k}={m(k):+.2f}" for k in ("vgg","casia","arc","glint","mbf")))
