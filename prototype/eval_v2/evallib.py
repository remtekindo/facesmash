"""Honest evaluator (research harness, not shipped in the Docker image): judges the FINAL image the way a consumer pipeline would.
Re-detects the face on the output, aligns, embeds with 3 independent models
(VGGFace2, CASIA-WebFace [facenet-pytorch], ArcFace-R50 [insightface, onnx]),
under several upload transforms."""
import io, os, sys
import numpy as np, torch
from PIL import Image
from skimage.metrics import structural_similarity
from skimage.transform import SimilarityTransform, warp

MODELS = os.environ.get("FACESMASH_MODELS", "models")   # dir with w600k_r50.onnx, heldout/w600k_mbf.onnx, heldout/antelopev2/glintr100.onnx
os.environ.setdefault("TORCH_HOME", os.environ.get("TORCH_HOME", ".torch_cache"))
from facenet_pytorch import InceptionResnetV1, MTCNN
import onnxruntime as ort

ARC_TPL = np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],[41.5493,92.3655],[70.7299,92.2041]],dtype=np.float32)

class Judge:
    def __init__(self):
        self.mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device="cpu")
        self.v2 = InceptionResnetV1(pretrained="vggface2").eval()
        self.cw = InceptionResnetV1(pretrained="casia-webface").eval()
        so = ort.SessionOptions(); so.intra_op_num_threads = 2
        mk = lambda p: ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
        self.arcs = {"arc": mk(f"{MODELS}/w600k_r50.onnx"),
                     "glint": mk(f"{MODELS}/heldout/antelopev2/glintr100.onnx"),
                     "mbf": mk(f"{MODELS}/heldout/w600k_mbf.onnx")}

    def _facenet_in(self, img, box, margin):
        x1,y1,x2,y2 = box; w,h = x2-x1, y2-y1
        m = margin*max(w,h)
        c = img.crop((int(x1-m),int(y1-m),int(x2+m),int(y2+m))).resize((160,160), Image.BILINEAR)
        a = (np.asarray(c).astype(np.float32)-127.5)/128.0
        return torch.from_numpy(a).permute(2,0,1).unsqueeze(0)

    def _arc_in(self, img, pts):
        tf = SimilarityTransform.from_estimate(pts.astype(np.float32), ARC_TPL)
        a = warp(np.asarray(img).astype(np.float32), tf.inverse, output_shape=(112,112), order=1, preserve_range=True)
        a = (a-127.5)/127.5
        return a.transpose(2,0,1)[None].astype(np.float32)

    @torch.no_grad()
    def embed(self, img):
        """Returns dict of embeddings for variants, or None if no face."""
        boxes, probs, pts = self.mtcnn.detect(img, landmarks=True)
        if boxes is None: return None
        i = int(np.argmax(probs)); box = boxes[i]
        out = {"prob": float(probs[i]), "box": box}
        for tag, margin in (("m0",0.0),("m15",0.15)):
            x = self._facenet_in(img, box, margin)
            out[f"vgg_{tag}"] = self.v2(x)[0].numpy()
            out[f"casia_{tag}"] = self.cw(x)[0].numpy()
        blob = self._arc_in(img, pts[i])
        for k, sess in self.arcs.items():
            a = sess.run(None, {sess.get_inputs()[0].name: blob})[0][0]
            out[k] = a/np.linalg.norm(a)
        return out

def jpeg(img, q):
    b = io.BytesIO(); img.save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")

TRANSFORMS = {
    "png":   lambda im: im,
    "jpeg75": lambda im: jpeg(im, 75),
    "jpeg50": lambda im: jpeg(im, 50),
    "down50+jpeg75": lambda im: jpeg(im.resize((max(1,im.width//2), max(1,im.height//2)), Image.LANCZOS), 75),
}

def distances(e0, e1):
    """L2 for facenet models (worst case = min over crop variants), cosine-sim for arcface."""
    r = {}
    for m in ("vgg","casia"):
        ds = [float(np.linalg.norm(e0[f"{m}_{t}"]-e1[f"{m}_{t}"])) for t in ("m0","m15")]
        r[m] = min(ds)           # worst case for the protector
    for k in ("arc","glint","mbf"):
        r[k+"_cos"] = float(np.dot(e0[k], e1[k]))
    return r

def visual(orig, out, box):
    a = np.asarray(orig).astype(np.float64); b = np.asarray(out).astype(np.float64)
    x1,y1,x2,y2 = [int(v) for v in box]
    fa, fb = a[y1:y2,x1:x2], b[y1:y2,x1:x2]
    def psnr(p,q):
        mse = np.mean((p-q)**2); return 99.0 if mse==0 else 10*np.log10(255**2/mse)
    ssim = structural_similarity(fa.astype(np.uint8), fb.astype(np.uint8), channel_axis=2, data_range=255)
    return {"psnr_face": round(psnr(fa,fb),1), "psnr_full": round(psnr(a,b),1),
            "ssim_face": round(float(ssim),4), "mean_abs_face": round(float(np.mean(np.abs(fa-fb))),2),
            "p99_abs_face": round(float(np.percentile(np.abs(fa-fb),99)),1)}

def evaluate(judge, orig, out, e0=None):
    e0 = e0 or judge.embed(orig)
    res = {"visual": visual(orig, out, e0["box"]), "tf": {}}
    for name, fn in TRANSFORMS.items():
        e1 = judge.embed(fn(out))
        res["tf"][name] = distances(e0, e1) if e1 else None
    return res
