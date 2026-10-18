"""Download ArcFace-family ONNX weights (insightface model zoo, release v0.7) into /app/models.

Kept as a script (not a Dockerfile heredoc) so it builds with the classic Docker builder too.
  mbf (MobileFaceNet) is REQUIRED: it is part of the attacked ensemble -> exit 1 loudly if it fails.
  r50 (ResNet50) is the independent hold-out check: optional -> warn and continue if it fails.
Research / non-commercial licence (insightface pretrained models).
"""
import io
import sys
import time
import urllib.request
import zipfile

BASE = "https://github.com/deepinsight/insightface/releases/download/v0.7/"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/app/models"
ITEMS = (("buffalo_s.zip", "w600k_mbf.onnx", True), ("buffalo_l.zip", "w600k_r50.onnx", False))


def fetch(zip_name: str, member: str) -> None:
    last = None
    for attempt in range(1, 4):
        try:
            data = urllib.request.urlopen(BASE + zip_name, timeout=300).read()
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                with open(f"{OUT}/{member}", "wb") as f:
                    f.write(z.read(member))
            print(f"fetched {member} ({len(data) // 1_000_000} MB zip)", flush=True)
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"attempt {attempt} for {zip_name} failed: {exc!r}", flush=True)
            time.sleep(5 * attempt)
    raise RuntimeError(f"could not fetch {member}: {last!r}")


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    for zip_name, member, required in ITEMS:
        try:
            fetch(zip_name, member)
        except Exception as exc:  # noqa: BLE001
            if required:
                print(f"FATAL: {exc}", flush=True)
                sys.exit(1)
            print(f"WARNING (optional hold-out model skipped): {exc}", flush=True)
