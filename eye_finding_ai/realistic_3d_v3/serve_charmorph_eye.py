"""Loopback-only research API for the frozen synthetic-domain eye segmenter."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from model import FaceEyeUNet
from train_charmorph_eye import SHARED_EYE_NAMES

MAX_IMAGE_BYTES = 16 * 1024 * 1024


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EyeResearchService:
    def __init__(self, checkpoint: Path, device_name: str | None = None):
        self.checkpoint = checkpoint
        self.digest = file_sha(checkpoint)
        self.device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("classes") != SHARED_EYE_NAMES:
            raise ValueError("checkpoint must be the frozen shared-eye three-class model")
        self.size = int(state["size"])
        self.model = FaceEyeUNet(base=int(state["base"]), classes=len(SHARED_EYE_NAMES)).to(self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()

    @torch.inference_mode()
    def predict(self, payload: bytes) -> dict:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        width, height = image.size
        resized = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        probs = self.model(tensor).softmax(1)[0].cpu().numpy()
        labels = probs.argmax(0)
        eyes = {}
        for side, xs, offset in (("image_left", slice(0, self.size // 2), 0),
                                 ("image_right", slice(self.size // 2, self.size), self.size // 2)):
            yy, local_x = np.where(labels[:, xs] == 2)
            if len(local_x) < 5:
                eyes[side] = {"status": "uncertain_or_not_detected", "pupil_center_xy_px": None,
                              "pupil_pixels": int(len(local_x)), "mean_pupil_probability": None}
            else:
                xx = local_x + offset
                eyes[side] = {"status": "synthetic_domain_candidate",
                              "pupil_center_xy_px": [round(float(xx.mean() * width / self.size), 2),
                                                     round(float(yy.mean() * height / self.size), 2)],
                              "pupil_pixels": int(len(xx)),
                              "mean_pupil_probability": round(float(probs[2, yy, xx].mean()), 5)}
        return {"status": "RESEARCH_ONLY_SYNTHETIC_DOMAIN_INFERENCE",
                "model_sha256": self.digest, "device": str(self.device),
                "input_size_px": [width, height], "model_size_px": [self.size, self.size],
                "eye_side_rule": "image halves; not anatomical laterality", "eyes": eyes,
                "limitations": ["one synthetic base identity/renderer", "not validated on people or cameras",
                                "no calibrated confidence or blink recovery", "not for motor control"]}


def handler_for(service: EyeResearchService):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, value: dict):
            body = json.dumps(value).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                return self._send(404, {"error": "not_found"})
            return self._send(200, {"status": "ready", "model_sha256": service.digest,
                                    "device": str(service.device), "research_only": True})

        def do_POST(self):
            if self.path != "/v1/frame":
                return self._send(404, {"error": "not_found"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._send(400, {"error": "invalid_content_length"})
            if size <= 0 or size > MAX_IMAGE_BYTES:
                return self._send(413, {"error": "image_size_out_of_range", "max_bytes": MAX_IMAGE_BYTES})
            try:
                result = service.predict(self.rfile.read(size))
            except Exception as exc:
                return self._send(400, {"error": "invalid_image", "detail": str(exc)[:200]})
            return self._send(200, result)

        def log_message(self, fmt, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1", help="Loopback default; keep local unless separately secured")
    parser.add_argument("--port", type=int, default=8781)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("This research service only binds to loopback; do not expose it to a network")
    device = None if args.device == "auto" else args.device
    service = EyeResearchService(args.checkpoint, device)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(service))
    print(json.dumps({"status": "ready", "url": f"http://{args.host}:{args.port}",
                      "model_sha256": service.digest, "device": str(service.device),
                      "research_only": True}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
