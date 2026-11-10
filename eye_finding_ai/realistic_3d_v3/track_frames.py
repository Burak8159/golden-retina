"""Run per-eye localization plus a short-gap temporal filter over ordered frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from model import FaceEyeUNet
from preprocess import letterbox_rgb, remap_xy
from sequence_filter import AlphaBeta


def main():
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); p.add_argument("--frames-dir", required=True)
    p.add_argument("--output", required=True); p.add_argument("--glob", default="*.png"); p.add_argument("--minimum-confidence", type=float, default=.35)
    a = p.parse_args(); files = sorted(Path(a.frames_dir).glob(a.glob))
    if not files: raise SystemExit(f"No frames matched {a.glob!r} in {a.frames_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.checkpoint, map_location=device, weights_only=False)
    model = FaceEyeUNet(base=ck.get("base", 16)).to(device); model.load_state_dict(ck["model"]); model.eval()
    filters = {s: AlphaBeta() for s in ("left", "right")}; rows = []
    with torch.inference_mode():
        for path in files:
            image = Image.open(path).convert("RGB"); w, h = image.size; size = int(ck.get("size", 384))
            x, transform = letterbox_rgb(image, size)
            probs = model(torch.from_numpy(x).permute(2, 0, 1)[None].to(device)).softmax(1)[0].cpu().numpy()
            pred = probs.argmax(0); observations = {}
            for side, cls in (("left", 4), ("right", 8)):
                yy, xx = np.where(pred == cls)
                conf = float(probs[cls, yy, xx].mean()) if len(xx) >= 5 else 0.
                raw = remap_xy(float(xx.mean()),float(yy.mean()),transform) if len(xx) >= 5 and conf >= a.minimum_confidence else None
                tracked = filters[side].update(raw)
                observations[side] = {"detection_xy_px": [round(v, 2) for v in raw] if raw else None,
                                      "confidence": round(conf, 4) if raw else None,
                                      "track_xy_px": [round(v, 2) for v in tracked["xy"]] if tracked["xy"] else None,
                                      "state": tracked["state"]}
            rows.append({"frame": path.name, "eyes": observations})
    out = {"status": "RESEARCH_SEQUENCE_TRACKING_SYNTHETIC_OR_USER_SUPPLIED_FRAMES", "frames": len(rows),
           "checkpoint": str(a.checkpoint), "minimum_confidence": a.minimum_confidence, "results": rows}
    dest = Path(a.output); dest.parent.mkdir(parents=True, exist_ok=True); dest.write_text(json.dumps(out, indent=2)+"\n")
    print(json.dumps({"status": out["status"], "frames": len(rows), "output": str(dest),
                      "left_detected_frames": sum(r["eyes"]["left"]["state"] == "detected" for r in rows),
                      "right_detected_frames": sum(r["eyes"]["right"]["state"] == "detected" for r in rows)}, indent=2))


if __name__ == "__main__": main()
