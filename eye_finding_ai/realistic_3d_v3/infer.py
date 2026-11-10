"""Single-frame full-face eye localization from a trained v3 checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from model import FaceEyeUNet
from preprocess import letterbox_rgb, remap_xy, unletterbox_mask

CLASSES = {"background": 0, "face": 1, "left_sclera": 2, "left_iris": 3, "left_pupil": 4,
           "left_glint": 5, "right_sclera": 6, "right_iris": 7, "right_pupil": 8,
           "right_glint": 9, "left_lid": 10, "right_lid": 11}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); p.add_argument("--image", required=True)
    p.add_argument("--output-dir", required=True); a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.checkpoint, map_location=device, weights_only=False)
    model = FaceEyeUNet(base=ck.get("base", 16), classes=len(CLASSES)).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    image = Image.open(a.image).convert("RGB"); orig_size = image.size; size = int(ck.get("size", 384))
    x, transform = letterbox_rgb(image, size)
    xt = torch.from_numpy(x).permute(2, 0, 1)[None].to(device)
    with torch.inference_mode():
        prob = model(xt).softmax(1)[0].cpu().numpy()
    mask = prob.argmax(0).astype(np.uint8)
    result = {"status": "RESEARCH_INFERENCE_SYNTHETIC_MODEL", "checkpoint": str(a.checkpoint), "device": str(device),
              "input_size_px": [orig_size[0], orig_size[1]], "model_size_px": [size, size], "eyes": {}}
    for side, cls in (("left", 4), ("right", 8)):
        yy, xx = np.where(mask == cls)
        if len(xx) < 5:
            result["eyes"][side] = {"status": "uncertain_or_occluded", "pupil_center_xy_px": None, "pupil_pixels": int(len(xx)), "confidence": None}
        else:
            conf = float(prob[cls, yy, xx].mean())
            center = [round(v, 2) for v in remap_xy(float(xx.mean()), float(yy.mean()), transform)]
            result["eyes"][side] = {"status": "visible_candidate", "pupil_center_xy_px": center, "pupil_pixels": int(len(xx)), "confidence": round(conf, 4)}
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    unletterbox_mask(mask,orig_size,transform).save(out / "semantic_mask.png")
    (out / "observation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
