"""Research-only inference for the frozen shared-eye CharMorph candidate.

The model was trained on square synthetic renders and returns shared iris and
pupil classes. Eye side is inferred from image half. This utility is not a
validated real-camera tracker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from model import FaceEyeUNet
from train_charmorph_eye import SHARED_EYE_NAMES


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--minimum-pupil-pixels", type=int, default=5)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if ck.get("classes") != SHARED_EYE_NAMES:
        raise ValueError("checkpoint must be the frozen shared-eye three-class model")
    size = int(ck["size"])
    model = FaceEyeUNet(base=int(ck["base"]), classes=len(SHARED_EYE_NAMES)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    image = Image.open(a.image).convert("RGB")
    width, height = image.size
    resized = image.resize((size, size), Image.Resampling.BILINEAR)
    x = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.inference_mode():
        probs = model(tensor).softmax(1)[0].cpu().numpy()
    labels = probs.argmax(0).astype(np.uint8)

    eyes = {}
    for side, sl, x_offset in (
        ("image_left", slice(0, size // 2), 0),
        ("image_right", slice(size // 2, size), size // 2),
    ):
        yy, xx_local = np.where(labels[:, sl] == 2)
        if len(xx_local) < a.minimum_pupil_pixels:
            eyes[side] = {
                "status": "uncertain_or_not_detected",
                "pupil_center_xy_px": None,
                "pupil_pixels_at_model_size": int(len(xx_local)),
                "mean_pupil_probability": None,
            }
            continue
        xx = xx_local + x_offset
        cx, cy = float(xx.mean()), float(yy.mean())
        eyes[side] = {
            "status": "synthetic_domain_candidate",
            "pupil_center_xy_px": [round(cx * width / size, 2), round(cy * height / size, 2)],
            "pupil_pixels_at_model_size": int(len(xx)),
            "mean_pupil_probability": round(float(probs[2, yy, xx].mean()), 5),
        }

    a.output_dir.mkdir(parents=True, exist_ok=True)
    # Upscale the semantic map to original image dimensions for inspection.
    semantic = Image.fromarray(labels).resize((width, height), Image.Resampling.NEAREST)
    semantic.save(a.output_dir / "semantic_mask.png")
    report = {
        "status": "RESEARCH_ONLY_SYNTHETIC_DOMAIN_INFERENCE",
        "checkpoint_sha256": sha256(a.checkpoint),
        "image_sha256": sha256(a.image),
        "device": str(device),
        "input_size_px": [width, height],
        "model_size_px": [size, size],
        "class_names": SHARED_EYE_NAMES,
        "eye_side_rule": "left and right are image halves, not subject anatomy metadata",
        "eyes": eyes,
        "limitations": [
            "trained/evaluated on one synthetic base identity and renderer",
            "not validated on real people or physical camera frames",
            "square resize may distort non-square inputs",
            "no calibrated confidence, blink recovery, or hardware calibration",
            "must not drive motors or be used as a clinical result",
        ],
    }
    (a.output_dir / "observation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
