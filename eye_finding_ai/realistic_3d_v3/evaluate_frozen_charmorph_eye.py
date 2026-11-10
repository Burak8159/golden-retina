"""Evaluate a frozen eye segmenter on fresh CharMorph morph seeds.

This is still a one-base-mesh synthetic-domain check, not a human or camera
validation. The input cohort must be generated after checkpoint selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from prepare_charmorph_eye_dataset import components
from train_charmorph_eye import (EyeDataset, SHARED_EYE_NAMES, evaluate)
from model import FaceEyeUNet


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()

    src = json.loads((a.cohort / "variant_label_audit.json").read_text())
    if src.get("status") != "PASS":
        raise RuntimeError("fresh cohort render/label audit did not pass")
    a.output.mkdir(parents=True, exist_ok=True)
    masks = a.output / "masks"
    masks.mkdir(exist_ok=True)
    rows, rejected = [], []
    for case in src["cases"]:
        name = case["case"]["name"]
        folder = a.cohort / name
        image = folder / "beauty.png"
        raw = np.load(folder / "labels.npy")
        iris, _ = components(raw == 1)
        pupil, _ = components(raw == 3)
        if len(iris) != 2 or len(pupil) != 2:
            rejected.append({"sample": name, "iris_components": len(iris),
                             "pupil_components": len(pupil)})
            continue
        mask = np.zeros(raw.shape, dtype=np.uint8)
        for c in iris:
            xy = c["pixels"]
            mask[xy[:, 0], xy[:, 1]] = 1
        for c in pupil:
            xy = c["pixels"]
            mask[xy[:, 0], xy[:, 1]] = 2
        mask_path = masks / f"{name}.png"
        Image.fromarray(mask).save(mask_path)
        rows.append({"image": str(image.resolve()), "mask": str(mask_path.resolve()),
                     "split": "test", "subject_id": "charmorph_vitruvian_base_01",
                     "sample_id": name, "left_blink": 0, "right_blink": 0})
    if rejected or len(rows) != len(src["cases"]):
        raise RuntimeError(f"fresh cohort rejected samples: {rejected}")
    manifest = a.output / "manifest.csv"
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if ck.get("classes") != SHARED_EYE_NAMES:
        raise ValueError("checkpoint is not the selected shared-eye model")
    model = FaceEyeUNet(base=int(ck["base"]), classes=len(SHARED_EYE_NAMES))
    model.load_state_dict(ck["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(EyeDataset(manifest, "test", a.size, False, "shared_eye"),
                        batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
                        pin_memory=device.type == "cuda")
    metrics = evaluate(model, loader, device, SHARED_EYE_NAMES, "shared_eye")
    report = {
        "status": "FRESH_SEED_SYNTHETIC_EVALUATION_ONLY",
        "device": str(device), "n_samples": len(rows),
        "seed_start": src.get("sampling_seed"),
        "checkpoint_sha256": sha256(a.checkpoint),
        "fresh_cohort_manifest_sha256": sha256(manifest),
        "metrics": metrics,
        "selection_used_this_cohort": False,
        "independent_identity_count": 1,
        "limitations": ["single synthetic base identity", "held-out renderer seeds only",
                        "not person-held-out", "not real-camera or real-world tracking validation",
                        "no physical eyeball-rotation or temporal tracking evaluation"],
        "rejected": rejected,
    }
    (a.output / "frozen_fresh_seed_evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
