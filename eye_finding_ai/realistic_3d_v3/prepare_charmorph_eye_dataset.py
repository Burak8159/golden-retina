"""Build a small, explicitly single-asset eye-mask dataset from CharMorph renders."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


def components(mask: np.ndarray, minimum: int = 20):
    ys, xs = np.where(mask)
    remaining = set(zip(ys.tolist(), xs.tolist()))
    result = []
    specks = 0
    h, w = mask.shape
    while remaining:
        p = remaining.pop()
        q = deque([p])
        pixels = [p]
        while q:
            y, x = q.popleft()
            for yy in range(max(0, y - 1), min(h, y + 2)):
                for xx in range(max(0, x - 1), min(w, x + 2)):
                    n = (yy, xx)
                    if n in remaining:
                        remaining.remove(n)
                        q.append(n)
                        pixels.append(n)
        arr = np.asarray(pixels, dtype=np.int32)
        if len(arr) < minimum:
            specks += len(arr)
            continue
        result.append({"pixels": arr, "cx": float(arr[:, 1].mean()),
                       "cy": float(arr[:, 0].mean()),
                       "bbox": [int(arr[:, 1].min()), int(arr[:, 0].min()),
                                int(arr[:, 1].max()), int(arr[:, 0].max())]})
    return sorted(result, key=lambda c: c["cx"]), specks


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--split-seed", type=int, default=815920)
    ap.add_argument("--train-fraction", type=float, default=.70)
    ap.add_argument("--validation-fraction", type=float, default=.15)
    ap.add_argument("--class-mode", choices=("five_class", "shared_eye"), default="five_class")
    a = ap.parse_args()
    if not 0 < a.train_fraction < 1 or not 0 < a.validation_fraction < 1:
        raise ValueError("split fractions must lie in (0,1)")
    if a.train_fraction + a.validation_fraction >= 1:
        raise ValueError("train + validation fractions must be below 1")

    src_report = json.loads((a.input_dir / "variant_label_audit.json").read_text())
    a.output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = a.output_dir / "masks"
    mask_dir.mkdir(exist_ok=True)
    accepted, rejected = [], []
    for record in src_report["cases"]:
        name = record["case"]["name"]
        case_dir = a.input_dir / name
        image_path = case_dir / "beauty.png"
        label_path = case_dir / "labels.npy"
        if not image_path.exists() or not label_path.exists():
            rejected.append({"sample": name, "reason": "missing image or labels"})
            continue
        raw = np.load(label_path)
        iris, iris_specks = components(raw == 1)
        pupil, pupil_specks = components(raw == 3)
        valid = len(iris) == 2 and len(pupil) == 2
        if valid:
            for i, p in enumerate(pupil):
                x0, y0, x1, y1 = iris[i]["bbox"]
                valid &= x0 <= p["cx"] <= x1 and y0 <= p["cy"] <= y1
        if not valid:
            rejected.append({"sample": name, "reason": "two-eye component geometry failed",
                             "iris_components": len(iris), "pupil_components": len(pupil)})
            continue

        # Labels are camera-image left/right, not anatomical laterality.
        classes = np.zeros(raw.shape, dtype=np.uint8)
        for eye_idx, (ir, pu) in enumerate(zip(iris, pupil)):
            if a.class_mode == "shared_eye":
                iris_class, pupil_class = 1, 2
            else:
                iris_class = 1 if eye_idx == 0 else 3
                pupil_class = 2 if eye_idx == 0 else 4
            classes[ir["pixels"][:, 0], ir["pixels"][:, 1]] = iris_class
            classes[pu["pixels"][:, 0], pu["pixels"][:, 1]] = pupil_class
        out_mask = mask_dir / f"{name}_eye_classes.png"
        Image.fromarray(classes, mode="L").save(out_mask)
        accepted.append({
            "sample_id": name,
            "seed": int(record["seed"]),
            "subject_id": "charmorph_vitruvian_base_01",
            "image": str(image_path.resolve()),
            "mask": str(out_mask.resolve()),
            "face_preset": record["case"].get("face_preset", "asset_default"),
            "gaze_x_m": record["case"].get("gaze_x_m", 0),
            "gaze_z_m": record["case"].get("gaze_z_m", 0),
            "image_left_pupil_center_xy_px": [pupil[0]["cx"], pupil[0]["cy"]],
            "image_right_pupil_center_xy_px": [pupil[1]["cx"], pupil[1]["cy"]],
            "ignored_iris_speck_pixels": iris_specks,
            "ignored_pupil_speck_pixels": pupil_specks,
            "image_sha256": sha256(image_path),
            "label_sha256": sha256(out_mask),
        })

    if len(accepted) < 10:
        raise RuntimeError(f"only {len(accepted)} valid cases; refusing to make a tiny split")
    order = list(range(len(accepted)))
    random.Random(a.split_seed).shuffle(order)
    n_train = int(round(len(order) * a.train_fraction))
    n_val = int(round(len(order) * a.validation_fraction))
    splits = {idx: ("train" if pos < n_train else
                    "validation" if pos < n_train + n_val else "test")
              for pos, idx in enumerate(order)}
    columns = ["image", "mask", "split", "subject_id", "sample_id", "left_blink", "right_blink"]
    with (a.output_dir / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for i, row in enumerate(accepted):
            writer.writerow({"image": row["image"], "mask": row["mask"], "split": splits[i],
                             "subject_id": row["subject_id"], "sample_id": row["sample_id"],
                             "left_blink": 0, "right_blink": 0})
    for i, row in enumerate(accepted):
        row["split"] = splits[i]

    split_counts = {name: sum(r["split"] == name for r in accepted)
                    for name in ("train", "validation", "test")}
    report = {
        "status": "SYNTHETIC_SINGLE_ASSET_DATASET_READY" if not rejected else "SYNTHETIC_SINGLE_ASSET_DATASET_WITH_REJECTIONS",
        "n_input_cases": len(src_report["cases"]),
        "n_accepted": len(accepted), "n_rejected": len(rejected),
        "split_counts": split_counts,
        "split_method": "deterministic sample-seed split; this does not create independent human subjects",
        "independent_identity_count": 1,
        "class_mode": a.class_mode,
        "source_asset": "CC0 CharMorph Vitruvian; one base mesh with recorded morph controls",
        "class_mapping": ({"0": "background", "1": "iris", "2": "pupil"}
                          if a.class_mode == "shared_eye" else
                          {"0": "background", "1": "image_left_iris", "2": "image_left_pupil",
                           "3": "image_right_iris", "4": "image_right_pupil"}),
        "mean_pixel_centers": {
            side: [float(np.mean([r[f"{side}_pupil_center_xy_px"][i] for r in accepted])) for i in (0,1)]
            for side in ("image_left", "image_right")
        },
        "accepted": accepted,
        "rejected": rejected,
        "limitations": ["synthetic-only", "one base identity", "seed holdout is not person-held-out",
                        "no real camera, human, temporal sequence, or demographic validation",
                        "blink frames are not included"],
    }
    (a.output_dir / "dataset_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "accepted": len(accepted),
                      "rejected": len(rejected), "split_counts": split_counts}, indent=2))


if __name__ == "__main__":
    main()
