"""Audit seeded CharMorph probe outputs and export eye-center metadata."""
from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


def components(mask: np.ndarray, min_pixels: int = 20):
    ys, xs = np.where(mask)
    remaining = set(zip(ys.tolist(), xs.tolist()))
    found = []
    while remaining:
        y0, x0 = remaining.pop()
        q = deque([(y0, x0)])
        pix = [(y0, x0)]
        while q:
            y, x = q.popleft()
            for ny in range(max(0, y - 1), min(mask.shape[0], y + 2)):
                for nx in range(max(0, x - 1), min(mask.shape[1], x + 2)):
                    pt = (ny, nx)
                    if pt in remaining:
                        remaining.remove(pt)
                        q.append(pt)
                        pix.append(pt)
        py = np.fromiter((p[0] for p in pix), dtype=np.int32)
        px = np.fromiter((p[1] for p in pix), dtype=np.int32)
        found.append({
            "pixels": int(len(pix)),
            "center_xy": [float(px.mean()), float(py.mean())],
            "bbox_xyxy": [int(px.min()), int(py.min()), int(px.max()), int(py.max())],
        })
    kept = [c for c in found if c["pixels"] >= min_pixels]
    specks = [c for c in found if c["pixels"] < min_pixels]
    return sorted(kept, key=lambda c: c["center_xy"][0]), specks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir
    source = json.loads((root / "variant_label_audit.json").read_text())
    rows = []
    ok = True
    for rec in source["cases"]:
        case_dir = root / rec["case"]["name"]
        image = Image.open(case_dir / "beauty.png")
        labels = np.load(case_dir / "labels.npy")
        iris, iris_specks = components(labels == 1)
        pupil, pupil_specks = components(labels == 3)
        geometry_ok = len(iris) == 2 and len(pupil) == 2
        if geometry_ok:
            for pi, ii in zip(pupil, iris):
                x, y = pi["center_xy"]
                x0, y0, x1, y1 = ii["bbox_xyxy"]
                if not (x0 <= x <= x1 and y0 <= y <= y1):
                    geometry_ok = False
        rgb_ok = image.mode == "RGB" and image.size == (768, 768)
        if labels.shape != (768, 768):
            rgb_ok = False
        item = {
            "sample": rec["case"]["name"],
            "seed": rec["seed"],
            "gaze_x_m": rec["case"]["gaze_x_m"],
            "gaze_z_m": rec["case"]["gaze_z_m"],
            "iris_components": iris,
            "pupil_components": pupil,
            "ignored_iris_specks": iris_specks,
            "ignored_pupil_specks": pupil_specks,
            "two_eye_geometry_ok": geometry_ok,
            "rgb_768_ok": rgb_ok,
        }
        rows.append(item)
        ok &= geometry_ok and rgb_ok

    center = rows[0]
    for row in rows:
        if row["sample"] == center["sample"]:
            row["pupil_shift_from_center_px"] = [[0.0, 0.0], [0.0, 0.0]]
        else:
            row["pupil_shift_from_center_px"] = [
                [round(b["center_xy"][0] - a["center_xy"][0], 3),
                 round(b["center_xy"][1] - a["center_xy"][1], 3)]
                for a, b in zip(center["pupil_components"], row["pupil_components"])
            ]

    report = {
        "status": "PASS" if ok else "FAIL",
        "n_cases": len(rows),
        "independent_identity_count": source.get("independent_identity_count"),
        "criteria": ["RGB 768x768 beauty image", "exactly two connected iris components of at least 20 pixels",
                     "exactly two connected pupil components of at least 20 pixels",
                     "each pupil centroid inside corresponding iris bbox"],
        "speck_policy": "Components under 20 px are reported but ignored by the eye-count check; original labels are unchanged.",
        "results": rows,
        "limitations": ["single synthetic base identity", "small iris/pupil translation approximates gaze",
                        "not a calibrated eyeball rotation or real-camera benchmark"],
    }
    (root / "center_geometry_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    with (root / "eye_centers.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "seed", "gaze_x_m", "gaze_z_m", "eye_side",
                         "pupil_center_x_px", "pupil_center_y_px", "pupil_pixels",
                         "iris_center_x_px", "iris_center_y_px", "iris_pixels",
                         "rgb_768_ok", "geometry_ok"])
        for row in rows:
            for side, (iris, pupil) in enumerate(zip(row["iris_components"], row["pupil_components"])):
                writer.writerow([row["sample"], row["seed"], row["gaze_x_m"], row["gaze_z_m"],
                                 "image_left" if side == 0 else "image_right",
                                 *[round(x, 3) for x in pupil["center_xy"]], pupil["pixels"],
                                 *[round(x, 3) for x in iris["center_xy"]], iris["pixels"],
                                 row["rgb_768_ok"], row["two_eye_geometry_ok"]])
    print(json.dumps({"status": report["status"], "n_cases": len(rows),
                      "bad_samples": [r["sample"] for r in rows if not r["two_eye_geometry_ok"] or not r["rgb_768_ok"]],
                      "center_pupil_xy": center["pupil_components"]}, indent=2))


if __name__ == "__main__":
    main()
