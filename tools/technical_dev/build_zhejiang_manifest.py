"""Create patient-grouped Zhejiang image manifest from the release metadata."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from inspect_xlsx_stdlib import read_workbook

base = Path("/data/golden-retina/workspace/external/zhejiang_uwf/unpacked")
image_base = base / "Original UWF Image"
demographics = read_workbook(str(base / "Demographics of the participants.xlsx"))[0][1]
quality_table = read_workbook(str(base / "Ground Truth.xlsx"))[0][1]
header = [v.strip() for v in demographics[0]]
index = {name: i for i, name in enumerate(header)}
required = {"Image ID", "Patient ID", "Eye ID", "Diagnosis"}
if not required.issubset(index):
    raise RuntimeError(f"Missing metadata columns: {required-set(index)}")
quality_header = [v.strip() for v in quality_table[0]]
quality_index = {name: i for i, name in enumerate(quality_header)}
quality_by_image = {}
for values in quality_table[1:]:
    if values and values[quality_index["Image ID"]].strip():
        quality_by_image[values[quality_index["Image ID"]].strip()] = {
            key: int(values[quality_index[key]])
            for key in ("Field of view", "Contrast", "Illumination", "Artifacts", "Overall quality")
        }
rows = []
for values in demographics[1:]:
    if not values or not values[index["Image ID"]].strip():
        continue
    image_id = values[index["Image ID"]].strip()
    diagnosis = values[index["Diagnosis"]].strip()
    image_path = image_base / diagnosis / image_id
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if image_id not in quality_by_image:
        raise RuntimeError(f"Missing quality labels for {image_id}")
    rows.append({"image_path": str(image_path),
                 "image_id": image_id,
                 "patient_id": values[index["Patient ID"]].strip(),
                 "eye_id": values[index["Eye ID"]].strip(),
                 "diagnosis": diagnosis,
                 **{f"quality_{k.lower().replace(' ', '_')}": v
                    for k, v in quality_by_image[image_id].items()}})
if len(rows) != 700 or len({r["image_id"] for r in rows}) != 700:
    raise RuntimeError(f"Expected 700 unique rows; got {len(rows)}")
patient_labels = defaultdict(set)
for row in rows:
    patient_labels[row["patient_id"]].add(row["diagnosis"])
if any(len(labels) != 1 for labels in patient_labels.values()):
    raise RuntimeError("At least one patient has multiple diagnosis categories; inspect before evaluation")
out = Path("/data/golden-retina/workspace/results/technical_dev_20261107/zhejiang_external_test_manifest.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
digest = hashlib.sha256(out.read_bytes()).hexdigest()
summary = {
    "status": "FROZEN_PATIENT_GROUPED_SOURCE_HOLDOUT_MANIFEST",
    "images": len(rows), "patients": len(patient_labels),
    "eyes": len({r["eye_id"] for r in rows}),
    "images_by_diagnosis": dict(Counter(r["diagnosis"] for r in rows)),
    "patients_by_diagnosis": {d: len({r["patient_id"] for r in rows if r["diagnosis"] == d})
                              for d in sorted({r["diagnosis"] for r in rows})},
    "manifest_sha256": digest,
    "manifest_path": str(out),
    "label_semantics": "Dataset source diagnosis category, not referable-DR severity grade.",
    "no_training_or_threshold_selection": True,
}
(out.parent / "zhejiang_external_test_manifest_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
print(json.dumps(summary, indent=2), flush=True)
