#!/usr/bin/env python3
"""Build a deterministic patient-grouped split for the staged Zhejiang UWF cohort."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_workbook_headers import _read_subject_rows


def split_sizes(n: int):
    train = int(round(n * 0.60))
    valid = int(round(n * 0.20))
    test = n - train - valid
    if min(train, valid, test) < 1:
        raise ValueError(f"Not enough patient groups for a 60/20/20 split: n={n}")
    return train, valid, test


def main(root: Path, output: Path, seed: str):
    workbook = root / "Demographics of the participants.xlsx"
    if not workbook.is_file():
        raise FileNotFoundError(workbook)
    rows = _read_subject_rows(workbook, "Sheet1")
    if not rows:
        raise ValueError("No demographic records found")
    needed = {"Image ID", "Patient ID", "Eye ID", "Diagnosis"}
    if not needed.issubset(rows[0]):
        raise ValueError("Demographics workbook is missing expected grouping/label columns")

    labels_by_patient = defaultdict(set)
    patients_by_label = defaultdict(set)
    for row in rows:
        patient = row["Patient ID"].strip()
        label = row["Diagnosis"].strip()
        if not patient or not label or not row["Image ID"].strip():
            raise ValueError("Missing patient, label, or image ID; cannot freeze split")
        labels_by_patient[patient].add(label)
        patients_by_label[label].add(patient)
    cross_label = [p for p, labels in labels_by_patient.items() if len(labels) > 1]
    if cross_label:
        raise ValueError(f"{len(cross_label)} patient groups have multiple exclusive diagnosis labels; resolve before binary split construction")

    split_by_patient = {}
    split_counts = {}
    for label, patients in sorted(patients_by_label.items()):
        ordered = sorted(patients, key=lambda patient: hashlib.sha256((seed + "\0" + patient).encode()).hexdigest())
        n_train, n_valid, n_test = split_sizes(len(ordered))
        assignments = ["train"] * n_train + ["validation"] * n_valid + ["test"] * n_test
        for patient, split in zip(ordered, assignments):
            split_by_patient[patient] = split
        split_counts[label] = {"patients": len(ordered), "train": n_train, "validation": n_valid, "test": n_test}

    image_roots = root / "Original UWF Image"
    image_paths = {}
    for path in image_roots.rglob("*"):
        if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            image_paths.setdefault(path.stem, []).append(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    fields = ["image_id", "relative_image_path", "patient_group", "eye_id", "source_label", "split"]
    row_counts = defaultdict(lambda: defaultdict(int))
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            image_id = row["Image ID"].strip()
            matches = image_paths.get(Path(image_id).stem, [])
            if len(matches) != 1:
                raise ValueError(f"Image ID resolution failed for one or more records: count={len(matches)}")
            split = split_by_patient[row["Patient ID"].strip()]
            label = row["Diagnosis"].strip()
            writer.writerow({
                "image_id": image_id,
                "relative_image_path": str(matches[0].relative_to(root)),
                "patient_group": row["Patient ID"].strip(),
                "eye_id": row["Eye ID"].strip(),
                "source_label": label,
                "split": split,
            })
            row_counts[label][split] += 1
    temp.replace(output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    assignments = defaultdict(set)
    for patient, split in split_by_patient.items():
        assignments[split].add(patient)
    overlap = {"train_validation": len(assignments["train"] & assignments["validation"]), "train_test": len(assignments["train"] & assignments["test"]), "validation_test": len(assignments["validation"] & assignments["test"])}
    if any(overlap.values()):
        raise RuntimeError("Patient leakage across frozen partitions")
    report = {
        "dataset": "Zhejiang UWF staged archive",
        "split_unit": "Patient ID; all eyes/images from a patient share one split",
        "seed": seed,
        "image_rows": len(rows),
        "patient_groups": len(split_by_patient),
        "split_rule": "Within each single-label diagnosis category: deterministic SHA-256 order by seed and patient ID, then 60/20/20 patient assignment.",
        "patient_counts_by_label": split_counts,
        "image_rows_by_label_and_split": {label: dict(counts) for label, counts in sorted(row_counts.items())},
        "patient_overlap_counts": overlap,
        "patient_ids_with_multiple_exclusive_diagnosis_labels": len(cross_label),
        "manifest_sha256": digest,
        "manifest_path": str(output),
        "boundary": "Internal patient-grouped holdout within one UWF source; not an independent-source or clinical validation.",
    }
    report_path = output.with_suffix(".summary.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", default="golden-retina-uwf-patient-split-v1")
    args = parser.parse_args()
    main(args.root, args.output, args.seed)
