"""Create a patient-disjoint train/validation/test manifest from MFIDDR labels."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", required=True, help="Output from prepare_mfiddr.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("validation fraction must be between 0 and 0.5")

    with Path(args.canonical).open("r", encoding="utf-8-sig", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    if not source_rows:
        raise ValueError("Canonical MFIDDR manifest is empty")

    patient_grades: dict[str, list[int]] = defaultdict(list)
    source_patients: dict[str, set[str]] = defaultdict(set)
    for row in source_rows:
        patient = str(row["patient_id"])
        split = str(row["source_split"])
        patient_grades[patient].append(int(row["grade"]))
        source_patients[split].add(patient)
    train_patients = source_patients.get("official_train", set())
    test_patients = source_patients.get("official_test", set())
    overlap = sorted(train_patients & test_patients)
    if overlap:
        raise ValueError(f"Official train/test patient overlap ({len(overlap)}): {overlap[:10]}")
    if not train_patients or not test_patients:
        raise ValueError("Both official train and official test patients are required")

    # Stratify the validation draw by each patient's highest observed eye grade,
    # while assigning both eyes and all four fields from a patient together.
    strata: dict[int, list[str]] = defaultdict(list)
    for patient in train_patients:
        strata[max(patient_grades[patient])].append(patient)
    rng = random.Random(args.seed)
    validation_patients: set[str] = set()
    for grade, patients in sorted(strata.items()):
        patients = sorted(patients)
        rng.shuffle(patients)
        if len(patients) > 1:
            n_validation = min(len(patients) - 1, max(1, round(len(patients) * args.validation_fraction)))
            validation_patients.update(patients[:n_validation])
    training_patients = train_patients - validation_patients

    output_rows = []
    seen_images: set[str] = set()
    for row in source_rows:
        patient = str(row["patient_id"])
        original_split = str(row["source_split"])
        if original_split == "official_test":
            split = "test"
        elif patient in validation_patients:
            split = "valid"
        elif patient in training_patients:
            split = "train"
        else:
            raise ValueError(f"Patient not assigned to a split: {patient}")
        image_id = Path(row["image_path"]).name
        if image_id in seen_images:
            raise ValueError(f"Duplicate image filename in manifest: {image_id}")
        seen_images.add(image_id)
        grade = int(row["grade"])
        output_rows.append({
            "patient_id": patient,
            "eye_id": str(row["eye_id"]),
            "image_id": image_id,
            "field": str(row["field"]),
            "grade": grade,
            "referable_dr": int(grade >= 2),
            "image_path": str(row["image_path"]),
            "source_split": f"mfiddr_{split}",
            "source_dataset": "mfiddr_paired",
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    split_patients = {
        "train": training_patients,
        "valid": validation_patients,
        "test": test_patients,
    }
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        shared = split_patients[left] & split_patients[right]
        if shared:
            raise AssertionError(f"Patient leakage between {left} and {right}: {sorted(shared)[:10]}")
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "valid", "test"):
        rows = [r for r in output_rows if r["source_split"] == f"mfiddr_{split}"]
        counts[split] = {
            "patients": len(split_patients[split]),
            "eyes": len({r["eye_id"] for r in rows}),
            "images": len(rows),
            "referable_eyes": len({r["eye_id"] for r in rows if r["referable_dr"]}),
            "grade_rows": dict(sorted(Counter(r["grade"] for r in rows).items())),
        }
    summary = {
        "status": "PATIENT_DISJOINT_MANIFEST_READY",
        "unit_of_split": "patient_id",
        "official_test_preserved": True,
        "validation_fraction_within_official_train": args.validation_fraction,
        "seed": args.seed,
        "patient_overlap": {"train_valid": 0, "train_test": 0, "valid_test": 0},
        "splits": counts,
        "image_manifest": str(output),
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
