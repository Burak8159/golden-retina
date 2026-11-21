#!/usr/bin/env python3
"""Build private disease-vs-Healthy task manifests from a frozen UWF split.

The script writes row-level manifests only on the authorized VM. Its stdout and
summary JSON contain aggregate counts and hashes, never patient or image IDs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

DISEASES = ("AMD", "DR", "PM", "RD", "RVO", "Uveitis")
HEALTHY = "Healthy"
FIELDS = ("image_id", "relative_image_path", "patient_group", "eye_id", "source_label", "split")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not set(FIELDS).issubset(reader.fieldnames or ()):
            raise ValueError(f"Frozen manifest is missing required fields: {FIELDS}")
        return list(reader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = read_rows(args.manifest)
    if not rows:
        raise ValueError("Frozen manifest is empty")
    if any(r["split"] not in {"train", "validation", "test"} for r in rows):
        raise ValueError("Unexpected split value")
    if any(not r["patient_group"] or not r["eye_id"] for r in rows):
        raise ValueError("Missing patient/eye linkage; refusing task build")

    all_patient_assignments: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        all_patient_assignments[row["patient_group"]].add(row["split"])
    if any(len(splits) != 1 for splits in all_patient_assignments.values()):
        raise ValueError("Patient leakage detected in frozen manifest")

    observed = {r["source_label"] for r in rows}
    if HEALTHY not in observed:
        raise ValueError("Healthy control category is missing")
    unknown = observed - set(DISEASES) - {HEALTHY}
    if unknown:
        raise ValueError(f"Unexpected labels in frozen manifest: {sorted(unknown)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "PRIVATE_DISEASE_TASK_MANIFESTS_BUILT",
        "source": "Zhejiang UWF; single-source internal development tasks",
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "tasks": {},
        "semantics": "Each task compares one source-defined disease category with Healthy only; all other disease categories are excluded.",
        "exposure_boundary": "The Zhejiang cohort was previously analyzed for DR versus Healthy. These are not independent external validation tasks. Other disease model outputs were not found in the previous DR-only report, but the source cohort itself is known to the project.",
        "clinical_claim": "None; retrospective development manifests only.",
    }
    for disease in DISEASES:
        selected = [r for r in rows if r["source_label"] in {disease, HEALTHY}]
        if not any(r["source_label"] == disease for r in selected):
            continue
        slug = disease.lower().replace(" ", "_")
        dest = args.output_dir / f"{slug}_vs_healthy.csv"
        counts: dict[str, dict[str, set[str] | int]] = {
            split: {"images_positive": 0, "images_negative": 0,
                    "patients_positive": set(), "patients_negative": set(),
                    "eyes_positive": set(), "eyes_negative": set()}
            for split in ("train", "validation", "test")
        }
        with dest.open("w", newline="", encoding="utf-8") as stream:
            fields = ("image_id", "image_path", "patient_id", "eye_id", "target", "split")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in selected:
                positive = row["source_label"] == disease
                target = int(positive)
                image = (args.image_root / row["relative_image_path"]).resolve()
                if not image.is_file():
                    raise FileNotFoundError("One or more frozen image paths are unresolved")
                writer.writerow({"image_id": row["image_id"], "image_path": str(image),
                                 "patient_id": row["patient_group"], "eye_id": row["eye_id"],
                                 "target": target, "split": row["split"]})
                c = counts[row["split"]]
                key = "positive" if positive else "negative"
                c[f"images_{key}"] += 1
                c[f"patients_{key}"].add(row["patient_group"])
                c[f"eyes_{key}"].add((row["patient_group"], row["eye_id"]))
        overlaps = {}
        by_split = {s: counts[s]["patients_positive"] | counts[s]["patients_negative"] for s in counts}
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            overlaps[f"{a}_{b}"] = len(by_split[a] & by_split[b])
        if any(overlaps.values()):
            raise RuntimeError("Task manifest contains patient overlap")
        compact = {}
        for split, c in counts.items():
            compact[split] = {
                "positive_images": c["images_positive"],
                "negative_images": c["images_negative"],
                "positive_patients": len(c["patients_positive"]),
                "negative_patients": len(c["patients_negative"]),
                "positive_eyes": len(c["eyes_positive"]),
                "negative_eyes": len(c["eyes_negative"]),
            }
        report["tasks"][disease] = {
            "manifest_file": dest.name,
            "manifest_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            "counts": compact,
            "patient_overlap": overlaps,
        }
    summary_path = args.output_dir / "task_manifest_summary.json"
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
