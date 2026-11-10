#!/usr/bin/env python3
"""Audit the Zhejiang UWF archive for metadata integrity and exact overlap.

Writes a private manifest with filenames/patient IDs and a deidentified
aggregate JSON summary. Does not extract or publish image data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--local-hashes", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    args = ap.parse_args()

    local_hashes: set[str] = set()
    with args.local_hashes.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            local_hashes.add(row["sha256"].strip().lower())

    with zipfile.ZipFile(args.archive) as zf:
        def workbook(name: str):
            return openpyxl.load_workbook(zf.open(name), read_only=True, data_only=True)

        demo_ws = workbook("Demographics of the participants.xlsx").active
        rows = demo_ws.iter_rows(values_only=True)
        headers = [str(x).strip() for x in next(rows)]
        demo = {h: i for i, h in enumerate(headers)}
        required = {"Image ID", "Patient ID", "Eye ID", "Diagnosis"}
        if not required.issubset(demo):
            raise ValueError(f"Unexpected demographics columns: {headers}")
        records: dict[str, dict] = {}
        for r in rows:
            if not r or not r[demo["Image ID"]]:
                continue
            filename = str(r[demo["Image ID"]]).strip()
            records[filename] = {
                "patient_id": str(r[demo["Patient ID"]]).strip(),
                "eye_id": str(r[demo["Eye ID"]]).strip(),
                "eye_category": str(r[demo["Eye category"]]).strip() if "Eye category" in demo else "",
                "diagnosis": str(r[demo["Diagnosis"]]).strip(),
            }

        gt_ws = workbook("Ground Truth.xlsx").active
        gt_rows = gt_ws.iter_rows(values_only=True)
        quality_headers = [str(x).strip() for x in next(gt_rows)]
        quality_map = {h: i for i, h in enumerate(quality_headers)}
        quality = {}
        for r in gt_rows:
            if r and r[quality_map["Image ID"]]:
                quality[str(r[quality_map["Image ID"]]).strip()] = {
                    h: r[i] for h, i in quality_map.items() if h != "Image ID"
                }

        image_names = [n for n in zf.namelist()
                       if n.startswith("Original UWF Image/") and n.lower().endswith(".jpg")]
        private_rows = []
        label_counts: Counter[str] = Counter()
        folder_counts: Counter[str] = Counter()
        patient_diagnoses: dict[str, set[str]] = defaultdict(set)
        patient_eyes: dict[str, set[str]] = defaultdict(set)
        sha_counts: Counter[str] = Counter()
        folder_vs_metadata: Counter[str] = Counter()
        overlap_hashes: set[str] = set()
        filename_to_folder = {}
        for member in image_names:
            leaf = member.rsplit("/", 1)[-1]
            folder = member.split("/")[-2]
            if leaf in filename_to_folder:
                raise ValueError(f"Filename collision in archive: {leaf}")
            filename_to_folder[leaf] = folder
            rec = records.get(leaf)
            if rec is None:
                raise ValueError(f"Image missing demographics row: {leaf}")
            with zf.open(member) as image_file:
                h = hashlib.sha256()
                for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                    h.update(chunk)
                digest = h.hexdigest()
            sha_counts[digest] += 1
            if digest in local_hashes:
                overlap_hashes.add(digest)
            folder_counts[folder] += 1
            label_counts[rec["diagnosis"]] += 1
            folder_vs_metadata[f"{folder} -> {rec['diagnosis']}"] += 1
            patient_diagnoses[rec["patient_id"]].add(rec["diagnosis"])
            patient_eyes[rec["patient_id"]].add(rec["eye_id"])
            private_rows.append({
                "member": member,
                "image_id": leaf,
                **rec,
                "quality": quality.get(leaf, {}),
                "sha256": digest,
            })

        image_set = set(filename_to_folder)
        demographic_set = set(records)
        quality_set = set(quality)
        patient_conflicts = sum(len(v) > 1 for v in patient_diagnoses.values())
        summary = {
            "source": "Zhejiang Provincial People's Hospital UWF dataset, Figshare article 26936446",
            "archive_md5_verified_before_audit": "ddee42dc99792545654a65413c0fdb6f",
            "archive_bytes_verified_before_audit": 9018811972,
            "images_in_archive": len(image_names),
            "demographics_rows": len(records),
            "quality_ground_truth_rows": len(quality),
            "image_demographics_filename_set_equal": image_set == demographic_set,
            "image_quality_filename_set_equal": image_set == quality_set,
            "diagnosis_counts": dict(sorted(label_counts.items())),
            "folder_counts": dict(sorted(folder_counts.items())),
            "folder_to_metadata_diagnosis_counts": dict(sorted(folder_vs_metadata.items())),
            "folder_metadata_disagreements": sum(
                n for pair, n in folder_vs_metadata.items()
                if pair.split(" -> ", 1)[0].casefold() != pair.split(" -> ", 1)[1].casefold()
            ),
            "unique_patients": len(patient_diagnoses),
            "unique_eye_ids": len({r["eye_id"] for r in records.values()}),
            "patients_with_both_eyes_by_eye_category": sum(len(v) > 1 for v in patient_eyes.values()),
            "patients_with_multiple_diagnoses": patient_conflicts,
            "exact_duplicate_image_groups_within_archive": sum(n > 1 for n in sha_counts.values()),
            "images_exactly_overlapping_local_inventory": sum(sha_counts[h] for h in overlap_hashes),
            "unique_exact_overlap_hashes_with_local_inventory": len(overlap_hashes),
            "quality_column_names": [h for h in quality_headers if h != "Image ID"],
            "images_missing_quality_ground_truth": len(image_set - quality_set),
            "manifest_status": "candidate only; use remains blocked until source/label semantics and overlap policy are reviewed",
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open("w", newline="", encoding="utf-8") as f:
            fields = ["member", "image_id", "patient_id", "eye_id", "eye_category", "diagnosis", "quality", "sha256"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in private_rows:
                row["quality"] = json.dumps(row["quality"], sort_keys=True)
                writer.writerow(row)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
