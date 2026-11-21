#!/usr/bin/env python3
"""VM-only, aggregate-only schema audit for an authorized mBRSET release.

The report intentionally excludes filenames, patient identifiers, and row-level
values. It checks the actual release instead of assuming the website's aggregate
label summary matches a CSV column or that described image centers are paired.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

DATA_ROOT = Path("/data/golden-retina")
DEFAULT_CSV = DATA_ROOT / "workspace/external/mbrset/labels_mbrset.csv"
DEFAULT_IMAGES = DATA_ROOT / "workspace/external/mbrset/images"
DEFAULT_REPORT = DATA_ROOT / "workspace/reports/mbrset_schema_audit.json"
SUPPRESS_BELOW = 10

ALIASES = {
    "patient": {"patient_id", "patientid", "patient", "id_patient", "subject_id", "subjectid"},
    "laterality": {"laterality", "eye", "eye_side", "side", "od_os"},
    "file": {"file", "filename", "image", "image_file", "image_path", "path"},
    "icdr": {"final_icdr", "icdr", "dr_grade", "grade", "diabetic_retinopathy_grade"},
    "edema": {"final_edema", "edema", "macular_edema", "dme"},
    "quality": {"final_quality", "quality", "gradable", "image_quality"},
    "artifacts": {"final_artifacts", "artifacts", "artifact"},
    "referable_dr": {"referable_dr", "referabledr", "referable_dr_label"},
}
VIEW_HINTS = ("view", "center", "centred", "centered", "field", "image_type", "photo_type")


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def is_missing(value: str | None) -> bool:
    return value is None or value.strip().lower() in {"", "na", "n/a", "nan", "null", "none"}


def safe_counts(values: Counter[str]) -> dict[str, int | str]:
    return {
        key: (count if count >= SUPPRESS_BELOW else f"<{SUPPRESS_BELOW}")
        for key, count in sorted(values.items())
    }


def safe_count(value: int | None) -> int | str | None:
    if value is None:
        return None
    return value if value >= SUPPRESS_BELOW else f"<{SUPPRESS_BELOW}"


def inside_root(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    root = DATA_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("All inputs and outputs must remain under /data/golden-retina.")
    return resolved


def find_column(headers: Iterable[str], semantic: str) -> str | None:
    lookup = {norm(header): header for header in headers}
    matches = ALIASES[semantic]
    return next((lookup[name] for name in sorted(matches) if name in lookup), None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    if platform.system() != "Linux":
        parser.error("This audit is restricted to the authorized Linux VM; do not run it on the Mac.")
    try:
        csv_path = inside_root(args.csv)
        images_root = inside_root(args.images)
        report_path = inside_root(args.report)
    except ValueError as exc:
        parser.error(str(exc))
    if not csv_path.is_file():
        parser.error("The authorized metadata CSV is missing at the supplied VM path.")
    if csv_path == report_path:
        parser.error("Report path must not overwrite an input file.")

    total = 0
    missing_by_column: Counter[str] = Counter()
    label_counts: dict[str, Counter[str]] = {name: Counter() for name in ALIASES if name not in {"patient", "laterality", "file"}}
    patient_images: Counter[str] = Counter()
    patient_eye_images: Counter[tuple[str, str]] = Counter()
    laterality_counts: Counter[str] = Counter()
    path_counts: Counter[str] = Counter()
    image_files_checked = 0
    image_files_found = 0
    unsafe_image_paths = 0
    headers: list[str] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        if not headers:
            parser.error("CSV has no header row.")
        normalized_headers = [norm(header) for header in headers]
        if len(normalized_headers) != len(set(normalized_headers)):
            parser.error("CSV has ambiguous duplicate headers after normalization; stop for manual schema review.")
        columns = {name: find_column(headers, name) for name in ALIASES}
        patient_col, side_col, file_col = columns["patient"], columns["laterality"], columns["file"]
        view_columns = [h for h in headers if any(hint in norm(h) for hint in VIEW_HINTS)]
        for row in reader:
            total += 1
            for header in headers:
                if is_missing(row.get(header)):
                    missing_by_column[header] += 1
            for semantic, column in columns.items():
                if semantic in label_counts and column:
                    value = (row.get(column) or "").strip()
                    if not is_missing(value):
                        label_counts[semantic][value] += 1
            patient_value = (row.get(patient_col) or "").strip() if patient_col else ""
            side_value = (row.get(side_col) or "").strip().lower() if side_col else ""
            filename_value = (row.get(file_col) or "").strip() if file_col else ""
            patient = "" if is_missing(patient_value) else patient_value
            side = "" if is_missing(side_value) else side_value
            filename = "" if is_missing(filename_value) else filename_value
            if patient:
                patient_images[patient] += 1
                if side:
                    patient_eye_images[(patient, side)] += 1
            if side:
                laterality_counts[side] += 1
            if filename:
                path_counts[norm(filename)] += 1
                if images_root.is_dir():
                    candidate = (images_root / filename).resolve(strict=False)
                    if candidate != images_root.resolve() and images_root.resolve() not in candidate.parents:
                        unsafe_image_paths += 1
                    else:
                        image_files_checked += 1
                        image_files_found += int(candidate.is_file())

    repeated_patient_groups = sum(count > 1 for count in patient_images.values())
    repeated_patient_eye_groups = sum(count > 1 for count in patient_eye_images.values())
    duplicate_file_keys = sum(count > 1 for count in path_counts.values())
    report = {
        "audit": "mBRSET aggregate schema and linkage audit",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "environment": "Linux VM; private report; aggregate counts only",
        "input_csv_relative": str(csv_path.relative_to(DATA_ROOT.resolve())),
        "image_directory_present": images_root.is_dir(),
        "rows": safe_count(total),
        "unique_patient_count": safe_count(len(patient_images)) if patient_col else None,
        "columns": headers,
        "semantic_columns": columns,
        "label_value_counts": {name: safe_counts(counts) for name, counts in label_counts.items()},
        "missing_cells_by_column": {key: safe_count(value) for key, value in sorted(missing_by_column.items())},
        "laterality_value_counts": safe_counts(laterality_counts),
        "patient_groups_with_multiple_rows": safe_count(repeated_patient_groups) if patient_col else None,
        "patient_laterality_groups_with_multiple_rows": safe_count(repeated_patient_eye_groups) if patient_col and side_col else None,
        "candidate_view_columns": view_columns,
        "view_pairing_conclusion": "unverified; repeated patient-eye rows alone do not establish distinct paired views",
        "referable_dr_column_present": columns["referable_dr"] is not None,
        "referable_dr_mapping_conclusion": "use supplied documented label only" if columns["referable_dr"] else "not found in CSV; do not derive without a documented prespecified mapping",
        "duplicate_normalized_file_keys": safe_count(duplicate_file_keys) if file_col else None,
        "image_file_checks": {
            "rows_checked": safe_count(image_files_checked),
            "files_found": safe_count(image_files_found),
            "unsafe_paths_rejected": safe_count(unsafe_image_paths),
            "filenames_or_patient_ids_written": False,
        },
        "small_category_counts_suppressed_below": SUPPRESS_BELOW,
        "limitations": [
            "This checks metadata structure and file references only; it does not grade images or assess model performance.",
            "Potential BRSET overlap requires a separate authorized, patient-level source overlap audit.",
            "A field name suggesting view/center is not proof of a paired acquisition protocol.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.chmod(report_path, 0o600)
    print(f"Private aggregate audit saved: {report_path}")
    print(f"Rows: {safe_count(total)}; patient column: {'present' if patient_col else 'missing'}; referable_dr column: {'present' if columns['referable_dr'] else 'not found'}")
    print(f"Candidate view columns: {len(view_columns)}; pairing remains unverified until protocol review.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
