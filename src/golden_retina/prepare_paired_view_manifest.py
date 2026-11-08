from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


DEEPDRID_SUFFIX_FIELD = {"1": "optic_disc", "2": "macula"}


def drtid_fields(root: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    grade_dir = root / "external/DRTiD_download/DRTiD/Ground Truths/DR_grade"
    for path in sorted(grade_dir.glob("*.csv")):
        if path.name.startswith("._"):
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                for column, field in (("Macula", "macula"), ("Optic disc", "optic_disc")):
                    stem = Path(row[column]).stem
                    if stem in mapping and mapping[stem] != field:
                        raise ValueError(f"Conflicting DRTiD anatomical field for {stem}")
                    mapping[stem] = field
    if not mapping:
        raise FileNotFoundError(f"No DRTiD field metadata under {grade_dir}")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a patient-split-preserving paired-view DR manifest.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--input", type=Path, required=True, help="Resolved deduplicated source manifest.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--first-view-only", action="store_true",
                        help="Keep only the macula-centered view from complete pairs as a matched single-view control.")
    args = parser.parse_args()
    root = args.root.resolve()
    field_map = drtid_fields(root)
    rows = list(csv.DictReader((root / args.input).open("r", encoding="utf-8", newline="")))
    selected: list[dict[str, str]] = []
    paired_groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    singleton_groups: list[dict[str, str]] = []
    excluded = Counter()

    for row in rows:
        source = row["source"]
        if source not in {"drtid_paired", "deepdrid", "ddr_hf_specificity300"}:
            continue
        grade = int(row["dr_grade"])
        if grade < 0 and int(row["ungradable"]) != 1:
            excluded[f"{source}:unlabeled_grade"] += 1
            continue
        grade = 5 if int(row["ungradable"]) == 1 else grade
        image_path = Path(row["image_path"])
        stem = image_path.stem
        if source == "drtid_paired":
            if stem not in field_map:
                raise ValueError(f"DRTiD image lacks anatomical metadata: {stem}")
            field = field_map[stem]
            eye_id = row["patient_id"]
            paired_groups[(source, row["split"], eye_id)].append({**row, "_field": field, "_grade": str(grade)})
        elif source == "deepdrid":
            match = re.search(r"_([lr])(\d)$", stem, flags=re.IGNORECASE)
            if not match:
                raise ValueError(f"Cannot parse DeepDRiD eye/view suffix: {stem}")
            side, suffix = match.group(1).lower(), match.group(2)
            field = DEEPDRID_SUFFIX_FIELD[suffix]
            eye_id = f"{row['patient_id']}:{side}"
            paired_groups[(source, row["split"], eye_id)].append({**row, "_field": field, "_grade": str(grade)})
        else:
            singleton_groups.append({**row, "_field": "single", "_grade": str(grade), "_eye_id": row["image_id"]})

    pair_audit = Counter()
    for (source, split, eye_id), group in sorted(paired_groups.items()):
        fields = [item["_field"] for item in group]
        grades = {item["_grade"] for item in group}
        patients = {item["patient_id"] for item in group}
        if len(grades) != 1 or len(patients) != 1:
            raise ValueError(f"Inconsistent label/patient within eye bag {source}/{eye_id}")
        expected = {"macula", "optic_disc"}
        if len(fields) != 2 or set(fields) != expected:
            pair_audit[f"{source}:{split}:incomplete"] += 1
            excluded[f"{source}:incomplete_eye_bag"] += len(group)
            continue
        pair_audit[f"{source}:{split}:complete"] += 1
        chosen = [item for item in group if item["_field"] == "macula"] if args.first_view_only else group
        for item in chosen:
            selected.append({
                "source_split": item["split"],
                "source_dataset": source,
                "patient_id": item["patient_id"],
                "eye_id": eye_id,
                "field": item["_field"],
                "image_id": item["image_id"],
                "image_path": item["image_path"],
                "grade": item["_grade"],
                "referable_dr": str(int(2 <= int(item["_grade"]) <= 4)),
            })
    for item in singleton_groups:
        selected.append({
            "source_split": item["split"],
            "source_dataset": item["source"],
            "patient_id": item["patient_id"],
            "eye_id": item["_eye_id"],
            "field": "single",
            "image_id": item["image_id"],
            "image_path": item["image_path"],
            "grade": item["_grade"],
            "referable_dr": str(int(2 <= int(item["_grade"]) <= 4)),
        })

    patients_by_split: dict[str, set[str]] = defaultdict(set)
    eye_counts = Counter()
    for item in selected:
        patients_by_split[item["source_split"]].add(item["patient_id"])
        eye_counts[(item["source_dataset"], item["source_split"])] += 1
    split_overlap = set(patients_by_split["train"]) & set(patients_by_split["valid"])
    split_overlap |= set(patients_by_split["train"]) & set(patients_by_split["test"])
    split_overlap |= set(patients_by_split["valid"]) & set(patients_by_split["test"])
    if split_overlap:
        raise ValueError(f"Patient leakage across splits: {sorted(split_overlap)[:5]}")

    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    audit = {
        "status": "PAIRED_VIEW_MANIFEST_READY_FOR_DEVELOPMENT_ONLY",
        "source_manifest": str(args.input),
        "source_manifest_note": "Resolved duplicate-cleaned manifest; original patient-level splits retained.",
        "view_mapping": {
            "DRTiD": "Official Ground Truth CSV fields Macula and Optic disc; same record includes ID and LR.",
            "DeepDRiD": "Suffix mapping from configs/deepdrid.yaml, inferred by sample-image visual audit (suffix 1=optic_disc, suffix 2=macula); five incomplete eye bags excluded.",
            "DDR": "Single image per pseudo-eye; patient identity is unverified.",
        },
        "input_condition": "macula-only control" if args.first_view_only else "paired-view fusion",
        "included_rows_by_source_split": {f"{s}:{sp}": n for (s, sp), n in sorted(eye_counts.items())},
        "complete_pair_eyes_by_source_split": dict(sorted(pair_audit.items())),
        "excluded_rows": dict(sorted(excluded.items())),
        "unique_patients_by_split": {sp: len(ids) for sp, ids in sorted(patients_by_split.items())},
        "patient_overlap_across_splits": 0,
        "clinical_claim": "none; development comparison only",
        "test_handling": "Test rows appear in the split manifest only. Invoke the training script with --skip-test-evaluation; do not tune or score test rows.",
    }
    audit_path = root / args.audit
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
