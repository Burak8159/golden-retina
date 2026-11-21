#!/usr/bin/env python3
"""Read XLSX headers and aggregate diagnosis counts; never print subject rows."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def column_index(cell_reference: str) -> Optional[int]:
    match = re.match(r"[A-Z]+", cell_reference)
    if not match:
        return None
    index = 0
    for char in match.group(0):
        index = index * 26 + ord(char) - 64
    return index - 1


def read_workbook(path: Path):
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        strings = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = ["".join(t.text or "" for t in si.findall(".//m:t", NS)) for si in root.findall("m:si", NS)]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {item.attrib["Id"]: item.attrib["Target"] for item in rels}
        summaries = []

        def value(cell):
            node = cell.find("m:v", NS)
            text = node.text if node is not None and node.text else ""
            if cell.attrib.get("t") == "s" and text:
                return strings[int(text)]
            if cell.attrib.get("t") == "inlineStr":
                inline = cell.find("m:is", NS)
                return "".join(t.text or "" for t in inline.findall(".//m:t", NS)) if inline is not None else ""
            return text

        for sheet in workbook.findall("m:sheets/m:sheet", NS):
            target = targets[sheet.attrib["{" + NS["r"] + "}id"]].lstrip("/")
            if not target.startswith("xl/"):
                target = "xl/" + target
            sheet_root = ET.fromstring(archive.read(target))
            rows = sheet_root.findall(".//m:sheetData/m:row", NS)
            if not rows:
                summaries.append((sheet.attrib.get("name", ""), [], 0, {}))
                continue
            headers = [""] * max(1, max((column_index(c.attrib.get("r", "")) or 0) + 1 for c in rows[0].findall("m:c", NS)))
            for cell in rows[0].findall("m:c", NS):
                idx = column_index(cell.attrib.get("r", ""))
                if idx is not None:
                    headers[idx] = value(cell).strip()
            aggregates = {}
            if {"Diagnosis", "Patient ID", "Image ID"}.issubset(set(headers)):
                diag_i, patient_i, image_i = headers.index("Diagnosis"), headers.index("Patient ID"), headers.index("Image ID")
                groups = collections.defaultdict(lambda: {"images": set(), "patients": set()})
                for row in rows[1:]:
                    vals = [""] * len(headers)
                    for cell in row.findall("m:c", NS):
                        idx = column_index(cell.attrib.get("r", ""))
                        if idx is not None and idx < len(vals):
                            vals[idx] = value(cell).strip()
                    label = vals[diag_i] or "(missing)"
                    if vals[image_i]:
                        groups[label]["images"].add(vals[image_i])
                    if vals[patient_i]:
                        groups[label]["patients"].add(vals[patient_i])
                aggregates = {k: {"images": len(v["images"]), "patients": len(v["patients"])} for k, v in sorted(groups.items())}
            summaries.append((sheet.attrib.get("name", ""), headers, max(0, len(rows) - 1), aggregates))
        return summaries


def image_inventory(root: Path, patient_rows, compute_hash=True):
    expected = {}
    patient_labels = collections.defaultdict(set)
    patient_eyes = set()
    missing_patient_ids = 0
    duplicate_image_ids = 0
    seen_metadata_ids = set()
    for row in patient_rows:
        image_id = row.get("Image ID", "").strip()
        diagnosis = row.get("Diagnosis", "").strip()
        patient = row.get("Patient ID", "").strip()
        eye = row.get("Eye ID", "").strip()
        if not patient:
            missing_patient_ids += 1
        if image_id:
            if image_id in seen_metadata_ids:
                duplicate_image_ids += 1
            seen_metadata_ids.add(image_id)
            expected[Path(image_id).stem] = {"diagnosis": diagnosis, "patient": patient, "eye": eye}
        if patient and diagnosis:
            patient_labels[patient].add(diagnosis)
        if patient and eye:
            patient_eyes.add((patient, eye))
    files = [p for p in (root / "Original UWF Image").rglob("*") if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    seen_hashes = collections.defaultdict(list)
    observed_by_label = collections.Counter()
    linked_patients = collections.defaultdict(set)
    unmatched_files = 0
    matched_ids = set()
    for path in files:
        image_id = path.stem
        rec = expected.get(image_id)
        if rec is None:
            unmatched_files += 1
        else:
            matched_ids.add(image_id)
            observed_by_label[rec["diagnosis"]] += 1
            if rec["patient"]:
                linked_patients[rec["diagnosis"]].add(rec["patient"])
        if compute_hash:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            seen_hashes[digest.hexdigest()].append(image_id)
    duplicate_groups = sum(1 for group in seen_hashes.values() if len(group) > 1) if compute_hash else None
    repeated_patient_labels = sum(1 for labels in collections.defaultdict(set).values() if len(labels) > 1)
    return {
        "image_files": len(files),
        "metadata_image_ids": len(expected),
        "matched_image_ids": len(matched_ids),
        "metadata_ids_without_image": len(set(expected) - matched_ids),
        "image_files_without_metadata": unmatched_files,
        "exact_duplicate_content_groups": duplicate_groups,
        "unique_patients_total": len(patient_labels),
        "patients_with_multiple_diagnosis_labels": sum(1 for labels in patient_labels.values() if len(labels) > 1),
        "unique_patient_eye_pairs": len(patient_eyes),
        "metadata_rows_missing_patient_id": missing_patient_ids,
        "duplicate_metadata_image_ids": duplicate_image_ids,
        "images_by_diagnosis": dict(sorted(observed_by_label.items())),
        "unique_patients_by_diagnosis": {k: len(v) for k, v in sorted(linked_patients.items())},
    }

def main(root: Path, archive_path: Optional[Path], json_out: Optional[Path], skip_hash: bool) -> None:
    all_summaries = []
    patient_rows = []
    for workbook in sorted(root.glob("*.xlsx")):
        print("WORKBOOK", workbook.name)
        for sheet, headers, row_count, aggregate in read_workbook(workbook):
            print("  sheet=", sheet, "rows=", row_count, "headers=", headers)
            if aggregate:
                print("  aggregate_diagnosis_counts=", aggregate)
            # Retain subject-level values only in process memory for group counts / ID matching;
            # never print or serialize them.
            if {"Diagnosis", "Patient ID", "Image ID", "Eye ID"}.issubset(set(headers)):
                patient_rows = _read_subject_rows(workbook, sheet)
    report = {"aggregate_diagnosis_counts": {}, "image_integrity": {}}
    if patient_rows:
        by_label = collections.defaultdict(lambda: {"images": set(), "patients": set()})
        for row in patient_rows:
            label = row.get("Diagnosis", "").strip() or "(missing)"
            if row.get("Image ID", "").strip():
                by_label[label]["images"].add(row["Image ID"].strip())
            if row.get("Patient ID", "").strip():
                by_label[label]["patients"].add(row["Patient ID"].strip())
        report["aggregate_diagnosis_counts"] = {k: {"images": len(v["images"]), "patients": len(v["patients"])} for k, v in sorted(by_label.items())}
        report["image_integrity"] = image_inventory(root, patient_rows, compute_hash=not skip_hash)
        print("IMAGE_ID_AND_EXACT_DUPLICATE_AUDIT", report["image_integrity"])
    if archive_path:
        with zipfile.ZipFile(archive_path) as archive:
            counts = collections.Counter()
            for name in archive.namelist():
                parts = name.split("/")
                if len(parts) > 2 and parts[-1].lower().endswith((".jpg", ".jpeg", ".png")):
                    counts[parts[1]] += 1
            report["archive_images_by_category"] = dict(sorted(counts.items()))
            print("ARCHIVE_IMAGE_COUNTS_BY_CATEGORY", report["archive_images_by_category"])
    if json_out:
        json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("AGGREGATE_REPORT_WRITTEN", json_out)


def _read_subject_rows(workbook_path: Path, wanted_sheet: str):
    with zipfile.ZipFile(workbook_path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in si.findall(".//m:t", NS)) for si in root.findall("m:si", NS)]
        wb = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {item.attrib["Id"]: item.attrib["Target"] for item in rels}
        for sheet in wb.findall("m:sheets/m:sheet", NS):
            if sheet.attrib.get("name") != wanted_sheet:
                continue
            target = targets[sheet.attrib["{" + NS["r"] + "}id"]].lstrip("/")
            if not target.startswith("xl/"):
                target = "xl/" + target
            root = ET.fromstring(archive.read(target))
            rows = root.findall(".//m:sheetData/m:row", NS)
            def cell_value(cell):
                node = cell.find("m:v", NS)
                text = node.text if node is not None and node.text else ""
                if cell.attrib.get("t") == "s" and text:
                    return shared[int(text)]
                if cell.attrib.get("t") == "inlineStr":
                    inline = cell.find("m:is", NS)
                    return "".join(t.text or "" for t in inline.findall(".//m:t", NS)) if inline is not None else ""
                return text
            headers = {}
            for cell in rows[0].findall("m:c", NS):
                idx = column_index(cell.attrib.get("r", ""))
                if idx is not None:
                    headers[idx] = cell_value(cell).strip()
            output = []
            for row in rows[1:]:
                values = [""] * (max(headers, default=-1) + 1)
                for cell in row.findall("m:c", NS):
                    idx = column_index(cell.attrib.get("r", ""))
                    if idx is not None and idx < len(values):
                        values[idx] = cell_value(cell).strip()
                output.append({name: values[idx] for idx, name in headers.items()})
            return output
    return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--skip-image-hash", action="store_true", help="Skip expensive exact-content hashing; use only after a prior hash audit.")
    args = parser.parse_args()
    main(args.root, args.archive, args.json_out, args.skip_image_hash)
