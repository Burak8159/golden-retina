#!/usr/bin/env python3
"""Stage selected CC BY retinal archives on the authorized VM only.

The script reads public Mendeley metadata, verifies the dataset-level license,
downloads only named source files to the supplied cloud directory, and emits an
aggregate SHA-256 manifest. It never prints URLs, image names, labels, or rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

UA = "GoldenRetinaResearch/1.0 (+https://data.mendeley.com/)"
SELECTIONS = {
    ("s9bfhswzjb", 1): {"Original Dataset.zip"},
    ("pc4mb3h8hz", 2): {"images.zip", "train_data.csv", "val_data.csv"},
    ("4k64fwnp4k", 5): {"Dataset_for_classification.zip"},
    # Official v4 CC BY 4.0 composite. Kept modality-separated during audit;
    # OCT must never enter the fundus-only classifier.
    ("trghs22fpg", 4): {"Dataset.zip"},
    # Recent three-hospital Bangladesh multi-disease source; keep this cohort
    # separate from the merged public-source datasets for a source-only audit.
    ("959whsbc6v", 1): {"Bangladesh Multi-Disease Retinal Dataset (BMRD).zip"},
    # 2026 disease/severity source. Original-vs-augmented lineage must be
    # audited before any partition is created or any fitting begins.
    ("z227d626vm", 1): {"Retinal Dataset.zip"},
}


def request(url: str, start: int | None = None):
    headers = {"User-Agent": UA, "Accept": "application/json,*/*"}
    if start:
        headers["Range"] = f"bytes={start}-"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90)


def file_digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, target: Path, expected_size: int):
    part = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, 7):
        offset = part.stat().st_size if part.exists() else 0
        try:
            with request(url, offset) as response:
                append = offset > 0 and response.status == 206
                if offset and not append:
                    offset = 0
                mode = "ab" if append else "wb"
                with part.open(mode) as stream:
                    while True:
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        stream.write(block)
            current = part.stat().st_size
            if current != expected_size:
                raise IOError(f"size mismatch {current}/{expected_size}")
            part.replace(target)
            return file_digest(target)
        except (OSError, urllib.error.URLError) as exc:
            print(json.dumps({"retry": attempt, "bytes_present": part.stat().st_size if part.exists() else 0,
                              "error_type": type(exc).__name__}), flush=True)
            if attempt == 6:
                raise
            time.sleep(min(5 * attempt, 30))
    raise RuntimeError("download retry loop exited unexpectedly")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for (dataset_id, version), selected_names in SELECTIONS.items():
        metadata_url = f"https://data.mendeley.com/public-api/datasets/{dataset_id}"
        with request(metadata_url) as response:
            record = json.load(response)
        record = record.get("data", record)
        if int(record.get("version", -1)) != version:
            raise RuntimeError(f"Unexpected published version for dataset {dataset_id}")
        license_name = str((record.get("data_licence") or {}).get("short_name", ""))
        if license_name != "CC BY 4.0":
            raise RuntimeError(f"Dataset {dataset_id} license did not match the recorded CC BY 4.0 gate")
        available = {str(item.get("filename")): item for item in record.get("files", [])}
        missing = selected_names - set(available)
        if missing:
            raise RuntimeError(f"Selected source files absent from published record {dataset_id}")
        destination = args.output_root / f"{dataset_id}_v{version}"
        destination.mkdir(parents=True, exist_ok=True)
        for filename in sorted(selected_names):
            item = available[filename]
            expected_size = int(item["size"])
            file_id = str(item["id"])
            target = destination / filename
            if target.exists() and target.stat().st_size == expected_size:
                digest = file_digest(target)
            else:
                download_url = (f"https://data.mendeley.com/public-files/datasets/{dataset_id}"
                                f"/files/{file_id}/file_downloaded?version={version}")
                digest = download(download_url, target, expected_size)
            manifest.append({"dataset_id": dataset_id, "version": version,
                             "file": filename, "bytes": expected_size,
                             "sha256": digest, "license": license_name})
            print(json.dumps({"dataset": dataset_id, "version": version,
                              "file": filename, "bytes": expected_size,
                              "sha256": digest}), flush=True)
    output = args.output_root / "download_manifest_aggregate.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "files": len(manifest),
                      "total_bytes": sum(item["bytes"] for item in manifest),
                      "manifest": str(output)}), flush=True)


if __name__ == "__main__":
    main()
