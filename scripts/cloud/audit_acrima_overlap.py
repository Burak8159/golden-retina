#!/usr/bin/env python3
"""Integrity and exact-source-overlap audit for the official ACRIMA archive.

Only aggregate counts and file hashes are written to the output report. Images
and path-level records remain on the cloud VM.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

from PIL import Image


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(p for p in args.root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rows = []
    invalid = 0
    for path in paths:
        try:
            with Image.open(path) as image:
                image.load()
            rows.append((path, digest(path), 1 if "_g_" in path.name else 0))
        except Exception:
            invalid += 1
    counts = collections.Counter(label for _, _, label in rows)
    image_hashes = [value for _, value, _ in rows]
    by_hash = collections.Counter(image_hashes)
    duplicate_rows = sum(n - 1 for n in by_hash.values() if n > 1)
    reference_hashes: set[str] = set()
    reference_roots = []
    for root in args.reference_root:
        root_count = 0
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    try:
                        reference_hashes.add(digest(path))
                        root_count += 1
                    except OSError:
                        continue
        reference_roots.append({"root_name": root.name, "image_files_hashed": root_count})
    overlaps = sum(value in reference_hashes for _, value, _ in rows)
    report = {
        "dataset": "ACRIMA v1 · official Figshare article 7613135",
        "archive_sha256": digest(args.root.parents[2] / "Database.zip"),
        "files_seen": len(paths),
        "decode_valid_images": len(rows),
        "decode_failures": invalid,
        "class_counts_by_filename": {"glaucoma": counts.get(1, 0), "normal": counts.get(0, 0)},
        "within_dataset_exact_duplicate_groups": sum(n > 1 for n in by_hash.values()),
        "within_dataset_exact_duplicate_rows": duplicate_rows,
        "overlap_rows_with_reference_sources": overlaps,
        "reference_roots": reference_roots,
        "patient_ids_available": False,
        "boundary": "Filename labels are source-provided. The collection has no usable patient IDs in this archive; downstream holdout is image/hash-grouped and exploratory. Exact duplicates against prior source roots are counted, but near-duplicate/subject overlap cannot be excluded.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
