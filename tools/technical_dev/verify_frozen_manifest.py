#!/usr/bin/env python3
"""Verify a staged train/validation manifest and its image bytes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True,
                        help="Workspace root used to resolve relative image paths.")
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-rows", type=int, default=7671)
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    root = args.root.resolve()
    manifest_hash = sha256_file(manifest)
    rows = []
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))

    split_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    missing = 0
    hash_mismatch = 0
    verified = 0
    total_image_bytes = 0
    for row in rows:
        split_counts[row["split"]] += 1
        source_counts[f'{row["source"]}/{row["split"]}'] += 1
        image = Path(row["image_path"])
        if not image.is_absolute():
            image = root / image
        if not image.is_file():
            missing += 1
            continue
        total_image_bytes += image.stat().st_size
        if sha256_file(image) != row["image_sha256"]:
            hash_mismatch += 1
        else:
            verified += 1

    ok = (
        manifest_hash == args.expected_manifest_sha256
        and len(rows) == args.expected_rows
        and set(split_counts) <= {"train", "valid"}
        and split_counts.get("train") == 6793
        and split_counts.get("valid") == 878
        and missing == 0
        and hash_mismatch == 0
        and verified == len(rows)
    )
    summary = {
        "status": "PASS" if ok else "FAIL",
        "manifest_sha256": manifest_hash,
        "expected_manifest_sha256": args.expected_manifest_sha256,
        "manifest_rows": len(rows),
        "expected_rows": args.expected_rows,
        "split_counts": dict(sorted(split_counts.items())),
        "source_split_counts": dict(sorted(source_counts.items())),
        "image_files_verified_by_sha256": verified,
        "missing_images": missing,
        "image_hash_mismatches": hash_mismatch,
        "image_bytes_verified": total_image_bytes,
        "test_rows": split_counts.get("test", 0),
        "manifest_path": str(manifest),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
