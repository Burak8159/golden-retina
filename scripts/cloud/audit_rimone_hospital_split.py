#!/usr/bin/env python3
"""Extract and audit only the official RIM-ONE DL hospital partition on the VM."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from PIL import Image
import numpy as np

PREFIX = "RIM-ONE_DL_images/partitioned_by_hospital/"
CLASSES = {"normal": 0, "glaucoma": 1}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path: Path) -> int:
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((9, 8)))
    return sum((int(a[y, x] > a[y, x + 1]) << (y * 8 + x)) for y in range(8) for x in range(8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--destination", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    archive = args.archive.resolve()
    dest = args.destination.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    with ZipFile(archive) as z:
        infos = z.infolist()
        total_uncompressed = sum(i.file_size for i in infos)
        if total_uncompressed > 1_000_000_000:
            raise RuntimeError(f"Unexpected archive expansion size: {total_uncompressed}")
        chosen = []
        for info in infos:
            name = info.filename
            posix = PurePosixPath(name)
            if posix.is_absolute() or ".." in posix.parts:
                raise RuntimeError(f"Unsafe archive path: {name}")
            if name.startswith(PREFIX) and not info.is_dir():
                chosen.append(info)
        if not chosen:
            raise RuntimeError("No files found under official hospital partition")
        for info in chosen:
            rel = PurePosixPath(info.filename).relative_to(PREFIX)
            target = dest.joinpath(*rel.parts).resolve()
            if dest not in target.parents:
                raise RuntimeError(f"Path escaped extraction root: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, target.open("wb") as out:
                while chunk := src.read(1024 * 1024):
                    out.write(chunk)

    rows = []
    invalid = []
    for partition in ("training_set", "test_set"):
        for cls, label in CLASSES.items():
            folder = dest / partition / cls
            files = sorted(p for p in folder.iterdir() if p.is_file()) if folder.is_dir() else []
            for path in files:
                try:
                    with Image.open(path) as im:
                        im.verify()
                    with Image.open(path) as im:
                        width, height = im.size
                        fmt = im.format
                except Exception as exc:
                    invalid.append((str(path), type(exc).__name__))
                    continue
                rows.append({
                    "relative_path": path.relative_to(dest).as_posix(),
                    "label": label,
                    "label_name": cls,
                    "official_partition": partition,
                    "width": width,
                    "height": height,
                    "format": fmt,
                    "sha256": sha256(path),
                    "dhash64": f"{dhash(path):016x}",
                })
    if invalid:
        raise RuntimeError(f"Invalid images: {len(invalid)}")
    counts = Counter((r["official_partition"], r["label_name"]) for r in rows)
    expected = {("training_set", "normal"): 195, ("training_set", "glaucoma"): 116,
                ("test_set", "normal"): 118, ("test_set", "glaucoma"): 56}
    if counts != Counter(expected):
        raise RuntimeError(f"Unexpected official hospital split counts: {dict(counts)}")

    hash_groups = defaultdict(list)
    for r in rows:
        hash_groups[r["sha256"]].append(r)
    exact_groups = [g for g in hash_groups.values() if len(g) > 1]
    cross_partition_exact = [g for g in exact_groups if len({x["official_partition"] for x in g}) > 1]

    # Near-duplicate candidates from perceptual hash; flag only, never silently remove.
    train_rows = [r for r in rows if r["official_partition"] == "training_set"]
    test_rows = [r for r in rows if r["official_partition"] == "test_set"]
    near = []
    for tr in train_rows:
        a = int(tr["dhash64"], 16)
        for te in test_rows:
            if (a ^ int(te["dhash64"], 16)).bit_count() <= 4:
                near.append((tr["relative_path"], te["relative_path"]))

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    manifest_hash = sha256(args.manifest)
    summary = {
        "status": "RIM_ONE_DL_HOSPITAL_PARTITION_AUDITED",
        "source": "miag-ull/rim-one-dl official repository; hospital partition variant",
        "archive_name": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "archive_uncompressed_bytes_all_variants": total_uncompressed,
        "extracted_partition": "partitioned_by_hospital only",
        "image_count": len(rows),
        "counts": {f"{p}/{c}": n for (p, c), n in sorted(counts.items())},
        "invalid_images": len(invalid),
        "exact_duplicate_groups": len(exact_groups),
        "cross_partition_exact_duplicate_groups": len(cross_partition_exact),
        "near_duplicate_cross_partition_candidates_dhash_le4": len(near),
        "patient_ids_available_in_release": False,
        "patient_disjointness_verified": False,
        "hospital_partition_means": "training hospital(s) versus held-out hospital(s), following source description; not patient-level confirmation",
        "manifest_sha256_private": manifest_hash,
        "manifest_rows": len(rows),
        "terms": "Research and education use only; redistribution prohibited per source README. Dataset/images/checkpoint and row-level manifest remain private.",
        "threshold_and_model_selection": "No model trained; official held-out hospital partition remains sealed for one-pass evaluation.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
