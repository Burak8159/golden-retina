#!/usr/bin/env python3
"""Cloud-only aggregate pHash audit for cross-partition similarity.

No paths, labels per image, or pair identities are written to the report.
This is a similarity screen, not proof of patient identity.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

sys.path.insert(0, "src")
from golden_retina.train_uwf_tumor_exploratory import scan, split  # noqa: E402


def phash(path: Path) -> int:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L").resize((32, 32), Image.Resampling.LANCZOS)
        pixels = np.asarray(im, dtype=np.float32)
    coeff = dctn(pixels, norm="ortho")[:8, :8].ravel()
    median = np.median(coeff[1:])
    value = 0
    for bit, coefficient in enumerate(coeff):
        value |= int(coefficient > median) << bit
    return value


def main() -> None:
    root = Path(sys.argv[1])
    out = Path(sys.argv[2])
    rows, _, scan_audit = scan(root)
    parts = split(rows)
    hashed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, records in parts.items():
        vals = np.fromiter((phash(row["path"]) for row in records), dtype=np.uint64, count=len(records))
        labels = np.fromiter((row["label"] for row in records), dtype=np.int8, count=len(records))
        hashed[name] = vals, labels

    byte_lut = np.fromiter((i.bit_count() for i in range(256)), dtype=np.uint8, count=256)
    aggregate: dict[str, dict[str, int]] = {}
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        left, left_labels = hashed[first]
        right, right_labels = hashed[second]
        counts = Counter()
        for start in range(0, len(left), 64):
            query = left[start : start + 64]
            xor = np.bitwise_xor(query[:, None], right[None, :])
            distances = np.zeros(xor.shape, dtype=np.uint8)
            for shift in range(0, 64, 8):
                distances += byte_lut[((xor >> shift) & 255).astype(np.uint8)]
            same = left_labels[start : start + len(query), None] == right_labels[None, :]
            for cutoff in (0, 2, 4, 6, 8):
                close = distances <= cutoff
                counts[f"pairs_hamming_le_{cutoff}"] += int(close.sum())
                counts[f"same_class_pairs_hamming_le_{cutoff}"] += int((close & same).sum())
                counts[f"different_class_pairs_hamming_le_{cutoff}"] += int((close & ~same).sum())
        aggregate[f"{first}_vs_{second}"] = dict(counts)

    report = {
        "dataset": "UWF Fundus Images of Intraocular Tumors",
        "image_counts": {name: len(values[0]) for name, values in hashed.items()},
        "pHash": "64-bit DCT perceptual hash on grayscale 32x32; aggregate pair counts only",
        "cross_partition_similarity": aggregate,
        "limitations": "Near perceptual similarity neither establishes patient identity nor proves leakage. The source has repeated-eye visits but no patient/eye IDs. Exact duplicate groups were already kept within partitions.",
        "scan_summary": {key: scan_audit[key] for key in ("eligible_unique_images", "exact_duplicate_groups", "exact_duplicate_rows_excluded", "cross_label_exact_hash_images_excluded", "patient_and_eye_ids_available")},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"saved": str(out), "image_counts": report["image_counts"], "cross_partition_similarity": aggregate}, sort_keys=True))


if __name__ == "__main__":
    main()
