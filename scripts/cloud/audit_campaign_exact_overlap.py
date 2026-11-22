#!/usr/bin/env python3
"""Exact SHA-256 overlap audit for new cloud-staged fundus cohorts.

Outputs only aggregate overlap counts and a private path-level JSON report on
the VM.  The path-level report must not be copied into the public repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def inventory(root: Path) -> dict[str, list[str]]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        digest = hashlib.sha256()
        try:
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            continue
        by_hash[digest.hexdigest()].append(str(path))
    return by_hash


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    roots = {}
    for spec in args.root:
        name, raw = spec.split("=", 1)
        root = Path(raw)
        if root.exists():
            roots[name] = inventory(root)
    pairs = {}
    names = sorted(roots)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared = set(roots[left]).intersection(roots[right])
            if shared:
                pairs[f"{left}__{right}"] = {
                    "shared_hash_groups": len(shared),
                    "left_paths": sum(len(roots[left][h]) for h in shared),
                    "right_paths": sum(len(roots[right][h]) for h in shared),
                    "private_paths": {
                        h: {"left": roots[left][h], "right": roots[right][h]}
                        for h in sorted(shared)
                    },
                }
    report = {
        "method": "Exact byte-level SHA-256; does not detect recompression, crops, or augmentation lineage.",
        "roots": {
            name: {
                "unique_hashes": len(hashes),
                "file_paths": sum(map(len, hashes.values())),
                "within_root_duplicate_hash_groups": sum(1 for v in hashes.values() if len(v) > 1),
            }
            for name, hashes in roots.items()
        },
        "pairs_with_exact_overlap": pairs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"roots": report["roots"], "pairs": {
        k: {kk: vv for kk, vv in v.items() if kk != "private_paths"}
        for k, v in pairs.items()
    }}, indent=2))


if __name__ == "__main__":
    main()
