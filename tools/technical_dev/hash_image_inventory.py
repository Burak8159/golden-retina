#!/usr/bin/env python3
"""Create a local-only, path-free SHA-256 inventory for image deduplication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def hash_file(item: tuple[str, Path]) -> tuple[str, str, int]:
    source, path = item
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return source, digest.hexdigest(), size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("external"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    args = parser.parse_args()

    files: list[tuple[str, Path]] = []
    for path in args.root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            relative = path.relative_to(args.root)
            source = relative.parts[0] if len(relative.parts) > 1 else "."
            files.append((source, path))
    files.sort(key=lambda item: str(item[1]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("source_folder", "sha256", "bytes"))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(hash_file, item) for item in files]
            for future in as_completed(futures):
                writer.writerow(future.result())
                written += 1
                if written % 5000 == 0:
                    target.flush()
                    print(f"hashed {written}/{len(files)}", flush=True)
    print(f"hashed {written} images; path-free inventory: {args.output}", flush=True)


if __name__ == "__main__":
    main()
