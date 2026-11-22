#!/usr/bin/env python3
"""Summarize candidate dataset presence and schema on the authorized VM only.

The report deliberately excludes image names, patient IDs, label rows and file
contents. It prints aggregate counts and CSV headers so it is safe to capture
in a project log without exporting row-level medical data.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
SCHEMA_SUFFIXES = {".csv", ".tsv"}
MAX_FILES_PER_SOURCE = 100_000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data/golden-retina/workspace"))
    args = parser.parse_args()
    external = args.root / "external"
    if not external.is_dir():
        raise SystemExit(f"External data root not found: {external}")

    for source in sorted(p for p in external.iterdir() if p.is_dir()):
        image_count = 0
        schema_files: list[Path] = []
        docs: list[Path] = []
        for index, path in enumerate(source.rglob("*")):
            if index >= MAX_FILES_PER_SOURCE:
                break
            if not path.is_file():
                continue
            if path.suffix.lower() in IMAGE_SUFFIXES:
                image_count += 1
            elif path.suffix.lower() in SCHEMA_SUFFIXES:
                schema_files.append(path)
            elif path.name.lower() in {"readme", "readme.md", "license", "license.txt"}:
                docs.append(path)
        print(f"SOURCE\t{source.name}\timages={image_count}\tschemas={len(schema_files)}")
        for schema in sorted(schema_files)[:8]:
            try:
                with schema.open("r", encoding="utf-8-sig", newline="") as stream:
                    reader = csv.reader(stream, delimiter="\t" if schema.suffix.lower() == ".tsv" else ",")
                    header = next(reader, [])
                    row_count = sum(1 for _ in reader)
                safe_header = ",".join(header[:80]).replace("\t", " ")
                print(f"SCHEMA\t{source.name}/{schema.relative_to(source)}\trows={row_count}\tcolumns={safe_header}")
            except (OSError, UnicodeError, csv.Error, StopIteration) as exc:
                print(f"SCHEMA_ERROR\t{source.name}/{schema.relative_to(source)}\t{type(exc).__name__}")
        for document in sorted(docs)[:4]:
            try:
                for line in document.read_text(encoding="utf-8", errors="replace").splitlines():
                    lower = line.lower()
                    if any(word in lower for word in ("license", "licence", "citation", "patient", "label", "source", "version")):
                        print(f"DOC_HINT\t{source.name}/{document.relative_to(source)}\t{line.strip()[:240]}")
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
