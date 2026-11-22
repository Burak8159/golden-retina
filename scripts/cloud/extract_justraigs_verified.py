#!/usr/bin/env python3
"""Verify and sequentially extract the public JustRAIGS archive shards on VM."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

BASE = Path("/data/golden-retina/workspace/external/justraigs_20261122")
EXPECTED = {
    "JustRAIGS_Train_0.zip": ("b77f8679cc02a2fafe9953abff89fafb", 9264600703),
    "JustRAIGS_Train_1.zip": ("fd21d85eafb54bd1d3f6190c739246d0", 9327768021),
    "JustRAIGS_Train_2.zip": ("aff8fb52e068fc1050e6a8f32238a638", 9343376218),
    "JustRAIGS_Train_3.zip": ("a411fd5cd1cebdaeedb5727204abdfba", 9300847661),
    "JustRAIGS_Train_4.zip": ("0fb8cd04c0d39a93ce97f05ec4b109e4", 9255450216),
    "JustRAIGS_Train_5.zip": ("76fa9a9f131a866054206392f7f1bfc6", 7683877847),
}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    archives = BASE / "archives"
    images = BASE / "images"
    images.mkdir(parents=True, exist_ok=True)
    report_path = Path("/data/golden-retina/workspace/results/justraigs_extract_all_verified.json")
    reports = json.loads(report_path.read_text()) if report_path.exists() else []
    done = {r["archive"] for r in reports if r.get("status") == "extracted_and_crc_checked"}
    for name, (expected_md5, expected_size) in EXPECTED.items():
        if name in done:
            continue
        archive = archives / name
        if not archive.exists():
            print("MISSING", name, flush=True)
            continue
        if archive.stat().st_size != expected_size:
            print("SIZE_NOT_FINAL", name, archive.stat().st_size, flush=True)
            continue
        digest = md5(archive)
        if digest != expected_md5:
            raise RuntimeError(f"Official MD5 mismatch for {name}: {digest}")
        target = images / name.removesuffix(".zip")
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            members = zf.namelist()
            bad = zf.testzip()
            if bad:
                raise RuntimeError(f"CRC failure in {name}: {bad}")
            zf.extractall(target)
        image_count = sum(1 for m in members if Path(m).suffix.lower() in IMAGE_SUFFIXES)
        extracted_count = sum(1 for p in target.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not image_count or extracted_count != image_count:
            raise RuntimeError(f"Image count mismatch for {name}: archive={image_count}, extracted={extracted_count}")
        row = {"archive": name, "md5": digest, "verified_size": expected_size,
               "zip_members": len(members), "image_members": image_count,
               "extracted_images": extracted_count, "extracted_under": str(target),
               "status": "extracted_and_crc_checked"}
        reports.append(row)
        report_path.write_text(json.dumps(reports, indent=2) + "\n")
        print("EXTRACTED", json.dumps(row), flush=True)
        archive.unlink()
        print("ARCHIVE_REMOVED_AFTER_VERIFY", name, flush=True)
    print("DONE", json.dumps({"verified_shards": len(reports)}), flush=True)


if __name__ == "__main__":
    main()
