"""Safely verify and extract the private Figshare Zhejiang UWF archive."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

archive = Path("/data/golden-retina/workspace/external/zhejiang_uwf/Dataset.zip")
out = archive.parent / "unpacked"
expected_size = 9_018_811_972
expected_md5 = "ddee42dc99792545654a65413c0fdb6f"
if archive.stat().st_size != expected_size:
    raise RuntimeError(f"Archive size mismatch: {archive.stat().st_size}")
digest = hashlib.md5()
with archive.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != expected_md5:
    raise RuntimeError(f"Archive MD5 mismatch: {digest.hexdigest()}")
with zipfile.ZipFile(archive) as zf:
    bad = [name for name in zf.namelist()
           if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts]
    if bad:
        raise RuntimeError(f"Unsafe archive paths: {bad[:5]}")
    # extractall reads every member and validates each member CRC as it writes.
    zf.extractall(out)
manifest = {
    "archive": str(archive), "bytes": archive.stat().st_size,
    "md5": digest.hexdigest(), "entries": len(zf.namelist()),
    "extracted_to": str(out), "status": "CHECKSUM_AND_ZIP_CRC_VERIFIED_AND_EXTRACTED",
}
(archive.parent / "archive_verification.json").write_text(json.dumps(manifest, indent=2)+"\n")
print(json.dumps(manifest, indent=2), flush=True)
