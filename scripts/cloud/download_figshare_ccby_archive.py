#!/usr/bin/env python3
"""Download one verified CC BY Figshare archive to cloud storage only."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def request(url: str, start: int = 0):
    headers = {"User-Agent": "GoldenRetinaResearch/1.0", "Accept": "application/json,*/*"}
    if start:
        headers["Range"] = f"bytes={start}-"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--article-id", type=int, required=True)
    ap.add_argument("--expected-doi", required=True)
    ap.add_argument("--expected-license", default="CC BY 4.0")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    metadata = json.load(request(f"https://api.figshare.com/v2/articles/{args.article_id}"))
    if str(metadata.get("doi", "")).lower() != args.expected_doi.lower():
        raise SystemExit("Figshare DOI did not match the specified record")
    license_name = str((metadata.get("license") or {}).get("name", ""))
    if license_name != args.expected_license:
        raise SystemExit("Published license did not match the required gate")
    files = metadata.get("files", [])
    if len(files) != 1:
        raise SystemExit("Expected exactly one source archive")
    item = files[0]
    size = int(item["size"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / Path(item["name"]).name
    part = target.with_suffix(target.suffix + ".part")
    if target.exists() and target.stat().st_size == size:
        sha = digest(target)
    else:
        for attempt in range(1, 7):
            offset = part.stat().st_size if part.exists() else 0
            try:
                with request(item["download_url"], offset) as response:
                    append = offset > 0 and response.status == 206
                    if offset and not append:
                        offset = 0
                    with part.open("ab" if append else "wb") as out:
                        while True:
                            chunk = response.read(8 * 1024 * 1024)
                            if not chunk:
                                break
                            out.write(chunk)
                if part.stat().st_size != size:
                    raise IOError("Downloaded byte count differs from official record")
                part.replace(target)
                sha = digest(target)
                break
            except OSError:
                if attempt == 6:
                    raise
                time.sleep(min(5 * attempt, 30))
    record = {"article_id": args.article_id, "doi": metadata["doi"], "title": metadata["title"],
              "license": license_name, "archive_bytes": size, "sha256": sha,
              "file": target.name, "source": "official Figshare API", "directory": str(args.output_dir)}
    (args.output_dir / "download_manifest_aggregate.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("article_id", "doi", "license", "archive_bytes", "sha256", "source")}))


if __name__ == "__main__":
    main()
