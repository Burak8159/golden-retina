#!/usr/bin/env python3
"""Create a local-only, SHA-pinned research API registry from RTX checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rvo-checkpoint", type=Path, required=True)
    ap.add_argument("--glaucoma-checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    records = [
        {
            "disease": "Retinal vein occlusion (RVO)",
            "source_label": "CLARUS700 RVO",
            "training_source": "CLARUS700 Figshare 2026 public release; one-center ZEISS CLARUS 700 UWF",
            "evidence_boundary": "Selection-exposed internal development; official test not accessed; not clinically validated.",
            "model_version": "clarus700-rvo-convnext-tiny-dev-20261127-e5",
            "architecture": "convnext_tiny",
            "labels": ["RVO"],
            "image_size": 224,
            "preprocessing": "resize_square",
            "checkpoint_path": str(a.rvo_checkpoint.resolve()),
            "checkpoint_sha256": sha256(a.rvo_checkpoint),
            "enabled": True,
        },
        {
            "disease": "Glaucoma",
            "source_label": "RIM-ONE DL hospital training partition",
            "training_source": "Official RIM-ONE DL hospital training partition; random initialization; exact-image grouped development split",
            "evidence_boundary": "Small internal image-hash-grouped development split without patient IDs; official test not accessed; not clinically validated.",
            "model_version": "rimone-glaucoma-mobilenet-v3-small-dev-20261127-e6",
            "architecture": "mobilenet_v3_small",
            "labels": ["glaucoma"],
            "image_size": 224,
            "preprocessing": "resize_center_crop",
            "checkpoint_path": str(a.glaucoma_checkpoint.resolve()),
            "checkpoint_sha256": sha256(a.glaucoma_checkpoint),
            "enabled": True,
        },
    ]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({"registry_version": 1, "models": records}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"registry": str(a.output), "models": [{k: x[k] for k in ("disease", "model_version", "checkpoint_sha256")} for x in records]}))


if __name__ == "__main__":
    main()
