#!/usr/bin/env python3
"""Measure how well non-diagnostic file/acquisition metadata predicts Mendeley labels.

This is a shortcut diagnostic, not a disease model. Exact-file groups are kept
within a single partition; only aggregate metrics are written to the report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
TARGETS = {
    "amd": {"dry_amd", "wet_amd"},
    "diabetic_retinopathy": {"mild_dr", "moderate_dr", "severe_dr", "very_severe_npdr", "proliferative_dr", "advanced_pdr"},
    "glaucoma": {"glaucoma"},
    "cataract": {"cataract"},
    "hypertensive_retinopathy": {"hypertensive_retinopathy"},
    "pathological_myopia": {"pathological_myopia"},
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def metadata(path: Path) -> list[float]:
    with Image.open(path) as source:
        im = ImageOps.exif_transpose(source)
        width, height = im.size
        mode = im.mode
        arr = np.asarray(im.convert("RGB").resize((32, 32)), dtype=np.float32) / 255.0
    means = arr.mean((0, 1))
    stds = arr.std((0, 1))
    return [
        np.log1p(width), np.log1p(height), width / max(1, height),
        np.log1p(path.stat().st_size),
        float(mode == "RGB"), float(mode == "RGBA"), float(mode == "L"),
        *means.tolist(), *stds.tolist(),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--exclude-hash-file", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    excluded = set(args.exclude_hash_file.read_text().split())
    rows = []
    failures = 0
    excluded_rows = 0
    for folder in sorted(p for p in args.data.iterdir() if p.is_dir()):
        name = folder.name.lower()
        if name in {"pterygium", "pt"}:
            continue
        for path in folder.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in EXT:
                continue
            try:
                digest = sha(path)
                if digest in excluded:
                    excluded_rows += 1
                    continue
                rows.append((name, digest, metadata(path)))
            except Exception:
                failures += 1
    names = sorted({r[0] for r in rows})
    folders = {n: i for i, n in enumerate(names)}
    yfolder = np.asarray([folders[r[0]] for r in rows])
    groups = np.asarray([r[1] for r in rows])
    x = np.asarray([r[2] for r in rows], dtype=np.float32)
    # Reproduce the image-grouped split policy used for the specialist models.
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=8159)
    tv, test = next(outer.split(np.zeros(len(yfolder)), yfolder, groups))
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=8160)
    train, _ = next(inner.split(np.zeros(len(tv)), yfolder[tv], groups[tv]))
    train = tv[train]
    results = {}
    for task, positive_folders in TARGETS.items():
        pos_ids = {folders[n] for n in positive_folders if n in folders}
        if not pos_ids:
            results[task] = {"status": "unsupported"}
            continue
        yy = np.isin(yfolder, list(pos_ids)).astype(np.int8)
        if len(np.unique(yy[train])) != 2 or len(np.unique(yy[test])) != 2:
            results[task] = {"status": "single_class_partition"}
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=400, random_state=8159))
        clf.fit(x[train], yy[train])
        score = clf.predict_proba(x[test])[:, 1]
        pred = score >= 0.5
        results[task] = {
            "status": "complete", "n_test": int(len(test)),
            "positive_test": int(yy[test].sum()),
            "metadata_only_balanced_accuracy_at_0_5": float(balanced_accuracy_score(yy[test], pred)),
            "metadata_only_auroc": float(roc_auc_score(yy[test], score)),
        }
    groups_by_part = {k: set(groups[v]) for k, v in {"train": train, "test": test}.items()}
    report = {
        "dataset": "Mendeley 6ry9n79gb4.1 13-folder compilation",
        "purpose": "shortcut diagnostic using file size, dimensions, mode, and low-resolution color statistics only",
        "n_images": len(rows), "n_exact_groups": len(set(groups)),
        "cross_source_exact_hash_excluded": excluded_rows, "decode_or_metadata_failures": failures,
        "image_group_disjoint": not bool(groups_by_part["train"] & groups_by_part["test"]),
        "folder_names": names, "tasks": results,
        "interpretation_boundary": "Metadata-only predictability is evidence of acquisition/compilation confounding. Low predictability does not rule out visual shortcuts, source leakage, patient overlap, or clinical non-generalization. No patient IDs or source-to-image map are provided.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
