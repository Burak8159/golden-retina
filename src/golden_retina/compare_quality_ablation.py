from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from train_scoped_retina_heads import (
    RetinaRows,
    ScopedRetinaNet,
    build_rows,
    deduplicate_exact_images,
    infer,
    summarize,
    validation_loss,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only, matched-loss quality-ablation comparison.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    root = args.root.resolve()
    rows, duplicate_audit = deduplicate_exact_images(build_rows(
        root,
        args.manifest,
        Path("results/mmrdr_uwf_manifest_20261013.csv"),
        Path("external/MMRDR/MMRDR-UWF/UWF.csv"),
        Path("external/HYGD_v1_1_0"),
        seed=20261107,
        include_mmrdr=False,
    ))
    valid_rows = [row for row in rows if row["split"] == "valid"]
    manifest_text = io.StringIO(newline="")
    writer = csv.DictWriter(manifest_text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    dataset = RetinaRows(valid_rows, args.size, False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0)

    train_rows = [row for row in rows if row["split"] == "train"]
    counts = np.bincount([row["dr_grade"] for row in train_rows if row["dr_grade"] >= 0], minlength=5)
    class_weights = np.sqrt(counts.sum() / np.maximum(counts, 1))
    class_weights /= class_weights.mean()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Matched checkpoint comparison requires the authorized L4 CUDA device")
    losses = (
        nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device, dtype=torch.float32)),
        nn.BCEWithLogitsLoss(),
        nn.BCEWithLogitsLoss(),
        nn.BCEWithLogitsLoss(),
        False,
    )
    result = {
        "status": "VALIDATION_ONLY_NO_TEST_INFERENCE",
        "split": "valid",
        "n_validation_images": len(valid_rows),
        "patient_grouping_note": "Source manifest grouping retained; DDR identity remains unverified.",
        "resolved_split_manifest_sha256": hashlib.sha256(
            manifest_text.getvalue().encode("utf-8")
        ).hexdigest(),
        "duplicate_audit": duplicate_audit,
        "selection_loss": "mean of DR-grade, lesion, and glaucoma task losses; quality loss excluded for both models",
        "models": {},
    }
    for name, path in (("multitask_baseline", args.baseline), ("no_quality_supervision", args.ablation)):
        model = ScopedRetinaNet().to(device)
        checkpoint = torch.load(root / path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        result["models"][name] = {
            "checkpoint": str(path),
            "checkpoint_sha256": hashlib.sha256((root / path).read_bytes()).hexdigest(),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "disease_only_validation_loss": validation_loss(model, loader, device, losses),
            "validation_metrics": summarize(valid_rows, infer(model, loader, device), include_quality=False),
        }
        del model, checkpoint
        torch.cuda.empty_cache()

    output = root / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
