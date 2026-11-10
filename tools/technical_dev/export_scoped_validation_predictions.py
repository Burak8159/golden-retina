#!/usr/bin/env python3
"""Export private validation predictions for the scoped ConvNeXt development run.

The row-level CSV is an internal audit artifact. Do not publish it; only share
aggregate summaries permitted by the source dataset terms.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from golden_retina.train_scoped_retina_heads import (
    RetinaRows,
    ScopedRetinaNet,
    infer,
    read_prepared_rows,
)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    root = args.root.resolve()
    rows = [row for row in read_prepared_rows(root, args.manifest)
            if row["split"] == "valid"]
    if not rows:
        raise ValueError("The prepared manifest contains no validation rows")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ScopedRetinaNet().to(device)
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    loader = DataLoader(
        RetinaRows(rows, args.image_size, False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    dr, quality, _lesions, glaucoma = infer(model, loader, device)
    dr = dr - dr.max(axis=1, keepdims=True)
    dr_probability = np.exp(dr)
    dr_probability /= dr_probability.sum(axis=1, keepdims=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "source", "split", "image_id", "patient_id", "patient_group_verified",
            "dr_grade_true", "dr_grade_pred", "dr_probability_referable_ge_2",
            "quality_ungradable_true", "quality_probability", "gon_true", "gon_probability",
        ])
        for i, row in enumerate(rows):
            writer.writerow([
                row["source"], row["split"], row["image_id"], row["patient_id"],
                str(bool(row["patient_group_verified"])).lower(), row["dr_grade"],
                int(dr_probability[i].argmax()), float(dr_probability[i, 2:].sum()),
                row["ungradable"], float(sigmoid(quality[i])), row["gon"],
                float(sigmoid(glaucoma[i])),
            ])
    print(f"Exported {len(rows)} private validation rows to {args.output}")


if __name__ == "__main__":
    main()
