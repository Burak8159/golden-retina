#!/usr/bin/env python3
"""One-shot cross-source DR transfer diagnostic; writes aggregate metrics only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class Rows(Dataset):
    def __init__(self, items):
        self.items = items
        self.tf = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        with Image.open(path) as im:
            x = self.tf(ImageOps.exif_transpose(im).convert("RGB"))
        return x, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    split = {}
    with args.predictions.open(newline="") as f:
        for row in csv.DictReader(f):
            split[row["image_id"]] = int(row["true_grade"])
    # Transfer endpoint is any DR grade vs grade 0; the model's Mendeley target
    # is any named DR folder vs all other labeled folders.
    items = []
    with args.metadata.open(newline="") as f:
        for row in csv.DictReader(f):
            image_id = str(row["image_id"]).strip()
            if image_id not in split:
                continue
            path = args.images / f"{image_id}.jpg"
            if path.is_file():
                items.append((path, int(split[image_id] > 0)))
    if not items or len({y for _, y in items}) < 2:
        raise RuntimeError("held-out FDR26 rows or both classes are unavailable")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.cuda().eval()
    ys, ps = [], []
    with torch.inference_mode():
        for x, y in DataLoader(Rows(items), batch_size=args.batch_size, num_workers=4, pin_memory=True):
            with torch.autocast("cuda", dtype=torch.float16):
                z = model(x.cuda(non_blocking=True))
            ys.extend(y.numpy().tolist())
            ps.extend(torch.softmax(z.float(), 1)[:, 1].cpu().numpy().tolist())
    y = np.asarray(ys, dtype=np.int8)
    p = np.asarray(ps, dtype=np.float64)
    pred = p >= args.threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    report = {
        "status": "complete",
        "source_train": "Mendeley 6ry9n79gb4.1, mixed-source folders, exact-image-group split",
        "target_evaluation": "FDR26 held-out patient-key split already used in project; secondary cross-source diagnostic, not a pristine validation",
        "endpoint": "any DR (FDR26 grade > 0) vs no DR",
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "threshold": args.threshold,
        "n": int(len(y)), "positive": int(y.sum()), "negative": int(len(y) - y.sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(tp / max(1, tp + fn)), "specificity": float(tn / max(1, tn + fp)),
        "false_negatives": int(fn), "false_positives": int(fp),
        "auroc": float(roc_auc_score(y, p)),
        "limitation": "FDR26 test labels have been exposed in prior project development; this is a new model's cross-source diagnostic only. It cannot establish independent external validation. No row-level scores or IDs are written.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
