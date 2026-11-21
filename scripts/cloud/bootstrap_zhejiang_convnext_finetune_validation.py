#!/usr/bin/env python3
"""Aggregate validation-only scores and patient-cluster intervals for a saved run.

The script reads only validation image rows and writes no per-eye predictions.
Thresholds are selected on validation, so bootstrap intervals are conditional
and do not remove selection optimism.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

LABELS = ["AMD", "DR", "PM", "RD", "RVO", "Uveitis"]


class ValidationRows(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        with Image.open(row["image_path"]) as image:
            return self.transform(image.convert("RGB")), i


def metrics(y, score, threshold):
    pred = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "auroc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
        "auprc": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else None,
    }


def threshold_at_specificity(y, score, target=0.90):
    candidates = np.r_[np.unique(score), np.nextafter(float(score.max()), np.inf)]
    eligible = []
    for t in candidates:
        m = metrics(y, score, t)
        if m["specificity"] >= target - 1e-12:
            eligible.append((m["sensitivity"], m["specificity"], float(t)))
    return max(eligible)[2]


def bootstrap(by_patient, pos_patients, neg_patients, threshold, replicates, rng):
    keys = ("accuracy", "sensitivity", "specificity", "auroc", "auprc")
    values = defaultdict(list)
    for _ in range(replicates):
        sampled = list(rng.choice(pos_patients, len(pos_patients), replace=True))
        sampled += list(rng.choice(neg_patients, len(neg_patients), replace=True))
        records = [eye for patient in sampled for eye in by_patient[patient]]
        y = np.asarray([x["target"] for x in records], dtype=int)
        score = np.asarray([x["score"] for x in records], dtype=float)
        for key, value in metrics(y, score, threshold).items():
            if value is not None and np.isfinite(value):
                values[key].append(value)
    return {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
            for k, v in values.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bootstrap-replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261121)
    args = ap.parse_args()

    with args.manifest.open(newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "validation"]
    if not rows:
        raise ValueError("No validation rows found")
    for r in rows:
        r["image_path"] = str(args.image_root / r["relative_image_path"])
        if not Path(r["image_path"]).is_file():
            raise FileNotFoundError("Validation image path unresolved")

    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(768, len(LABELS))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ckpt.get("labels") != LABELS:
        raise ValueError("Checkpoint label order mismatch")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    loader = DataLoader(ValidationRows(rows, weights.transforms()), batch_size=8,
                        shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    row_scores = np.empty((len(rows), len(LABELS)), dtype=np.float32)
    with torch.inference_mode():
        for x, idx in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=device.type == "cuda"):
                scores = torch.sigmoid(model(x)).float().cpu().numpy()
            row_scores[np.asarray(idx)] = scores

    eye_map = {}
    for row, score in zip(rows, row_scores):
        key = (row["patient_group"], row["eye_id"])
        item = eye_map.setdefault(key, {"patient": row["patient_group"],
            "label": row["source_label"], "score": score.copy()})
        if item["label"] != row["source_label"]:
            raise ValueError("Conflicting labels within eye")
        item["score"] = np.maximum(item["score"], score)
    eyes = list(eye_map.values())
    rng = np.random.default_rng(args.seed)
    result = {
        "status": "VALIDATION_ONLY_PATIENT_CLUSTER_BOOTSTRAP",
        "unit": "eye; max score pooled across images within eye",
        "validation_eyes": len(eyes),
        "patient_count": len({x["patient"] for x in eyes}),
        "bootstrap_replicates": args.bootstrap_replicates,
        "threshold_policy": "selected on this validation cohort to maximize sensitivity at specificity >=0.90",
        "interval_policy": "patient-stratified cluster bootstrap; intervals are conditional on the selected threshold and do not correct model/threshold selection optimism",
        "checkpoint_epoch": int(ckpt["epoch"]),
        "best_validation_loss": float(ckpt["validation_loss"]),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "diseases": {},
    }
    for j, label in enumerate(LABELS):
        by_patient = defaultdict(list)
        for eye in eyes:
            by_patient[eye["patient"]].append({
                "target": int(eye["label"] == label), "score": float(eye["score"][j])})
        pos = [p for p, rs in by_patient.items() if rs[0]["target"] == 1]
        neg = [p for p, rs in by_patient.items() if rs[0]["target"] == 0]
        if not pos or not neg:
            raise ValueError(f"Missing positive or negative patient group for {label}")
        records = [x for rs in by_patient.values() for x in rs]
        y = np.asarray([x["target"] for x in records], dtype=int)
        score = np.asarray([x["score"] for x in records], dtype=float)
        threshold = threshold_at_specificity(y, score)
        point = metrics(y, score, threshold)
        fn = int(np.sum((y == 1) & (score < threshold)))
        fp = int(np.sum((y == 0) & (score >= threshold)))
        result["diseases"][label] = {
            **point, "positive_eyes": int(y.sum()), "misses": fn,
            "false_positives": fp, "threshold_validation_only": threshold,
            "positive_patients": len(pos), "negative_patients": len(neg),
            "patient_cluster_95pct_ci": bootstrap(by_patient, pos, neg, threshold,
                                                    args.bootstrap_replicates, rng),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
