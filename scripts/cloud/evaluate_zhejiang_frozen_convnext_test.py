#!/usr/bin/env python3
"""One-shot test evaluation for the frozen Zhejiang ConvNeXt checkpoint.

Validation-selected thresholds are loaded from the prior aggregate report.
The test partition is used only for scoring; this script never selects a model,
epoch, threshold, or preprocessing option from test outcomes. Row-level scores
are kept in memory and only aggregate metrics are written to the output JSON.
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


class TestImages(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image_path"]) as image:
            return self.transform(image.convert("RGB")), index


def summarize(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "eyes": int(len(y)), "positive_eyes": int(y.sum()),
        "negative_eyes": int(len(y) - y.sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
        "auroc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
        "auprc": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else None,
        "threshold_from_validation": float(threshold),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--validation-summary", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = ap.parse_args()

    with args.manifest.open(newline="", encoding="utf-8-sig") as f:
        all_rows = list(csv.DictReader(f))
    required = {"relative_image_path", "patient_group", "eye_id", "source_label", "split"}
    if not all_rows or not required.issubset(all_rows[0]):
        raise ValueError(f"Manifest must contain {sorted(required)}")
    split_by_patient: dict[str, set[str]] = defaultdict(set)
    for row in all_rows:
        split_by_patient[row["patient_group"]].add(row["split"])
    if any(len(splits) != 1 for splits in split_by_patient.values()):
        raise ValueError("Patient appears in multiple split partitions")

    # Do not resolve paths outside test; no train/validation image is read here.
    rows = [dict(row) for row in all_rows if row["split"] == "test"]
    if not rows:
        raise ValueError("No frozen test rows found")
    for row in rows:
        row["image_path"] = str(args.image_root / row["relative_image_path"])
        if not Path(row["image_path"]).is_file():
            raise FileNotFoundError("Unresolved frozen test image path")

    val_summary = json.loads(args.validation_summary.read_text(encoding="utf-8"))
    if val_summary.get("status") != "PARTIAL_CONVNEXT_FINE_TUNING_DISEASE_VS_ALL_VALIDATION_ONLY":
        raise ValueError("Unexpected threshold provenance; refusing test evaluation")
    thresholds = {k: float(v["threshold_validation_only"])
                  for k, v in val_summary["metrics"].items()}
    if set(thresholds) != set(LABELS):
        raise ValueError("Validation threshold labels do not match model labels")

    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(768, len(LABELS))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("labels") != LABELS:
        raise ValueError("Checkpoint disease label order mismatch")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    loader = DataLoader(TestImages(rows, weights.transforms()), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers,
                        pin_memory=device.type == "cuda")
    image_scores = np.zeros((len(rows), len(LABELS)), dtype=np.float32)
    with torch.inference_mode():
        for x, indexes in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits = model(x)
            image_scores[indexes.numpy()] = torch.sigmoid(logits.float()).cpu().numpy()

    eyes: dict[tuple[str, str], dict] = {}
    for row, scores in zip(rows, image_scores):
        key = (row["patient_group"], row["eye_id"])
        rec = eyes.setdefault(key, {"patient": row["patient_group"],
                                    "label": row["source_label"],
                                    "score": scores.copy()})
        if rec["label"] != row["source_label"]:
            raise ValueError("Conflicting source labels within an eye")
        rec["score"] = np.maximum(rec["score"], scores)
    eye_rows = list(eyes.values())
    reports = {}
    rng = np.random.default_rng(20261121)
    for j, label in enumerate(LABELS):
        y = np.asarray([int(e["label"] == label) for e in eye_rows])
        s = np.asarray([e["score"][j] for e in eye_rows])
        if len(np.unique(y)) != 2:
            raise ValueError(f"Test partition lacks one class for {label}")
        base = summarize(y, s, thresholds[label])
        by_patient: dict[str, list[int]] = defaultdict(list)
        for i, eye in enumerate(eye_rows):
            by_patient[eye["patient"]].append(i)
        pos_patients = [p for p, ix in by_patient.items() if any(y[i] for i in ix)]
        neg_patients = [p for p, ix in by_patient.items() if not any(y[i] for i in ix)]
        if not pos_patients or not neg_patients:
            raise ValueError(f"Patient bootstrap strata missing for {label}")
        vals = defaultdict(list)
        for _ in range(args.bootstrap_replicates):
            sampled = list(rng.choice(pos_patients, len(pos_patients), replace=True))
            sampled += list(rng.choice(neg_patients, len(neg_patients), replace=True))
            ix = [i for p in sampled for i in by_patient[p]]
            m = summarize(y[ix], s[ix], thresholds[label])
            for metric in ("accuracy", "sensitivity", "specificity", "auroc", "auprc"):
                if m[metric] is not None:
                    vals[metric].append(m[metric])
        base["patient_cluster_bootstrap_95ci"] = {
            metric: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
            for metric, v in vals.items() if v
        }
        reports[label] = base

    report = {
        "status": "FROZEN_PATIENT_GROUPED_TEST_EVALUATION",
        "task": "six one-vs-all logits trained against the source's mutually exclusive diagnosis classes and Healthy",
        "analysis_unit": "eye; max probability over images within eye",
        "threshold_provenance": "fixed from the previously saved validation-only aggregate report; never recomputed on test",
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "validation_summary_sha256": hashlib.sha256(args.validation_summary.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "test_eyes": len(eye_rows),
        "test_patients": len({e["patient"] for e in eye_rows}),
        "diseases": reports,
        "caveat": "Internal single-source patient-grouped test partition only. It does not establish external, clinical, prospective, or comorbidity performance; previous project exposure and same-source development limit independence.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
