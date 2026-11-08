#!/usr/bin/env python3
"""Run the official OPENEye-FM head on the frozen validation rows only.

Row-level predictions stay on the private GPU VM. The public-facing summary
contains only aggregate metrics and source counts.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from inference import build_eval_transform, load_openeye_checkpoint
from models import openeye


class FundusRows(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image_path"]) as im:
            image = self.transform(im.convert("RGB"))
        return image, index


def safe_auc(y, score):
    return float(roc_auc_score(y, score)) if len(set(y)) == 2 else None


def metrics(y, score):
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    pred = score >= 0.5
    tn = int(((y == 0) & (~pred)).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & (~pred)).sum())
    tp = int(((y == 1) & pred).sum())
    return {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "accuracy_at_0_5": float(accuracy_score(y, pred)),
        "sensitivity_at_0_5": float(tp / (tp + fn)) if tp + fn else None,
        "specificity_at_0_5": float(tn / (tn + fp)) if tn + fp else None,
        "auroc": safe_auc(y, score),
        "auprc": float(average_precision_score(y, score)) if y.sum() else None,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "threshold": 0.5,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--labels-json", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    rows = [r for r in all_rows if r["split"] == "valid" and r["source"] in {"drtid_paired", "hygd_v1_1_0"}]
    if not rows:
        raise RuntimeError("No eligible DRTiD/HYGD validation rows; refusing to evaluate other splits.")
    # Explicitly refuse any test rows even if the supplied manifest changes.
    assert all(r["split"] == "valid" for r in rows)

    with open(args.labels_json, encoding="utf-8") as f:
        labels = list(json.load(f).keys())
    label_idx = {name: i for i, name in enumerate(labels)}
    required = ["moderate diabetic retinopathy", "severe diabetic retinopathy", "proliferative diabetic retinopathy", "glaucoma"]
    missing = sorted(set(required) - set(label_idx))
    if missing:
        raise RuntimeError(f"Required official output labels are missing: {missing}")

    model = openeye.vit_base_dinov2(
        img_size=224, num_classes=len(labels), drop_path_rate=0.0,
        global_pool="avg", dynamic_img_size=True,
    )
    epoch, ignored = load_openeye_checkpoint(model, args.checkpoint)
    model.cuda().eval()
    transform = build_eval_transform(224)
    loader = DataLoader(FundusRows(rows, transform), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        pin_memory=True)
    probs = np.zeros((len(rows), len(labels)), dtype=np.float32)
    with torch.inference_mode():
        for images, indexes in loader:
            logits = model(images.cuda(non_blocking=True))
            probs[indexes.numpy()] = logits.sigmoid().float().cpu().numpy()

    result_rows = []
    for i, row in enumerate(rows):
        p = probs[i]
        result_rows.append({
            "image_id": row["image_id"], "patient_id": row["patient_id"],
            "source": row["source"], "split": row["split"],
            "dr_grade": row["dr_grade"], "gon": row["gon"],
            "dr_referable_score": float(max(p[label_idx[x]] for x in required[:3])),
            "glaucoma_score": float(p[label_idx["glaucoma"]]),
            "dr_grade_prediction": int(np.argmax([p[label_idx[x]] for x in [
                "normal", "mild diabetic retinopathy", "moderate diabetic retinopathy",
                "severe diabetic retinopathy", "proliferative diabetic retinopathy"]])),
        })

    dr = [r for r in result_rows if r["source"] == "drtid_paired" and r["dr_grade"] not in {"", "-1"}]
    glaucoma = [r for r in result_rows if r["source"] == "hygd_v1_1_0" and r["gon"] not in {"", "-1"}]
    dr_y = [int(r["dr_grade"]) >= 2 for r in dr]
    dr_s = [r["dr_referable_score"] for r in dr]
    grade_y = [int(r["dr_grade"]) for r in dr]
    grade_p = [r["dr_grade_prediction"] for r in dr]
    g_y = [int(r["gon"]) for r in glaucoma]
    g_s = [r["glaucoma_score"] for r in glaucoma]
    summary = {
        "status": "FROZEN_VALIDATION_ONLY_EXPLORATORY_TRANSFER",
        "model": "official OPENEye-FM checkpoint and official inference transform",
        "checkpoint_epoch": epoch,
        "checkpoint_sha256": __import__("hashlib").sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "official_label_order": labels,
        "source_overlap_policy": "DeepDRiD and DDR validation omitted because listed as OPENEye-FM pretraining sources; DRTiD/HYGD retained, with exact-image overlap not ruled out because the released pretraining manifest was unavailable in this checkout.",
        "test_rows_loaded": 0,
        "rows": {"drtid_paired_valid": len(dr), "hygd_valid": len(glaucoma)},
        "referable_dr_ge2_drtid": metrics(dr_y, dr_s),
        "dr_grade_0_4_drtid": {"n": len(grade_y), "accuracy": float(accuracy_score(grade_y, grade_p)) if grade_y else None},
        "glaucoma_hygd": metrics(g_y, g_s),
        "model_load_ignored_checkpoint_keys": int(ignored),
        "interpretation": "Exploratory source-transfer benchmark only; scores and the fixed 0.5 threshold are not clinical diagnoses or a safety certificate.",
    }
    pred_path = outdir / "predictions_private.csv"
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(result_rows[0]))
        w.writeheader(); w.writerows(result_rows)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
