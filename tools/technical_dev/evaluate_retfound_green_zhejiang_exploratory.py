#!/usr/bin/env python3
"""Exploratory RETFound-Green transfer probe on previously exposed Zhejiang UWF.

The endpoint is source-defined DR diagnosis versus healthy control, not
referable-DR grade. No binary referral threshold is applied. Row predictions
must remain private; aggregate metrics are written separately.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

import timm
from train_retfound_green_adaptation import MultiTask


class ZhejiangRows(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.transform = T.Compose([
            T.Resize((392, 392), antialias=True),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        with Image.open(row["image_path"]) as image:
            return self.transform(image.convert("RGB")), i


def metrics(y, score):
    return {
        "n_patients": int(len(y)),
        "positive_DR_patients": int(np.sum(y)),
        "healthy_patients": int(len(y) - np.sum(y)),
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--base-checkpoint", type=Path, required=True)
    ap.add_argument("--adapted-checkpoint", type=Path, required=True)
    ap.add_argument("--predictions-private", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--boots", type=int, default=5000)
    args = ap.parse_args()

    all_rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8-sig")))
    required = {"image_path", "patient_id", "diagnosis"}
    if not all_rows or not required.issubset(all_rows[0]):
        raise ValueError(f"Manifest requires {sorted(required)}")
    rows = [r for r in all_rows if r["diagnosis"] in {"DR", "Healthy"}]
    if not rows or any(not Path(r["image_path"]).is_file() for r in rows):
        raise FileNotFoundError("Missing DR/Healthy images in frozen manifest")
    p_labels = {}
    for r in rows:
        p_labels.setdefault(r["patient_id"], set()).add(r["diagnosis"])
    if any(len(v) != 1 for v in p_labels.values()):
        raise ValueError("Conflicting DR/Healthy labels within patient")

    base = timm.create_model("vit_small_patch14_reg4_dinov2", img_size=392, num_classes=0)
    base_state = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)
    missing = base.load_state_dict(base_state, strict=False)
    if missing.missing_keys or missing.unexpected_keys:
        raise RuntimeError("Base RETFound-Green checkpoint key mismatch")
    base.global_pool = "avg"
    ckpt = torch.load(args.adapted_checkpoint, map_location="cpu", weights_only=True)
    source_map = ckpt.get("source_to_idx", {})
    net = MultiTask(base, base.num_features, quality_aux=False,
                    source_count=len(source_map),
                    source_reversal=float(ckpt.get("source_reversal", 0.0)))
    net.load_state_dict(ckpt["state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.eval().to(device)
    loader = DataLoader(ZhejiangRows(rows), batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=device.type == "cuda")
    scores = np.empty(len(rows), dtype=np.float64)
    with torch.inference_mode():
        for x, indices in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                # MultiTask appends optional auxiliary heads (source and
                # quality), so read the fixed DR referable head by index.
                outputs = net(x)
                refer_logit = outputs[1]
            scores[indices.numpy()] = torch.sigmoid(refer_logit.float()).cpu().numpy()

    patient = {}
    for i, r in enumerate(rows):
        z = patient.setdefault(r["patient_id"], {"diagnosis": r["diagnosis"], "scores": [], "image_ids": []})
        z["scores"].append(float(scores[i]))
        z["image_ids"].append(r.get("image_id", ""))
    patients = [{"patient_id": k, "diagnosis": v["diagnosis"], "score": max(v["scores"]), "n_images": len(v["scores"])} for k, v in patient.items()]
    y = np.asarray([p["diagnosis"] == "DR" for p in patients], dtype=int)
    s = np.asarray([p["score"] for p in patients], dtype=float)
    point = metrics(y, s)
    rng = np.random.default_rng(20261113)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws = []
    for _ in range(args.boots):
        ix = np.r_[rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        draws.append((roc_auc_score(y[ix], s[ix]), average_precision_score(y[ix], s[ix])))
    draws = np.asarray(draws)

    args.predictions_private.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions_private.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "diagnosis", "n_images", "referable_dr_score"])
        w.writeheader()
        for p in patients:
            w.writerow({"patient_id": p["patient_id"], "diagnosis": p["diagnosis"],
                        "n_images": p["n_images"], "referable_dr_score": p["score"]})
    out = {
        "status": "EXPLORATORY_SOURCE_TRANSFER_ONLY_NOT_REFERABLE_DR_VALIDATION",
        "dataset": "Previously exposed Zhejiang single-center Optos UWF dataset",
        "label_semantics": "Source-defined DR diagnosis versus healthy; not referable-DR grade.",
        "checkpoint_sha256": hashlib.sha256(args.adapted_checkpoint.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "base_weights_sha256": hashlib.sha256(args.base_checkpoint.read_bytes()).hexdigest(),
        "images": len(rows),
        "patients": point["n_patients"],
        "patient_aggregation": "Maximum referable-DR score across available images per patient.",
        "fixed_referral_threshold_used": False,
        "patient_metrics": point,
        "stratified_patient_bootstrap_95ci": {
            "replicates": args.boots,
            "auroc": np.quantile(draws[:, 0], [0.025, 0.975]).tolist(),
            "auprc": np.quantile(draws[:, 1], [0.025, 0.975]).tolist(),
        },
        "claim_boundary": "Previously exposed retrospective exploratory domain-shift analysis; not independent confirmation, referable-DR validation, or clinical performance.",
        "row_predictions": "Private only; do not publish or transfer to GitHub.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
