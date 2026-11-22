#!/usr/bin/env python3
"""Train and one-shot evaluate patient-grouped glaucoma candidates on HYGD.

All manifests, predictions, and checkpoints are intended to stay on the private
authorized compute VM. The official test fold is untouched until model and
threshold selection are complete on the internal validation fold.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED = 8159
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(root: Path):
    label_file = root / "Labels.csv"
    image_root = root / "Images"
    if not label_file.is_file() or not image_root.is_dir():
        raise FileNotFoundError("HYGD requires Labels.csv and Images/ from the official release")
    with label_file.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = {str(c or "").strip(): c for c in (reader.fieldnames or [])}
        required = {"Image Name", "Patient", "Label"}
        if not required.issubset(columns):
            raise ValueError(f"HYGD label table is missing required columns: {sorted(required - set(columns))}")
        quality_key = columns.get("Quality Score")
        rows = []
        for item in reader:
            filename = str(item[columns["Image Name"]]).strip()
            patient = str(item[columns["Patient"]]).strip()
            raw_label = str(item[columns["Label"]]).strip().upper()
            if raw_label not in {"GON+", "GON-"} or not patient or not filename:
                raise ValueError("HYGD has an empty patient/image or unexpected label")
            path = image_root / filename
            if not path.is_file():
                raise FileNotFoundError(f"Official HYGD image absent: {filename}")
            quality = None
            if quality_key and str(item.get(quality_key, "")).strip():
                quality = float(item[quality_key])
            rows.append({"path": path, "relative_path": f"Images/{filename}",
                         "patient": patient, "label": int(raw_label == "GON+"),
                         "quality": quality})
    if len(rows) < 100 or len({row["label"] for row in rows}) != 2:
        raise ValueError("HYGD image/label support is unexpectedly small or single-class")
    return rows


def attach_duplicate_components(rows):
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    seen = {}
    exact_duplicate_images = 0
    for row in rows:
        key = sha256(row["path"])
        row["sha256"] = key
        prior = seen.get(key)
        if prior:
            exact_duplicate_images += 1
            union(row["patient"], prior)
        else:
            seen[key] = row["patient"]
    for row in rows:
        row["group"] = find(row["patient"])
    return exact_duplicate_images, len({row["group"] for row in rows})


def split_rows(rows):
    y = np.asarray([r["label"] for r in rows], dtype=np.int64)
    groups = np.asarray([r["group"] for r in rows])
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    trainval_idx, test_idx = next(outer.split(np.zeros(len(y)), y, groups))
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
    inner_train, val_local = next(inner.split(np.zeros(len(trainval_idx)), y[trainval_idx], groups[trainval_idx]))
    train_idx = trainval_idx[inner_train]
    val_idx = trainval_idx[val_local]
    partitions = {"train": [rows[i] for i in train_idx],
                  "validation": [rows[i] for i in val_idx],
                  "test": [rows[i] for i in test_idx]}
    group_sets = {k: {r["group"] for r in subset} for k, subset in partitions.items()}
    if (group_sets["train"] & group_sets["validation"] or
            group_sets["train"] & group_sets["test"] or
            group_sets["validation"] & group_sets["test"]):
        raise RuntimeError("Patient/duplicate component overlap across frozen partitions")
    for name, subset in partitions.items():
        if {r["label"] for r in subset} != {0, 1}:
            raise RuntimeError(f"Both labels are required in {name}; stopping before training")
    return partitions


class FundusRows(Dataset):
    def __init__(self, rows, training=False):
        self.rows = rows
        ops = [transforms.Resize(256), transforms.CenterCrop(224)]
        if training:
            ops += [transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(8),
                    transforms.ColorJitter(brightness=0.10, contrast=0.10,
                                           saturation=0.08)]
        ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        return self.transform(image), torch.tensor(row["label"], dtype=torch.float32)


def make_model(name: str):
    if name == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)
        head = model.classifier
    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 1)
        head = model.fc
    else:
        raise ValueError(f"Unsupported architecture: {name}")
    return model, head


def loaders(parts, batch_size, workers):
    return {
        "train": DataLoader(FundusRows(parts["train"], True), batch_size=batch_size,
                             shuffle=True, num_workers=workers, pin_memory=True),
        "validation": DataLoader(FundusRows(parts["validation"]), batch_size=batch_size,
                                  shuffle=False, num_workers=workers, pin_memory=True),
        "test": DataLoader(FundusRows(parts["test"]), batch_size=batch_size,
                            shuffle=False, num_workers=workers, pin_memory=True),
    }


def infer(model, loader, device):
    model.eval()
    logits, targets = [], []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(images).reshape(-1)
            logits.extend(out.float().cpu().numpy().tolist())
            targets.extend(labels.numpy().astype(int).tolist())
    logits = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40))), np.asarray(targets, dtype=np.int64)


def metric_pack(y, p, threshold):
    pred = (p >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n_images": int(len(y)), "positive_images": int(y.sum()),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
            "specificity": float(tn / (tn + fp)) if tn + fp else None,
            "false_negatives": int(fn), "false_positives": int(fp),
            "true_positives": int(tp), "true_negatives": int(tn),
            "auroc": float(roc_auc_score(y, p)),
            "average_precision": float(average_precision_score(y, p))}


def select_threshold(y, p):
    candidates = np.unique(np.r_[0.0, p, 1.0])
    scored = [(metric_pack(y, p, float(t)), float(t)) for t in candidates]
    m, t = max(scored, key=lambda item: (item[0]["balanced_accuracy"],
                                         item[0]["sensitivity"], item[0]["accuracy"]))
    return t, m


def patient_bootstrap(rows, probabilities, threshold, reps=2000):
    y = np.asarray([r["label"] for r in rows], dtype=np.int64)
    groups = np.asarray([r["group"] for r in rows])
    unique = np.unique(groups)
    indices = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(SEED + 2)
    values = defaultdict(list)
    for _ in range(reps):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        ix = np.concatenate([indices[g] for g in chosen])
        if np.unique(y[ix]).size < 2:
            continue
        m = metric_pack(y[ix], probabilities[ix], threshold)
        for key in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "auroc"):
            values[key].append(m[key])
    return {key: [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]
            for key, vals in values.items() if vals}


def write_private_manifest(path, parts):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("relative_path", "label", "patient_id", "duplicate_component",
                         "quality_score", "sha256", "split"))
        for split, rows in parts.items():
            for row in rows:
                writer.writerow((row["relative_path"], row["label"], row["patient"],
                                 row["group"], row["quality"], row["sha256"], split))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--architectures", nargs="+", choices=("convnext_tiny", "resnet50"),
                        default=("convnext_tiny", "resnet50"))
    args = parser.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("L4 is required; refusing a CPU fallback")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.data_root)
    exact_duplicates, component_count = attach_duplicate_components(rows)
    parts = split_rows(rows)
    write_private_manifest(args.output_dir / "split_manifest_private.csv", parts)
    loaders_by_split = loaders(parts, args.batch_size, args.workers)
    train_y = np.asarray([r["label"] for r in parts["train"]], dtype=np.int64)
    counts = np.bincount(train_y, minlength=2)
    pos_weight = torch.tensor([counts[0] / max(1, counts[1])], dtype=torch.float32, device=device)
    candidates = {}
    for architecture in args.architectures:
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        model, head = make_model(architecture)
        model = model.to(device)
        head_ids = {id(p) for p in head.parameters()}
        backbone = [p for p in model.parameters() if id(p) not in head_ids]
        optimizer = torch.optim.AdamW([{"params": backbone, "lr": 1e-5},
                                       {"params": list(head.parameters()), "lr": 2e-4}],
                                      weight_decay=0.02)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        scaler = torch.amp.GradScaler("cuda")
        best_auc, best_state, best_epoch, stale = -1.0, None, 0, 0
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train(); running = 0.0; seen = 0
            for images, labels in loaders_by_split["train"]:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images).reshape(-1)
                    loss = loss_fn(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                running += float(loss.detach()) * len(labels); seen += len(labels)
            val_p, val_y = infer(model, loaders_by_split["validation"], device)
            val_auc = float(roc_auc_score(val_y, val_p))
            val_ap = float(average_precision_score(val_y, val_p))
            record = {"architecture": architecture, "epoch": epoch,
                      "train_loss": running / max(seen, 1),
                      "validation_auroc": val_auc, "validation_auprc": val_ap}
            history.append(record); print(json.dumps(record), flush=True)
            if val_auc > best_auc + 1e-4:
                best_auc, best_epoch, stale = val_auc, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= 4:
                    break
        if best_state is None:
            raise RuntimeError(f"No validation-selected checkpoint for {architecture}")
        model.load_state_dict(best_state)
        val_p, val_y = infer(model, loaders_by_split["validation"], device)
        threshold, val_metrics = select_threshold(val_y, val_p)
        checkpoint = args.output_dir / f"hygd_{architecture}_best_private.pt"
        torch.save({"state_dict": best_state, "architecture": architecture,
                    "pretrained": "torchvision ImageNet default", "input_size": 224,
                    "mean": MEAN, "std": STD, "seed": SEED, "best_epoch": best_epoch,
                    "validation_threshold": threshold}, checkpoint)
        candidates[architecture] = {"checkpoint": checkpoint,
                                    "validation_auroc": best_auc,
                                    "validation_metrics": val_metrics,
                                    "threshold": threshold, "best_epoch": best_epoch,
                                    "history": history}
        del optimizer, scaler, model
        torch.cuda.empty_cache()

    selected_name = max(candidates, key=lambda name: (candidates[name]["validation_auroc"],
                                                       candidates[name]["validation_metrics"]["balanced_accuracy"]))
    selected = candidates[selected_name]
    frozen_model, _ = make_model(selected_name)
    checkpoint_data = torch.load(selected["checkpoint"], map_location="cpu", weights_only=False)
    frozen_model.load_state_dict(checkpoint_data["state_dict"])
    frozen_model = frozen_model.to(device)
    test_p, test_y = infer(frozen_model, loaders_by_split["test"], device)
    test_metrics = metric_pack(test_y, test_p, selected["threshold"])
    test_ci = patient_bootstrap(parts["test"], test_p, selected["threshold"])
    selected["checkpoint"].replace(args.output_dir / "selected_hygd_glaucoma_private.pt")
    output = {
        "status": "PATIENT_GROUPED_ONE_SHOT_TEST_COMPLETE",
        "source": "PhysioNet HYGD v1.1.0; single-source 45-degree TOPCON fundus images",
        "task": "GON+ versus GON- image label; this is a glaucoma-associated optic neuropathy screening experiment",
        "data_counts": {"images": len(rows), "patients": len({r["patient"] for r in rows}),
                        "duplicate_components": component_count,
                        "exact_duplicate_images": exact_duplicates,
                        "gon_positive_images": int(sum(r["label"] for r in rows)),
                        "gon_negative_images": int(len(rows) - sum(r["label"] for r in rows)),
                        "patient_mixed_label_groups": int(sum(1 for group in {r["group"] for r in rows}
                            if len({r["label"] for r in rows if r["group"] == group}) > 1)),
                        "splits": {k: {"images": len(v), "patients": len({r["group"] for r in v}),
                                      "positive_images": int(sum(r["label"] for r in v))}
                                   for k, v in parts.items()}},
        "split_policy": "Nested deterministic StratifiedGroupKFold by patient; exact-duplicate patient components merged; approximately 64/16/20 train/validation/test. Test held until candidate and threshold freeze.",
        "candidate_validation": {name: {"auroc": value["validation_auroc"],
                                        "threshold": value["threshold"],
                                        "metrics": value["validation_metrics"],
                                        "best_epoch": value["best_epoch"]}
                                 for name, value in candidates.items()},
        "selected_candidate": selected_name,
        "test_threshold": float(selected["threshold"]),
        "held_out_patient_group_test": test_metrics,
        "patient_cluster_bootstrap_95ci": test_ci,
        "checkpoint_sha256_private": sha256(args.output_dir / "selected_hygd_glaucoma_private.pt"),
        "private_split_manifest_sha256": sha256(args.output_dir / "split_manifest_private.csv"),
        "selection_boundary": "Validation selected architecture, checkpoint and threshold; held-out test opened once after freeze. Do not retune on this test.",
        "limitations": ["Single-center, single-camera cohort; no external validation.",
                        "Performance is not a clinical diagnosis or deployment clearance.",
                        "Sensitivity and specificity depend on selected threshold; report all metrics and confidence intervals.",
                        "All patient-level manifests, predictions, and checkpoints remain on the VM."],
    }
    report_path = args.output_dir / "aggregate_report.json"
    report_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "selected_candidate": selected_name,
                      "threshold": selected["threshold"], "test": test_metrics,
                      "patient_bootstrap_95ci": test_ci}, indent=2), flush=True)


if __name__ == "__main__":
    main()
