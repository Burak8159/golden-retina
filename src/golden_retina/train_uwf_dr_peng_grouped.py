#!/usr/bin/env python3
"""Compare UWF DR classifiers using a duplicate- and filename-grouped split.

The source does not document filename-token identity semantics. The grouping
key is therefore a conservative filename proxy, not a verified patient ID.
Outputs and images are intended to remain on the authorized cloud VM.
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
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED = 8159
LABELS = ("Normal", "NPDR", "PDR")
LABEL_TO_INDEX = {"0": 0, "1": 1, "2": 2}
EXTENSIONS = {".png", ".jpg", ".jpeg"}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(root: Path):
    split_dir = next(root.rglob("train.csv")).parent
    images = {}
    for path in split_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            key = path.name.casefold()
            if key in images:
                raise ValueError("Image basename collision prevents safe label mapping")
            images[key] = path
    rows, missing, source_splits = [], 0, {}
    for split in ("train", "val", "test"):
        csv_path = split_dir / f"{split}.csv"
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            records = list(csv.DictReader(stream))
        source_splits[split] = len(records)
        for record in records:
            filename = Path(record["Image Name"].strip()).name
            path = images.get(filename.casefold())
            if path is None:
                missing += 1
                continue
            label = LABEL_TO_INDEX.get(str(record["Label"]).strip())
            if label is None:
                raise ValueError("Unknown source label; stop before training")
            try:
                with Image.open(path) as image:
                    image.verify()
                digest = sha256(path)
            except Exception:
                continue
            parts = filename.split("_")
            if len(parts) < 3:
                raise ValueError("Unexpected filename; identity grouping is unsafe")
            rows.append({"path": path, "filename": filename, "label": label,
                         "sha256": digest, "filename_group_proxy": "_".join(parts[:2]),
                         "source_split": split})
    if len(rows) < 500 or set(r["label"] for r in rows) != {0, 1, 2}:
        raise RuntimeError(f"Insufficient usable source rows or classes: {len(rows)}")
    return rows, {"archive_pngs": len(images), "official_split_rows": source_splits,
                  "unmatched_csv_rows": missing, "usable_rows": len(rows)}


def connected_groups(rows):
    parent = list(range(len(rows)))
    hash_counts = Counter(row["sha256"] for row in rows)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    by_filename_group, by_hash = {}, {}
    for i, row in enumerate(rows):
        key = row["filename_group_proxy"]
        if key in by_filename_group:
            union(i, by_filename_group[key])
        else:
            by_filename_group[key] = i
        digest = row["sha256"]
        if digest in by_hash:
            union(i, by_hash[digest])
        else:
            by_hash[digest] = i
    components = [str(find(i)) for i in range(len(rows))]
    return components, {"filename_group_proxy_count": len(by_filename_group),
                        "exact_image_hash_duplicate_groups": sum(count > 1 for count in hash_counts.values()),
                        "exact_image_hash_duplicate_rows": sum(count - 1 for count in hash_counts.values() if count > 1),
                        "connected_component_count": len(set(components)),
                        "identity_boundary": "Filename group tokens are undocumented; components are duplicate-aware filename proxies, not verified patients."}


def make_splits(rows, groups):
    y = np.asarray([r["label"] for r in rows])
    g = np.asarray(groups)
    selected = None
    # Select a group-safe outer fold, then a group-safe validation fold.
    for seed in range(SEED, SEED + 100):
        try:
            outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            remain_idx, test_idx = next(outer.split(np.zeros(len(y)), y, g))
            inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 997)
            tr_local, va_local = next(inner.split(np.zeros(len(remain_idx)), y[remain_idx], g[remain_idx]))
        except ValueError:
            continue
        train_idx, val_idx = remain_idx[tr_local], remain_idx[va_local]
        partitions = [set(g[idx]) for idx in (train_idx, val_idx, test_idx)]
        if any(partitions[i] & partitions[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            continue
        if all(set(y[idx]) == {0, 1, 2} for idx in (train_idx, val_idx, test_idx)):
            selected = (train_idx, val_idx, test_idx, seed)
            break
    if selected is None:
        raise RuntimeError("Could not create three group-disjoint partitions with all labels")
    ti, vi, xi, seed = selected
    parts = {name: [rows[i] for i in idx] for name, idx in zip(("train", "validation", "test"), (ti, vi, xi))}
    summary = {"seed": seed, "partition_rows": {k: len(v) for k, v in parts.items()},
               "filename_proxy_count": {k: len({r["filename_group_proxy"] for r in v}) for k, v in parts.items()},
               "duplicate_aware_component_count": {k: len(set(g[idx])) for k, idx in zip(("train", "validation", "test"), (ti, vi, xi))},
               "partition_class_counts": {k: dict(Counter(LABELS[r["label"]] for r in v)) for k, v in parts.items()},
               "filename_group_proxy_overlap": {f"{a}-{b}": len({r["filename_group_proxy"] for r in parts[a]} & {r["filename_group_proxy"] for r in parts[b]}) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))},
               "exact_hash_overlap": {f"{a}-{b}": len({r["sha256"] for r in parts[a]} & {r["sha256"] for r in parts[b]}) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))},
               "boundary": "Reconstructed development split grouped by exact image and undocumented filename proxy. Not verified patient-independent; source modality is 220-240 degree UWF, not ordinary camera fundus."}
    return parts, summary


class ImageRows(Dataset):
    def __init__(self, rows, size, train=False):
        self.rows = rows
        ops = [transforms.Resize((size, size))]
        if train:
            ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(8),
                    transforms.ColorJitter(.05, .05, .04, .01)]
        ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        self.tf = transforms.Compose(ops)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            x = self.tf(ImageOps.exif_transpose(image).convert("RGB"))
        return x, row["label"]


def create_model(architecture):
    if architecture == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 3)
    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    elif architecture == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 3)
    else:
        raise ValueError(architecture)
    return model


def predict(model, loader):
    model.eval(); probabilities, labels = [], []
    with torch.inference_mode():
        for x, y in loader:
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x.cuda(non_blocking=True))
            probabilities.extend(torch.softmax(logits.float(), dim=1).cpu().numpy())
            labels.extend(y.numpy())
    return np.asarray(labels), np.asarray(probabilities)


def evaluate(labels, probabilities):
    prediction = probabilities.argmax(axis=1)
    cm = confusion_matrix(labels, prediction, labels=[0, 1, 2])
    p, r, f1, support = precision_recall_fscore_support(labels, prediction, labels=[0, 1, 2], zero_division=0)
    one_vs_rest_auc = {}
    for index, name in enumerate(LABELS):
        binary = (labels == index).astype(int)
        one_vs_rest_auc[name] = float(roc_auc_score(binary, probabilities[:, index])) if len(np.unique(binary)) == 2 else None
    # Referable grouping is a descriptive post-hoc grouping of the three source grades.
    ref_true = labels > 0
    ref_pred = prediction > 0
    tn, fp, fn, tp = confusion_matrix(ref_true, ref_pred, labels=[False, True]).ravel()
    return {"n": int(len(labels)), "accuracy": float(accuracy_score(labels, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
            "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
            "confusion_matrix_rows_true_normal_npdr_pdr_cols_pred_normal_npdr_pdr": cm.tolist(),
            "per_grade": {name: {"support": int(support[i]), "precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]), "one_vs_rest_auroc": one_vs_rest_auc[name]} for i, name in enumerate(LABELS)},
            "source_grade_positive_vs_normal": {"n_positive": int(ref_true.sum()), "n_negative": int((~ref_true).sum()), "sensitivity": float(tp / (tp + fn)) if tp + fn else None, "specificity": float(tn / (tn + fp)) if tn + fp else None, "misses": int(fn), "false_positives": int(fp)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seconds-per-model", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA L4 is required")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    rows, archive_audit = load_rows(args.root)
    groups, group_audit = connected_groups(rows)
    parts, split_audit = make_splits(rows, groups)
    args.out.mkdir(parents=True, exist_ok=True)
    audit = {"dataset": "UWF SLO fundus image dataset for intelligent DR system (Peng et al., 2026)",
             "archive_audit": archive_audit, "group_audit": group_audit, "split_audit": split_audit,
             "selection_metric": "validation accuracy; macro-F1 then validation loss as tie-breaks",
             "test_policy": "one evaluation after architecture and epoch are selected on validation only",
             "evidence_boundary": "Exploratory reconstructed split; filename grouping is not verified patient identity, source is UWF, and no clinical/general-camera claim is supported."}
    (args.out / "split_audit_aggregate.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"status": "audit_complete", "audit": audit}), flush=True)
    if args.audit_only:
        return
    loaders = {name: DataLoader(ImageRows(part, args.image_size, name == "train"),
                                batch_size=args.batch_size, shuffle=name == "train",
                                num_workers=args.workers, pin_memory=True)
               for name, part in parts.items()}
    counts = np.bincount([row["label"] for row in parts["train"]], minlength=3)
    weights = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** .5, dtype=torch.float32, device="cuda")
    model_results = []
    for architecture in ("convnext_tiny", "efficientnet_b0", "resnet50"):
        start = time.time(); model = create_model(architecture).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=2e-4)
        best_key = (-1.0, -1.0, float("inf")); best_epoch = 0; stale = 0
        checkpoint = args.out / f"{architecture}_validation_best.pt"
        for epoch in range(1, args.epochs + 1):
            model.train(); total_loss = 0.0; seen = 0
            for x, y in loaders["train"]:
                x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(x); loss = nn.functional.cross_entropy(logits, y, weight=weights)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
                total_loss += loss.item() * len(y); seen += len(y)
            vy, vp = predict(model, loaders["validation"])
            metrics = evaluate(vy, vp)
            val_loss = float(nn.functional.cross_entropy(torch.tensor(np.log(np.maximum(vp, 1e-8)), dtype=torch.float32), torch.tensor(vy)).item())
            key = (metrics["accuracy"], metrics["macro_f1"], -val_loss)
            print(json.dumps({"architecture": architecture, "epoch": epoch,
                              "train_loss": total_loss / max(seen, 1), "validation_loss": val_loss,
                              "validation": metrics}), flush=True)
            if key > best_key:
                best_key = key; best_epoch = epoch; stale = 0
                torch.save({"state_dict": model.state_dict(), "architecture": architecture,
                            "labels": list(LABELS), "epoch": epoch, "image_size": args.image_size}, checkpoint)
            else:
                stale += 1
            if stale >= 3 or time.time() - start >= args.seconds_per_model:
                break
        model_results.append({"architecture": architecture, "best_epoch": best_epoch,
                              "validation_accuracy": best_key[0], "validation_macro_f1": best_key[1],
                              "checkpoint": str(checkpoint)})
        del model; torch.cuda.empty_cache()
    chosen = max(model_results, key=lambda x: (x["validation_accuracy"], x["validation_macro_f1"]))
    checkpoint = torch.load(chosen["checkpoint"], map_location="cpu", weights_only=False)
    model = create_model(chosen["architecture"]); model.load_state_dict(checkpoint["state_dict"]); model.cuda()
    ty, tp = predict(model, loaders["test"])
    report = {"status": "complete", "dataset": audit["dataset"], "selected_architecture": chosen["architecture"],
              "selected_epoch": chosen["best_epoch"], "selection": "Validation accuracy, then macro-F1; test used once after freezing.",
              "candidate_summaries": [{k: v for k, v in item.items() if k != "checkpoint"} for item in model_results],
              "test_metrics": evaluate(ty, tp), "split_audit": split_audit,
              "boundary": audit["evidence_boundary"]}
    (args.out / "aggregate_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "complete", "selected_architecture": report["selected_architecture"],
                      "test_metrics": report["test_metrics"], "boundary": report["boundary"]}), flush=True)


if __name__ == "__main__":
    main()
