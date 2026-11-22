#!/usr/bin/env python3
"""Image/hash-grouped exploratory glaucoma comparison on official ACRIMA.

ACRIMA supplies no patient identifiers. The report therefore describes an
image/duplicate-group holdout only and is not patient-independent validation.
All image manifests, scores, and checkpoints stay on the VM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_rows(root: Path):
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"})
    rows, bad = [], 0
    for path in paths:
        label = int("_g_" in path.name)
        try:
            with Image.open(path) as image:
                image.load()
            rows.append({"path": path, "label": label, "hash": sha256(path)})
        except Exception:
            bad += 1
    seen, unique, duplicate_rows = set(), [], 0
    for row in rows:
        if row["hash"] in seen:
            duplicate_rows += 1
            continue
        seen.add(row["hash"])
        unique.append(row)
    if Counter(r["label"] for r in unique) != {1: 396, 0: 309}:
        raise ValueError(f"Unexpected ACRIMA label counts: {Counter(r['label'] for r in unique)}")
    audit = {"files_seen": len(paths), "decode_valid": len(rows), "decode_failures": bad,
             "exact_duplicate_rows_removed": duplicate_rows, "eligible_unique_images": len(unique),
             "glaucoma_images": 396, "normal_images": 309,
             "patient_ids_available": False}
    return unique, audit


def grouped_split(rows):
    y = np.asarray([r["label"] for r in rows], dtype=np.int64)
    groups = np.asarray([r["hash"] for r in rows])
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    trainval, test = next(outer.split(np.zeros(len(y)), y, groups))
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + 1)
    tr, va = next(inner.split(np.zeros(len(trainval)), y[trainval], groups[trainval]))
    parts = {"train": [rows[trainval[i]] for i in tr],
             "validation": [rows[trainval[i]] for i in va],
             "test": [rows[i] for i in test]}
    sets = {k: {r["hash"] for r in v} for k, v in parts.items()}
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]:
        raise RuntimeError("Exact-hash groups crossed partitions")
    for name, subset in parts.items():
        if len({r["label"] for r in subset}) != 2:
            raise RuntimeError(f"Both classes are required in {name}")
    return parts


class Fundus(Dataset):
    def __init__(self, rows, train=False):
        self.rows = rows
        ops = [transforms.Resize(256), transforms.CenterCrop(224)]
        if train:
            ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(8),
                    transforms.ColorJitter(brightness=.08, contrast=.08, saturation=.06)]
        ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        self.tf = transforms.Compose(ops)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        return self.tf(image), torch.tensor(row["label"], dtype=torch.float32)


def make_model(name):
    if name == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)
        head = model.classifier
    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 1)
        head = model.fc
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        head = model.classifier
    else:
        raise ValueError(name)
    return model, head


def predict(model, loader):
    model.eval(); scores, labels = [], []
    with torch.inference_mode():
        for images, y in loader:
            images = images.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(images).reshape(-1)
            scores.extend(torch.sigmoid(logits.float()).cpu().numpy().tolist())
            labels.extend(y.numpy().astype(int).tolist())
    return np.asarray(scores), np.asarray(labels)


def metrics(y, p, threshold=.5):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n_images": int(len(y)), "positive_images": int(y.sum()),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "sensitivity": float(tp / (tp + fn)), "specificity": float(tn / (tn + fp)),
            "false_negatives": int(fn), "false_positives": int(fp),
            "true_positives": int(tp), "true_negatives": int(tn),
            "auroc": float(roc_auc_score(y, p)),
            "average_precision": float(average_precision_score(y, p))}


def choose_threshold(y, p):
    candidates = np.unique(np.r_[0., p, 1.])
    return max(candidates, key=lambda t: (metrics(y, p, float(t))["balanced_accuracy"],
                                         metrics(y, p, float(t))["sensitivity"],
                                         metrics(y, p, float(t))["accuracy"]))


def grouped_bootstrap(rows, y, p, threshold, reps=2000):
    groups = np.asarray([r["hash"] for r in rows])
    unique = np.unique(groups)
    ix_by_group = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(SEED + 2); values = defaultdict(list)
    for _ in range(reps):
        selected = rng.choice(unique, size=len(unique), replace=True)
        ix = np.concatenate([ix_by_group[g] for g in selected])
        if np.unique(y[ix]).size < 2:
            continue
        result = metrics(y[ix], p[ix], threshold)
        for key in ("accuracy", "sensitivity", "specificity", "auroc"):
            values[key].append(result[key])
    return {key: [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]
            for key, vals in values.items() if vals}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--architectures", nargs="+", choices=("convnext_tiny", "resnet50", "efficientnet_b0"),
                        default=("convnext_tiny", "resnet50", "efficientnet_b0"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA L4 required; refusing a CPU fallback")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    rows, audit = load_rows(args.data); parts = grouped_split(rows); args.out.mkdir(parents=True, exist_ok=True)
    audit["splits"] = {k: {"images": len(v), "glaucoma": sum(r["label"] for r in v),
                           "normal": sum(1-r["label"] for r in v)} for k, v in parts.items()}
    audit["split_boundary"] = "Stratified exact-hash-group split only. No patient identifiers are available; same-patient leakage cannot be ruled out."
    (args.out / "split_audit_aggregate.json").write_text(json.dumps(audit, indent=2) + "\n")
    loaders = {k: DataLoader(Fundus(v, k == "train"), batch_size=args.batch_size,
                             shuffle=k == "train", num_workers=args.workers, pin_memory=True)
               for k, v in parts.items()}
    candidates = {}
    train_counts = Counter(r["label"] for r in parts["train"])
    pos_weight = torch.tensor([train_counts[0] / train_counts[1]], dtype=torch.float32, device="cuda")
    for arch in args.architectures:
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        model, head = make_model(arch); model.cuda()
        head_ids = {id(p) for p in head.parameters()}
        backbone = [p for p in model.parameters() if id(p) not in head_ids]
        optimizer = torch.optim.AdamW([{"params": backbone, "lr": 1e-5},
                                       {"params": list(head.parameters()), "lr": 2e-4}], weight_decay=.02)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight); scaler = torch.amp.GradScaler("cuda")
        best_auc, best_state, best_epoch, stale = -1., None, 0, 0
        for epoch in range(1, args.epochs + 1):
            model.train(); total = 0.; seen = 0
            for images, labels in loaders["train"]:
                images = images.cuda(non_blocking=True); labels = labels.cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(images).reshape(-1); loss = loss_fn(logits, labels)
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer); scaler.update(); total += float(loss.detach()) * len(labels); seen += len(labels)
            vp, vy = predict(model, loaders["validation"]); auc = float(roc_auc_score(vy, vp)); ap = float(average_precision_score(vy, vp))
            print(json.dumps({"architecture": arch, "epoch": epoch, "train_loss": total/max(seen, 1),
                              "validation_auroc": auc, "validation_average_precision": ap,
                              "validation_metrics_at_0_5": metrics(vy, vp)}), flush=True)
            if auc > best_auc + 1e-4:
                best_auc, best_epoch, stale = auc, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= 3:
                    break
        if best_state is None:
            raise RuntimeError(f"No selected checkpoint for {arch}")
        model.load_state_dict(best_state); vp, vy = predict(model, loaders["validation"])
        threshold = float(choose_threshold(vy, vp))
        ck = args.out / f"{arch}_validation_best_private.pt"
        torch.save({"state_dict": best_state, "architecture": arch, "classes": ["normal", "glaucoma"],
                    "pretrained": "ImageNet", "input_size": 224, "mean": MEAN, "std": STD,
                    "seed": SEED, "best_epoch": best_epoch, "validation_threshold": threshold}, ck)
        candidates[arch] = {"checkpoint": ck, "validation_auroc": best_auc,
                            "validation_metrics": metrics(vy, vp, threshold), "threshold": threshold,
                            "best_epoch": best_epoch}
        del optimizer, scaler, model; torch.cuda.empty_cache()
    selected_name = max(candidates, key=lambda n: (candidates[n]["validation_auroc"],
                                                    candidates[n]["validation_metrics"]["balanced_accuracy"]))
    selected = candidates[selected_name]; ckdata = torch.load(selected["checkpoint"], map_location="cpu", weights_only=False)
    model, _ = make_model(selected_name); model.load_state_dict(ckdata["state_dict"]); model.cuda()
    test_p, test_y = predict(model, loaders["test"]); test_metrics = metrics(test_y, test_p, selected["threshold"])
    ci = grouped_bootstrap(parts["test"], test_y, test_p, selected["threshold"])
    report = {"status": "complete", "dataset": "ACRIMA v1 · official Figshare CC BY 4.0",
              "task": "Glaucoma versus normal fundus image classification",
              "data_audit": audit,
              "candidate_validation": {k: {"auroc": v["validation_auroc"], "threshold": v["threshold"],
                                            "best_epoch": v["best_epoch"], "metrics": v["validation_metrics"]}
                                       for k, v in candidates.items()},
              "selected_candidate": selected_name, "selected_threshold": selected["threshold"],
              "test_metrics": test_metrics, "test_exact_hash_group_bootstrap_95ci": ci,
              "patient_ids_available": False,
              "boundary": "Exploratory image/duplicate-group holdout only; same-patient leakage cannot be ruled out. No external or clinical validation. Checkpoints, image lists, and per-image scores remain on the VM."}
    (args.out / "aggregate_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "complete", "selected_candidate": selected_name,
                      "test_metrics": test_metrics, "group_bootstrap_95ci": ci,
                      "boundary": report["boundary"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
