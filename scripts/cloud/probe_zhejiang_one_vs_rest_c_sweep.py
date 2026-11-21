#!/usr/bin/env python3
"""Regularization sensitivity for disease-vs-all frozen-feature probes.

Uses the frozen, patient-grouped Zhejiang train/validation rows only. For each
disease, every other source label (including Healthy) is a negative class.
Test rows are filtered before image loading. Row-level outputs stay on the VM.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import Resize


class PairedTransforms(Dataset):
    def __init__(self, rows, conv_transform, flair_side=512):
        self.rows = rows
        self.conv_transform = conv_transform
        self.flair_side = flair_side
        self.dino_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        with Image.open(self.rows[i]["image_path"]) as im:
            im = im.convert("RGB")
            xc = self.conv_transform(im)
            xd = self.dino_transform(im)
            arr = np.asarray(im, dtype=np.float32)
        if arr.max() > 0:
            arr /= 255.0
        xf = torch.from_numpy(arr).permute(2, 0, 1)
        h, w = xf.shape[-2:]
        scale = max(h, w) / self.flair_side
        size = (max(1, int(h / scale)), max(1, int(w / scale)))
        xf = Resize(size, antialias=True)(xf)
        xf = torch.nn.functional.pad(xf, (0, self.flair_side-xf.shape[-1], 0, self.flair_side-xf.shape[-2]))
        return xc, xd, xf, i


def model_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eye_pool(rows, image_scores):
    pooled = {}
    for row, p in zip(rows, image_scores):
        key = (row["patient_group"], row["eye_id"])
        y = row["source_label"]
        rec = pooled.setdefault(key, {"label": y, "score": float(p)})
        if rec["label"] != y:
            raise ValueError("Conflicting source labels within eye")
        rec["score"] = max(rec["score"], float(p))
    return list(pooled.values())


def choose_eye_threshold(labels, scores, target_spec=0.90):
    candidates = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
    feasible = []
    for threshold in candidates:
        pred = scores >= threshold
        tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
        spec = tn / max(tn + fp, 1)
        sens = tp / max(tp + fn, 1)
        if spec + 1e-12 >= target_spec:
            feasible.append((sens, spec, float(threshold)))
    return max(feasible)[2]


def weights_for_patient_and_source_class(train_rows, target_label):
    patient_rows = Counter(r["patient_group"] for r in train_rows)
    patients_by_label = defaultdict(set)
    for r in train_rows:
        patients_by_label[r["source_label"]].add(r["patient_group"])
    negative_labels = sorted(set(r["source_label"] for r in train_rows) - {target_label})
    weights = []
    for r in train_rows:
        label = r["source_label"]
        if label == target_label:
            weights.append(1.0 / (len(patients_by_label[label]) * patient_rows[r["patient_group"]]))
        else:
            weights.append(1.0 / (len(negative_labels) * len(patients_by_label[label]) * patient_rows[r["patient_group"]]))
    return np.asarray(weights, dtype=np.float64)


def fit_task(train_rows, val_rows, x_train, x_val, target_label, c_value):
    ytr = np.asarray([int(r["source_label"] == target_label) for r in train_rows], dtype=int)
    scaler = StandardScaler().fit(x_train)
    clf = LogisticRegression(C=c_value, max_iter=2500, solver="lbfgs", random_state=20261121)
    clf.fit(scaler.transform(x_train), ytr, sample_weight=weights_for_patient_and_source_class(train_rows, target_label))
    pos = list(clf.classes_).index(1)
    image_score = clf.predict_proba(scaler.transform(x_val))[:, pos]
    eyes = eye_pool(val_rows, image_score)
    y = np.asarray([int(e["label"] == target_label) for e in eyes], dtype=int)
    score = np.asarray([e["score"] for e in eyes], dtype=float)
    threshold = choose_eye_threshold(y, score)
    pred = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    per_negative_label = {}
    for label in sorted({e["label"] for e in eyes} - {target_label}):
        group = [e for e in eyes if e["label"] == label]
        false_positives = sum(e["score"] >= threshold for e in group)
        per_negative_label[label] = {
            "eyes": int(len(group)),
            "false_positives": int(false_positives),
            "false_positive_rate": float(false_positives / max(len(group), 1)),
        }
    summary = {
        "validation_eyes": int(len(y)), "positive_eyes": int(y.sum()),
        "negative_eyes_all_other_labels": int(len(y)-y.sum()),
        "validation_patients": len({r["patient_group"] for r in val_rows}),
        "accuracy_validation_selected_threshold": float(accuracy_score(y, pred)),
        "sensitivity_validation_selected_threshold": float(tp/max(tp+fn, 1)),
        "specificity_validation_selected_threshold": float(tn/max(tn+fp, 1)),
        "false_negatives": int(fn), "false_positives": int(fp),
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "brier_score": float(brier_score_loss(y, score)),
        "threshold_selected_on_validation_eyes": float(threshold),
        "false_positives_by_competing_label": per_negative_label,
        "selection_note": "Optimistic model/threshold selection on this validation set; no test partition loaded."
    }
    return summary, clf, scaler, image_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--dinov2-source", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--flair-source", type=Path, required=True)
    ap.add_argument("--flair-checkpoint", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--c-values", type=float, nargs="+", default=[0.03, 0.1, 1.0, 10.0])
    args = ap.parse_args()

    with args.manifest.open(newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] in {"train", "validation"}]
    if not rows:
        raise ValueError("No train/validation rows found")
    for r in rows:
        r["image_path"] = str(args.image_root / r["relative_image_path"])
        if not Path(r["image_path"]).is_file():
            raise FileNotFoundError("Unresolved VM image path")
    labels = sorted({r["source_label"] for r in rows})
    label_names = {"AMD": "AMD", "DR": "DR", "Healthy": "Healthy", "PM": "Pathological myopia",
                   "RD": "Retinal detachment", "RVO": "RVO", "Uveitis": "Uveitis"}
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validation"]

    conv_weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    conv_transform = conv_weights.transforms()
    conv = models.convnext_tiny(weights=conv_weights)
    conv.classifier[-1] = torch.nn.Identity()
    conv.eval()
    for p in conv.parameters(): p.requires_grad_(False)

    import sys
    sys.path.insert(0, str(args.dinov2_source))
    dino = torch.hub.load(str(args.dinov2_source.resolve()), "dinov2_vitb14", source="local", pretrained=True, trust_repo=True)
    dino.eval()
    for p in dino.parameters(): p.requires_grad_(False)
    import sys
    sys.path.insert(0, str(args.flair_source))
    from flair import FLAIRModel
    flair = FLAIRModel.from_pretrained(str(args.flair_checkpoint), vision_pretrained=False)
    flair.eval(); flair.vision_model.eval()
    for p in flair.parameters(): p.requires_grad_(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conv.to(device); dino.to(device); flair.vision_model.to(device)
    loader = DataLoader(PairedTransforms(rows, conv_transform), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    conv_f = np.empty((len(rows), 768), dtype=np.float32)
    dino_f = np.empty((len(rows), 768), dtype=np.float32)
    flair_f = np.empty((len(rows), int(flair.vision_model.out_dim)), dtype=np.float32)
    with torch.inference_mode():
        for xc, xd, xf, idx in loader:
            xc, xd, xf = xc.to(device, non_blocking=True), xd.to(device, non_blocking=True), xf.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                conv_z = conv(xc)
                dino_z = dino.forward_features(xd)["x_norm_clstoken"]
                flair_z = flair.vision_model(xf)
            conv_f[idx.numpy()] = conv_z.float().cpu().numpy()
            dino_f[idx.numpy()] = dino_z.float().cpu().numpy()
            flair_f[idx.numpy()] = flair_z.float().cpu().numpy()

    index = {r["image_id"]: i for i, r in enumerate(rows)}
    itrain = np.asarray([index[r["image_id"]] for r in train_rows])
    ival = np.asarray([index[r["image_id"]] for r in val_rows])
    results = {
        "status": "PATIENT_GROUPED_DISEASE_VS_ALL_REGULARIZATION_SWEEP_VALIDATION_ONLY",
        "task_definition": "For each disease, target label is positive; Healthy and every other disease label are negative. Binary heads trained on training patients only.",
        "split": "Frozen patient-grouped 60/20/20 source manifest; only train and validation rows were loaded; test not accessed.",
        "analysis_unit": "Eye; max-pooled across images. Patient and source-label balancing within each binary task.",
        "threshold_policy": "For every C, validation eye-level threshold maximizes sensitivity subject to specificity >=90%. C and threshold are both selected on this same small validation set; thresholded metrics are optimistic and for development only.",
        "label_map": {k: v for k, v in label_names.items() if k in labels},
        "models": {},
        "boundary": "Single-source development validation only. Validation is used for candidate/threshold selection; no external or clinical validation. The initial source test was previously scored by another probe and is excluded from this run."
    }
    if not args.c_values or any(c <= 0 for c in args.c_values):
        raise ValueError("All --c-values must be positive")
    for name, model, features in (
        ("ConvNeXt-Tiny ImageNet1K_V1", conv, conv_f),
        ("DINOv2 ViT-B/14", dino, dino_f),
        ("FLAIR", flair.vision_model, flair_f),
    ):
        model_result = {"encoder": name, "tasks": {}}
        for target in labels:
            if target == "Healthy":
                continue
            xtr, xv = features[itrain], features[ival]
            sweep = []
            fitted = []
            for c_value in args.c_values:
                summary, clf, scaler, score = fit_task(train_rows, val_rows, xtr, xv, target, c_value)
                summary["C"] = float(c_value)
                sweep.append(summary)
                fitted.append((summary, clf, scaler))
            # Match the prespecified development objective: maximize sensitivity
            # subject to >=90% validation specificity. This uses validation for
            # both C and threshold selection, so it is not an unbiased estimate.
            best_index = max(range(len(sweep)), key=lambda i: (
                sweep[i]["sensitivity_validation_selected_threshold"],
                sweep[i]["specificity_validation_selected_threshold"],
                sweep[i]["auroc"], -sweep[i]["C"],
            ))
            best_summary, clf, scaler = fitted[best_index]
            task_summary = dict(best_summary)
            task_summary["regularization_sweep"] = sweep
            task_summary["selection_note"] = "C and threshold selected on the same validation set to maximize sensitivity subject to >=90% specificity; highly optimistic development-only selection, not generalization performance."
            model_result["tasks"][label_names.get(target, f"source_label_{target}")] = task_summary
            args.output_dir.mkdir(parents=True, exist_ok=True)
            safe = label_names.get(target, str(target)).lower().replace(" ", "_")
            np.savez_compressed(args.output_dir/f"{name.split()[0].lower()}_{safe}_head_private.npz",
                                coef=clf.coef_.astype(np.float32), intercept=clf.intercept_.astype(np.float32),
                                scaler_mean=scaler.mean_.astype(np.float32), scaler_scale=scaler.scale_.astype(np.float32))
        results["models"][name] = model_result

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dino_weights = list((Path(torch.hub.get_dir())/"checkpoints").glob("dinov2_vitb14*.pth"))
    conv_weights_files = list((Path(torch.hub.get_dir())/"checkpoints").glob("convnext_tiny-*.pth"))
    flair_digest = hashlib.sha256()
    with (args.flair_checkpoint/"model.safetensors").open("rb") as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b""):
            flair_digest.update(chunk)
    results["checkpoint_sha256"] = {
        "dinov2": model_hash(dino_weights[0]) if len(dino_weights) == 1 else None,
        "convnext": model_hash(conv_weights_files[0]) if conv_weights_files else None,
        "flair": flair_digest.hexdigest(),
    }
    out = args.output_dir/"validation_only_summary.json"
    out.write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
