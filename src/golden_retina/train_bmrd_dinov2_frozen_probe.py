"""Frozen DINOv2 feature probe for exploratory BMRD site-transfer comparisons.

The BMRD held-out sites have no patient identifiers. This script keeps site
separation, uses only training-site validation to select logistic C, and writes
row-level outputs only to the private VM result directory. The test sites have
already been opened by the CNN comparison, so these are secondary exploratory
comparisons, not fresh confirmatory estimates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SEED = 8159
TASKS = {
    "dr": {"train": "Shiromoni BNSB Hospital", "test": "Khulna Laser Clinic", "prefix": "DR_Level_"},
    "amd": {"train": "Shiromoni BNSB Hospital", "test": "Khulna Laser Clinic", "prefix": "AMD_"},
    "glaucoma": {"train": "Khulna Laser Clinic", "test": "Anwar khan Modern Medical", "prefix": "Glaucoma_Level_"},
}


class Fundus(Dataset):
    def __init__(self, paths: list[str]):
        self.paths = paths
        self.tf = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            x = self.tf(ImageOps.exif_transpose(im).convert("RGB"))
        return x, i


def metrics(y: np.ndarray, probs: np.ndarray, labels: list[str]) -> dict:
    pred = probs.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=np.arange(len(labels)))
    per_class = {}
    aucs = []
    for i, label in enumerate(labels):
        tp = int(cm[i, i]); fn = int(cm[i].sum() - tp)
        fp = int(cm[:, i].sum() - tp); tn = int(len(y) - tp - fn - fp)
        auc = None
        try:
            auc = float(roc_auc_score((y == i).astype(int), probs[:, i]))
            aucs.append(auc)
        except ValueError:
            pass
        per_class[label] = {
            "support": int((y == i).sum()), "sensitivity": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "false_negatives": fn, "false_positives": fp, "auroc_ovr": auc,
        }
    return {
        "n_images": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "macro_auroc_ovr": float(np.mean(aucs)) if aucs else None,
        "confusion_matrix_rows_true_columns_pred": cm.tolist(), "classes": per_class,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--dinov2-source", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=24)
    a = ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    with a.manifest.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    selected: dict[str, dict] = {}
    task_rows = {}
    for task, cfg in TASKS.items():
        rs = [r for r in rows if r["label"].startswith(cfg["prefix"]) and r["hospital"] in {cfg["train"], cfg["test"]}]
        names = sorted({r["label"] for r in rs})
        if len(names) < 2:
            task_rows[task] = {"status": "insufficient_classes"}
            continue
        lut = {n: i for i, n in enumerate(names)}
        tr = [r for r in rs if r["hospital"] == cfg["train"]]
        te = [r for r in rs if r["hospital"] == cfg["test"]]
        train_ids, val_ids = train_test_split(np.arange(len(tr)), test_size=.2, random_state=SEED,
                                               stratify=[lut[r["label"]] for r in tr])
        task_rows[task] = {"names": names, "train": [tr[i] for i in train_ids],
                           "valid": [tr[i] for i in val_ids], "test": te,
                           "lut": lut, "train_site": cfg["train"], "test_site": cfg["test"]}
        for r in tr + te:
            selected[r["path"]] = r
    paths = list(selected)
    for p in paths:
        if not Path(p).is_file():
            raise FileNotFoundError(p)
    model = torch.hub.load(str(a.dinov2_source), "dinov2_vitb14", source="local",
                           pretrained=True, trust_repo=True).eval().cuda()
    for param in model.parameters():
        param.requires_grad_(False)
    ds = Fundus(paths)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
                        pin_memory=True, persistent_workers=a.workers > 0)
    feats = np.zeros((len(paths), 768), dtype=np.float32)
    start = time.time()
    with torch.inference_mode():
        for x, ix in loader:
            x = x.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                z = model.forward_features(x)["x_norm_clstoken"]
            feats[ix.numpy()] = z.float().cpu().numpy()
    index = {p: i for i, p in enumerate(paths)}
    results = {}
    for task, data in task_rows.items():
        if "names" not in data:
            results[task] = data
            continue
        labels = data["names"]; lut = data["lut"]
        ix = {split: np.array([index[r["path"]] for r in data[split]], dtype=int)
              for split in ("train", "valid", "test")}
        y = {split: np.array([lut[r["label"]] for r in data[split]], dtype=int)
             for split in ("train", "valid", "test")}
        candidates = []
        for c in (.01, .1, 1.0, 10.0):
            scaler = StandardScaler().fit(feats[ix["train"]])
            model_head = LogisticRegression(C=c, max_iter=3000, class_weight="balanced",
                                            random_state=SEED, solver="lbfgs")
            model_head.fit(scaler.transform(feats[ix["train"]]), y["train"])
            pv = model_head.predict_proba(scaler.transform(feats[ix["valid"]]))
            score = float(f1_score(y["valid"], pv.argmax(1), average="macro", zero_division=0))
            candidates.append((score, c))
        best_score, chosen_c = max(candidates)
        scaler = StandardScaler().fit(feats[ix["train"]])
        head = LogisticRegression(C=chosen_c, max_iter=3000, class_weight="balanced",
                                  random_state=SEED, solver="lbfgs")
        head.fit(scaler.transform(feats[ix["train"]]), y["train"])
        pt = head.predict_proba(scaler.transform(feats[ix["test"]]))
        results[task] = {
            "status": "exploratory_secondary_comparison_test_previously_opened_by_CNN",
            "train_site": data["train_site"], "validation_site": data["train_site"],
            "test_site": data["test_site"], "class_names": labels,
            "n_train": int(len(y["train"])), "n_validation": int(len(y["valid"])),
            "n_test": int(len(y["test"])), "selected_C_validation_macro_f1": chosen_c,
            "validation_macro_f1": best_score, "validation_candidates": [
                {"C": c, "macro_f1": s} for s, c in candidates],
            "test_metrics_secondary": metrics(y["test"], pt, labels),
        }
        print(json.dumps({"task": task, "C": chosen_c,
                          "metrics": results[task]["test_metrics_secondary"]}), flush=True)
    ckpts = list((Path(torch.hub.get_dir()) / "checkpoints").glob("dinov2_vitb14*.pth"))
    h = hashlib.sha256(ckpts[0].read_bytes()).hexdigest() if len(ckpts) == 1 else None
    report = {
        "status": "completed_secondary_exploratory_probe",
        "encoder": "Official DINOv2 ViT-B/14 frozen normalized CLS embeddings; 768 dimensions",
        "weights_sha256": h, "preprocessing": "bicubic resize 256, center crop224, ImageNet normalization",
        "selection": "C in {0.01,0.1,1,10} selected only on training-hospital image-level validation by macro-F1",
        "known_boundary": "BMRD has no patient identifiers; test hospitals have already been opened by the CNN comparison. This is a secondary exploratory comparison, not independent confirmation.",
        "manifest_sha256": hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        "n_unique_image_paths": len(paths), "elapsed_seconds": round(time.time() - start, 1),
        "tasks": results,
    }
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / "dinov2_frozen_aggregate_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(a.output_dir / "dinov2_frozen_aggregate_report.json"),
                      "elapsed_seconds": report["elapsed_seconds"], "weights_sha256": h}), flush=True)


if __name__ == "__main__":
    main()
