#!/usr/bin/env python3
"""Validation-only FLAIR frozen feature comparison on Zhejiang disease tasks.

Loads train and validation partitions only. Test rows are never read. Raw
predictions, embeddings, and manifests stay on the private VM; stdout contains
aggregate validation metrics only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize


class Rows(Dataset):
    def __init__(self, rows, side=512):
        self.rows, self.side = rows, side

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        with Image.open(self.rows[i]["image_path"]) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32)
        if arr.max() > 0:
            arr /= 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)
        h, w = x.shape[-2:]
        scale = max(h, w) / self.side
        size = (max(1, int(h / scale)), max(1, int(w / scale)))
        x = Resize(size, antialias=True)(x)
        x = torch.nn.functional.pad(x, (0, self.side-x.shape[-1], 0, self.side-x.shape[-2]))
        return x, i


def threshold_for_specificity(y, score, target=0.90):
    candidates = np.r_[np.unique(score), np.nextafter(np.max(score), np.inf)]
    feasible = []
    for t in candidates:
        pred = score >= t
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        spec = tn / max(tn + fp, 1)
        sens = tp / max(tp + fn, 1)
        if spec + 1e-12 >= target:
            feasible.append((sens, spec, float(t)))
    return max(feasible)[2]


def summarize(rows, score):
    eyes = {}
    for row, p in zip(rows, score):
        key = (row["patient_id"], row["eye_id"])
        rec = eyes.setdefault(key, {"y": int(row["target"]), "p": float(p)})
        if rec["y"] != int(row["target"]):
            raise ValueError("Conflicting labels within eye")
        rec["p"] = max(rec["p"], float(p))
    y = np.asarray([r["y"] for r in eyes.values()], dtype=int)
    p = np.asarray([r["p"] for r in eyes.values()], dtype=float)
    threshold = threshold_for_specificity(y, p)
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "validation_eyes": int(len(y)), "positive_eyes": int(y.sum()),
        "negative_eyes": int(len(y)-y.sum()),
        "accuracy_validation_selected_threshold": float(accuracy_score(y, pred)),
        "sensitivity_validation_selected_threshold": float(tp/max(tp+fn, 1)),
        "specificity_validation_selected_threshold": float(tn/max(tn+fp, 1)),
        "false_negatives": int(fn), "false_positives": int(fp),
        "auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "selection_note": "Optimistic validation-only candidate selection and threshold metrics; not final performance claims."
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", type=Path, required=True)
    ap.add_argument("--flair-source", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    tasks, unique = {}, {}
    for path in sorted(args.task_dir.glob("*_vs_healthy.csv")):
        with path.open(newline="", encoding="utf-8-sig") as f:
            rows = [r for r in csv.DictReader(f) if r.get("split") in {"train", "validation"}]
        if not rows:
            continue
        for r in rows:
            if not Path(r["image_path"]).is_file():
                raise FileNotFoundError("Unresolved VM image path")
            unique[r["image_id"]] = r
        tasks[path.stem] = rows
    rows = list(unique.values())
    if not rows:
        raise ValueError("No train/validation task rows")

    import sys
    sys.path.insert(0, str(args.flair_source))
    from flair import FLAIRModel
    model = FLAIRModel.from_pretrained(str(args.checkpoint), vision_pretrained=False)
    model.eval()
    model.vision_model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.vision_model.to(device)
    loader = DataLoader(Rows(rows), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    dim = int(model.vision_model.out_dim)
    features = np.empty((len(rows), dim), dtype=np.float32)
    with torch.inference_mode():
        for x, idx in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                z = model.vision_model(x)
            features[idx.numpy()] = z.float().cpu().numpy()
    index = {r["image_id"]: i for i, r in enumerate(rows)}
    digest = hashlib.sha256()
    weight_file = args.checkpoint / "model.safetensors"
    with weight_file.open("rb") as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b""):
            digest.update(chunk)
    result = {
        "status": "VALIDATION_ONLY_CANDIDATE_COMPARISON",
        "encoder": "Official FLAIR frozen vision encoder and projection",
        "checkpoint_sha256": digest.hexdigest(),
        "preprocessing": "Official FLAIR long-edge resize to 512, bottom/right zero pad, RGB [0,1]",
        "probe": "Train-only StandardScaler and L2 logistic regression (C=1); inverse patient and class weighting",
        "selection": "Train/validation only; no test rows loaded or scored",
        "tasks": {}
    }
    private = []
    for task, task_rows in tasks.items():
        tr = [r for r in task_rows if r["split"] == "train"]
        va = [r for r in task_rows if r["split"] == "validation"]
        ytr = np.asarray([int(r["target"]) for r in tr], dtype=int)
        cnt = Counter(r["patient_id"] for r in tr)
        npat = {c: len({r["patient_id"] for r in tr if int(r["target"]) == c}) for c in (0, 1)}
        sw = np.asarray([1/(npat[int(r["target"])]*cnt[r["patient_id"]]) for r in tr])
        xtr = np.stack([features[index[r["image_id"]]] for r in tr])
        scaler = StandardScaler().fit(xtr)
        clf = LogisticRegression(C=1, max_iter=2000, solver="lbfgs", random_state=20261121)
        clf.fit(scaler.transform(xtr), ytr, sample_weight=sw)
        xv = np.stack([features[index[r["image_id"]]] for r in va])
        pos = list(clf.classes_).index(1)
        sv = clf.predict_proba(scaler.transform(xv))[:, pos]
        eye_rows = summarize(va, sv)
        eye_rows["validation_patients"] = len({r["patient_id"] for r in va})
        result["tasks"][task] = eye_rows
        for row, p in zip(va, sv):
            private.append({"task": task, "image_id": row["image_id"], "score": float(p)})
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output_dir/f"{task}_head_private.npz",
                            coef=clf.coef_.astype(np.float32), intercept=clf.intercept_.astype(np.float32),
                            scaler_mean=scaler.mean_.astype(np.float32), scaler_scale=scaler.scale_.astype(np.float32))
    result["boundary"] = "Small single-source development validation only, optimistic, not external or clinical validation; test untouched."
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"validation_only_summary.json").write_text(json.dumps(result, indent=2)+"\n")
    with (args.output_dir/"validation_scores_private.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "image_id", "score"]); w.writeheader(); w.writerows(private)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
