#!/usr/bin/env python3
"""Frozen ImageNet ConvNeXt-Tiny linear probes for private UWF disease tasks.

One binary head is fit per task. Only train rows fit the head; validation rows
select the specificity-constrained threshold in the separate evaluator; test
rows are scored once and remain a single-source development estimate.
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
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


class ImageRows(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, index


def sha_model(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261121)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    task_files = sorted(args.task_dir.glob("*_vs_healthy.csv"))
    if not task_files:
        raise FileNotFoundError(f"No disease task manifests in {args.task_dir}")

    tasks = {}
    all_rows_by_id = {}
    for path in task_files:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"Empty task manifest: {path.name}")
        required = {"image_id", "image_path", "patient_id", "eye_id", "target", "split"}
        if not required.issubset(rows[0]):
            raise ValueError(f"Unexpected task schema in {path.name}")
        if len({row["image_id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate image IDs in {path.name}")
        for row in rows:
            if not Path(row["image_path"]).is_file():
                raise FileNotFoundError("One or more manifest image paths are unresolved")
            all_rows_by_id[row["image_id"]] = row
        tasks[path.stem] = rows

    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = convnext_tiny(weights=weights)
    model.classifier = torch.nn.Identity()
    model.to(device).eval()
    transform = weights.transforms()
    unique_rows = list(all_rows_by_id.values())
    loader = DataLoader(ImageRows(unique_rows, transform), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers,
                        pin_memory=device.type == "cuda")
    features = np.empty((len(unique_rows), 768), dtype=np.float32)
    model_hash = sha_model(model)
    with torch.inference_mode():
        for images, indices in loader:
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(images)
            else:
                output = model(images)
            output = torch.flatten(output, start_dim=1)
            features[indices.numpy()] = output.float().cpu().numpy()
    feature_index = {row["image_id"]: i for i, row in enumerate(unique_rows)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = {
        "status": "FROZEN_IMAGENET_CONVNEXT_TINY_LINEAR_PROBES",
        "encoder": "torchvision ConvNeXt-Tiny ImageNet1K_V1; fully frozen",
        "encoder_state_sha256": model_hash,
        "transform": "torchvision ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()",
        "head": "L2 logistic regression, C=1.0, lbfgs; trained on train images only",
        "sample_weighting": "Each training patient contributes equal total weight within its class; classes receive equal total weight.",
        "score_semantics": "Uncalibrated model score, not disease risk or diagnosis.",
        "task_manifests": {},
        "boundary": "Retrospective single-source development probe. The Zhejiang DR cohort has prior project outputs; no task is independent clinical validation.",
    }
    for task_name, rows in tasks.items():
        train = [r for r in rows if r["split"] == "train"]
        if {r["target"] for r in train} != {"0", "1"}:
            raise ValueError(f"Both classes required in train: {task_name}")
        y_train = np.asarray([int(r["target"]) for r in train], dtype=np.int64)
        x_train = np.stack([features[feature_index[r["image_id"]]] for r in train])
        # Equal total contribution per patient within each class.
        class_patients = {}
        patient_counts = Counter(r["patient_id"] for r in train)
        for label in (0, 1):
            class_patients[label] = len({r["patient_id"] for r in train if int(r["target"]) == label})
        sample_weight = np.asarray([
            1.0 / (class_patients[int(r["target"])] * patient_counts[r["patient_id"]])
            for r in train
        ], dtype=np.float64)
        clf = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=2000,
                                 random_state=args.seed)
        clf.fit(x_train, y_train, sample_weight=sample_weight)
        score_rows = []
        for row in rows:
            score = float(clf.predict_proba(features[feature_index[row["image_id"]]][None, :])[0, 1])
            score_rows.append({"image_id": row["image_id"], "score": score})
        score_path = args.output_dir / f"{task_name}_scores_private.csv"
        with score_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("image_id", "score"))
            writer.writeheader()
            writer.writerows(score_rows)
        # Store only the compact linear head; no source rows or embeddings.
        np.savez_compressed(args.output_dir / f"{task_name}_head_private.npz",
                            coef=clf.coef_.astype(np.float32), intercept=clf.intercept_.astype(np.float32))
        run["task_manifests"][task_name] = {
            "images": len(rows),
            "patients": len({r["patient_id"] for r in rows}),
            "train_images": len(train),
            "train_patients": len({r["patient_id"] for r in train}),
            "train_class_patient_counts": {"negative": class_patients[0], "positive": class_patients[1]},
            "score_file_sha256": hashlib.sha256(score_path.read_bytes()).hexdigest(),
            "head_file": f"{task_name}_head_private.npz",
        }
    summary_path = args.output_dir / "probe_run_summary.json"
    summary_path.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(run, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
