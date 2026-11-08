#!/usr/bin/env python3
"""Development-only frozen RETFound feature probes on train/validation rows.

Uses the official RETFound-MAE ViT-L/16 fundus checkpoint and official
ImageNet-normalized evaluation transform. The held-out internal test and the
reserved Zhejiang source are never loaded. Per-image predictions/embeddings
stay on the private VM; only aggregate metrics are intended for local review.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class FundusRows(Dataset):
    def __init__(self, rows: list[dict[str, str]]):
        self.rows = rows
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        with Image.open(self.rows[idx]["image_path"]) as image:
            x = self.transform(image.convert("RGB"))
        return x, idx


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = y.astype(int)
    pred = (p >= 0.5).astype(int)
    return {
        "n": int(len(y)), "positives": int(y.sum()),
        "accuracy_at_0_5": float(accuracy_score(y, pred)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, pred)),
        "sensitivity_at_0_5": float((pred[y == 1] == 1).mean()) if (y == 1).any() else None,
        "specificity_at_0_5": float((pred[y == 0] == 0).mean()) if (y == 0).any() else None,
        "f1_macro_at_0_5": float(f1_score(y, pred, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "auprc": float(average_precision_score(y, p)) if (y == 1).any() else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--retfound-source", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--max-iter", type=int, default=5000)
    ap.add_argument("--cached-embeddings", type=Path,
                    help="Reuse a private train/validation embedding matrix and skip image inference")
    ap.add_argument("--prior-summary", type=Path,
                    help="Original run metadata to preserve when refitting cached embeddings")
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    sys.path.insert(0, str(args.retfound_source))
    import models_vit

    with args.manifest.open(newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("split") in {"train", "valid"}]
    if not rows:
        raise ValueError("Manifest has no train/valid rows")
    for row in rows:
        p = Path(row["image_path"])
        if not p.is_file():
            raise FileNotFoundError(p)

    started = time.time()
    prior = json.loads(args.prior_summary.read_text()) if args.prior_summary else {}
    encoder_loaded, missing, unexpected = None, set(), set()
    if args.cached_embeddings:
        features = np.load(args.cached_embeddings).astype(np.float32)
        if features.shape != (len(rows), 1024):
            raise ValueError(f"Cached feature shape {features.shape} does not match {(len(rows), 1024)}")
    else:
        # Checkpoint came from the official Hugging Face publisher and its digest
        # is recorded in the run card. Allow only the Namespace object needed by
        # this published checkpoint; do not unpickle arbitrary globals.
        with torch.serialization.safe_globals([argparse.Namespace]):
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        state = checkpoint["model"]
        # Keep the pretrained CLS-token LayerNorm; the global-pool path creates
        # a new fc_norm that is uninitialized in the published MAE checkpoint.
        model = models_vit.RETFound_mae(global_pool=False, num_classes=0, img_size=224)
        load = model.load_state_dict(state, strict=False)
        missing = set(load.missing_keys)
        unexpected = set(load.unexpected_keys)
        if missing:
            raise RuntimeError(f"Unexpected missing RETFound encoder weights: {sorted(missing)}")
        encoder_prefixes = ("patch_embed.", "blocks.", "cls_token", "pos_embed")
        encoder_loaded = sum(k.startswith(encoder_prefixes) for k in state)
        if encoder_loaded < 200:
            raise RuntimeError(f"Too few encoder tensors loaded: {encoder_loaded}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval().to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        loader = DataLoader(FundusRows(rows), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers,
                            pin_memory=device.type == "cuda",
                            persistent_workers=args.workers > 0)
        features = np.zeros((len(rows), 1024), dtype=np.float32)
        with torch.inference_mode():
            for x, indices in loader:
                x = x.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                    z = model.forward_features(x)
                z = z.float().cpu().numpy()
                features[indices.numpy()] = z
                if (int(indices[-1]) + 1) % 512 < len(indices) or int(indices[-1]) + 1 == len(rows):
                    print(json.dumps({"phase": "embedding", "done": int(indices[-1]) + 1,
                                      "total": len(rows), "elapsed_s": round(time.time()-started, 1)}),
                          flush=True)

    split = np.asarray([r["split"] for r in rows])
    train, valid = split == "train", split == "valid"
    tasks = {
        "dr_grade_0_4": lambda r: int(r["dr_grade"]) if int(r["dr_grade"]) >= 0 else None,
        "referable_dr_ge2": lambda r: int(int(r["dr_grade"]) >= 2) if int(r["dr_grade"]) >= 0 else None,
        "ungradable_ddr": lambda r: int(r["ungradable"]) if int(r["ungradable"]) >= 0 else None,
        "glaucoma_hygd": lambda r: int(r["gon"]) if int(r["gon"]) >= 0 else None,
    }
    if args.cached_embeddings and not prior:
        raise ValueError("--prior-summary is required with --cached-embeddings")
    checkpoint_hash = prior.get("checkpoint_sha256")
    if not checkpoint_hash:
        digest = hashlib.sha256()
        with args.checkpoint.open("rb") as checkpoint_file:
            for chunk in iter(lambda: checkpoint_file.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        checkpoint_hash = digest.hexdigest()
    result = {
        "status": "development train/validation only; no internal test or Zhejiang rows loaded",
        "encoder": "Official RETFound-MAE ViT-L/16 fundus encoder; frozen pretrained CLS token + pretrained LayerNorm",
        "checkpoint_repo": "YukunZhou/RETFound_mae_natureCFP",
        "checkpoint_license": "CC-BY-NC-4.0; non-commercial project use authorized by user",
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_loaded_encoder_tensors": prior.get("checkpoint_loaded_encoder_tensors", encoder_loaded),
        "checkpoint_missing_keys": prior.get("checkpoint_missing_keys", sorted(missing)),
        "checkpoint_unexpected_key_count": prior.get("checkpoint_unexpected_key_count", len(unexpected)),
        "preprocessing": "official RETFound eval resize 256 bicubic, center crop 224, ImageNet RGB mean/std",
        "probe": f"training-only StandardScaler + LogisticRegression(C=1, balanced, max_iter={args.max_iter}); scaler fit only on eligible training rows",
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "rows_train_valid": len(rows), "train_rows": int(train.sum()),
        "validation_rows": int(valid.sum()), "embedding_dim": int(features.shape[1]),
        "feature_extraction_seconds": prior.get("feature_extraction_seconds", round(time.time()-started, 2)),
        "cached_embedding_refit": bool(args.cached_embeddings), "tasks": {},
    }
    private_predictions = []
    for name, label_fn in tasks.items():
        labels = [label_fn(r) for r in rows]
        eligible = np.asarray([v is not None for v in labels])
        tr, va = train & eligible, valid & eligible
        ytr = np.asarray([labels[i] for i in np.where(tr)[0]], dtype=int)
        yva = np.asarray([labels[i] for i in np.where(va)[0]], dtype=int)
        if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
            result["tasks"][name] = {"status": "insufficient class support",
                                     "n_train": int(tr.sum()), "n_valid": int(va.sum())}
            continue
        clf = LogisticRegression(max_iter=args.max_iter, class_weight="balanced", C=1.0,
                                 random_state=args.seed, solver="lbfgs")
        scaler = StandardScaler()
        xtr = scaler.fit_transform(features[tr])
        xva = scaler.transform(features[va])
        clf.fit(xtr, ytr)
        prob = clf.predict_proba(xva)
        by_source = {}
        for source in sorted(set(rows[i]["source"] for i in np.where(va)[0])):
            mask = np.asarray([bool(va[i] and rows[i]["source"] == source)
                               for i in range(len(rows))])
            if int(mask.sum()) < 2:
                continue
            yy = np.asarray([labels[i] for i in np.where(mask)[0]], dtype=int)
            if name == "dr_grade_0_4":
                pred = clf.predict(scaler.transform(features[mask]))
                by_source[source] = {
                    "n": int(mask.sum()), "accuracy": float(accuracy_score(yy, pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(yy, pred)),
                    "macro_f1": float(f1_score(yy, pred, average="macro", zero_division=0)),
                }
            else:
                pos = list(clf.classes_).index(1)
                by_source[source] = binary_metrics(yy, clf.predict_proba(scaler.transform(features[mask]))[:, pos])
        if name == "dr_grade_0_4":
            pred = clf.predict(xva)
            metrics = {
                "n_train": int(tr.sum()), "n_valid": int(va.sum()),
                "accuracy": float(accuracy_score(yva, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(yva, pred)),
                "macro_f1": float(f1_score(yva, pred, average="macro", zero_division=0)),
                "class_support_valid": {str(int(c)): int((yva == c).sum())
                                         for c in sorted(set(yva))},
            }
        else:
            pos = list(clf.classes_).index(1)
            metrics = {"n_train": int(tr.sum()), "n_valid": int(va.sum()),
                       **binary_metrics(yva, prob[:, pos])}
        metrics["by_source"] = by_source
        result["tasks"][name] = metrics
        for j, i in enumerate(np.where(va)[0]):
            score = (float(prob[j, pos]) if name != "dr_grade_0_4"
                     else float(clf.predict_proba(scaler.transform(features[[i]]))[0].max()))
            private_predictions.append({"source": rows[i]["source"],
                                        "image_id": rows[i]["image_id"],
                                        "patient_id": rows[i]["patient_id"],
                                        "task": name, "truth": int(labels[i]), "score": score})
    result["status"] = "complete"
    result["elapsed_seconds_total"] = round(time.time()-started, 2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "aggregate_summary.json").write_text(json.dumps(result, indent=2)+"\n")
    with (args.output_dir / "private_validation_predictions.csv").open("w", newline="") as f:
        fields = ["source", "image_id", "patient_id", "task", "truth", "score"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(private_predictions)
    np.save(args.output_dir / "private_embeddings.npy", features.astype(np.float16))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
