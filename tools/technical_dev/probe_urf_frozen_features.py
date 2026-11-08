#!/usr/bin/env python3
"""UrFound ViT-B/16 frozen-feature probe on the frozen development split.

The official public checkpoint is loaded with weights_only=True and only the
official encoder is used. Image-level outputs and embeddings stay on the
authorized VM; the aggregate JSON is the only intended export.
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
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--urf-source", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-iter", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261108)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sys.path.insert(0, str(args.urf_source))
    from finetune import models_vit
    from util.pos_embed import interpolate_pos_embed

    with args.manifest.open(newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("split") in {"train", "valid"}]
    if not rows:
        raise ValueError("Manifest has no train/valid rows")
    for row in rows:
        if not Path(row["image_path"]).is_file():
            raise FileNotFoundError(row["image_path"])

    # The official checkpoint stores argparse.Namespace in its metadata.
    # Allowlist that metadata type only; never fall back to unsafe pickle load.
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise RuntimeError("Official UrFound checkpoint has no model state dictionary")
    model = models_vit.vit_base_patch16(
        img_size=224, num_classes=0, drop_path_rate=0.0, global_pool=True)
    interpolate_pos_embed(model, state)
    load = model.load_state_dict(state, strict=False)
    if set(load.missing_keys) != {"fc_norm.weight", "fc_norm.bias"}:
        raise RuntimeError(f"Unexpected missing UrFound encoder keys: {load.missing_keys}")
    allowed_unexpected = {"mask_token", "decoder_pos_embed", "norm.weight", "norm.bias"}
    unexpected = set(load.unexpected_keys)
    unexpected_research_heads = {
        key for key in unexpected - allowed_unexpected
        if key.startswith(("decoder_", "bert_encoder.", "bert_mlp."))
    }
    if unexpected - allowed_unexpected != unexpected_research_heads:
        raise RuntimeError(f"Unexpected checkpoint keys: {load.unexpected_keys[:20]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    loader = DataLoader(FundusRows(rows), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda",
                        persistent_workers=args.workers > 0)
    features = np.zeros((len(rows), 768), dtype=np.float32)
    started = time.time()
    with torch.inference_mode():
        for x, indices in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=device.type == "cuda"):
                z = model.forward_features(x)
            features[indices.numpy()] = z.float().cpu().numpy()
            done = int(indices[-1]) + 1
            if done % 512 < len(indices) or done == len(rows):
                print(json.dumps({"phase": "embedding", "done": done,
                                  "total": len(rows), "elapsed_s": round(time.time()-started, 1)}),
                      flush=True)

    split = np.asarray([r["split"] for r in rows])
    train, valid = split == "train", split == "valid"
    task_fns = {
        "dr_grade_0_4": lambda r: int(r["dr_grade"]) if int(r["dr_grade"]) >= 0 else None,
        "referable_dr_ge2": lambda r: int(int(r["dr_grade"]) >= 2) if int(r["dr_grade"]) >= 0 else None,
        "ungradable_ddr": lambda r: int(r["ungradable"]) if int(r["ungradable"]) >= 0 else None,
        "glaucoma_hygd": lambda r: int(r["gon"]) if int(r["gon"]) >= 0 else None,
    }
    result = {
        "status": "completed; development train/validation only; no internal test or Zhejiang rows loaded",
        "encoder": "Official UrFound ViT-Base/16 fundus encoder; frozen global-average pooled representation",
        "checkpoint_source": "Official yukkai/UrFound repository link to public yyyyk/UrFound/urfound_mm.pth",
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_missing_keys": list(load.missing_keys),
        "checkpoint_unexpected_key_count": len(load.unexpected_keys),
        "checkpoint_tensor_count": len(state),
        "pretraining_provenance": "Model card reports FLAIR pretraining and RETFound fine-tuning; exact source-level overlap with current development rows is not established.",
        "preprocessing": "Official UrFound evaluation transform: bicubic resize 256, center crop 224, ImageNet mean/std",
        "probe": f"training-only StandardScaler + balanced LogisticRegression(C=1, max_iter={args.max_iter}); scaler fit only on eligible train rows",
        "manifest_sha256": sha256(args.manifest),
        "rows_train_valid": len(rows), "train_rows": int(train.sum()),
        "validation_rows": int(valid.sum()), "embedding_dim": int(features.shape[1]),
        "feature_extraction_seconds": round(time.time()-started, 2), "tasks": {},
    }
    private_predictions = []
    for name, label_fn in task_fns.items():
        labels = [label_fn(r) for r in rows]
        eligible = np.asarray([v is not None for v in labels])
        tr, va = train & eligible, valid & eligible
        ytr = np.asarray([labels[i] for i in np.where(tr)[0]], dtype=int)
        yva = np.asarray([labels[i] for i in np.where(va)[0]], dtype=int)
        if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
            result["tasks"][name] = {"status": "insufficient class support",
                                     "n_train": int(tr.sum()), "n_valid": int(va.sum())}
            continue
        scaler = StandardScaler()
        xtr, xva = scaler.fit_transform(features[tr]), scaler.transform(features[va])
        clf = LogisticRegression(max_iter=args.max_iter, class_weight="balanced", C=1.0,
                                 random_state=args.seed, solver="lbfgs")
        clf.fit(xtr, ytr)
        probabilities = clf.predict_proba(xva)
        by_source = {}
        if name == "dr_grade_0_4":
            pred = clf.predict(xva)
            metrics = {"n_train": int(tr.sum()), "n_valid": int(va.sum()),
                       "accuracy": float(accuracy_score(yva, pred)),
                       "balanced_accuracy": float(balanced_accuracy_score(yva, pred)),
                       "macro_f1": float(f1_score(yva, pred, average="macro", zero_division=0)),
                       "class_support_valid": {str(int(c)): int((yva == c).sum()) for c in sorted(set(yva))}}
            for source in sorted({rows[i]["source"] for i in np.where(va)[0]}):
                mask = np.asarray([bool(va[i] and rows[i]["source"] == source) for i in range(len(rows))])
                yy = np.asarray([labels[i] for i in np.where(mask)[0]], dtype=int)
                pp = clf.predict(scaler.transform(features[mask]))
                by_source[source] = {"n": int(mask.sum()), "accuracy": float(accuracy_score(yy, pp)),
                                     "balanced_accuracy": float(balanced_accuracy_score(yy, pp)),
                                     "macro_f1": float(f1_score(yy, pp, average="macro", zero_division=0))}
            for j, i in enumerate(np.where(va)[0]):
                private_predictions.append({"source": rows[i]["source"], "image_id": rows[i]["image_id"],
                    "patient_id": rows[i]["patient_id"], "task": name, "truth": int(labels[i]),
                    "score": float(probabilities[j].max())})
        else:
            pos = list(clf.classes_).index(1)
            metrics = {"n_train": int(tr.sum()), "n_valid": int(va.sum()),
                       **binary_metrics(yva, probabilities[:, pos])}
            for source in sorted({rows[i]["source"] for i in np.where(va)[0]}):
                mask = np.asarray([bool(va[i] and rows[i]["source"] == source) for i in range(len(rows))])
                yy = np.asarray([labels[i] for i in np.where(mask)[0]], dtype=int)
                pp = clf.predict_proba(scaler.transform(features[mask]))[:, pos]
                by_source[source] = binary_metrics(yy, pp)
            for j, i in enumerate(np.where(va)[0]):
                private_predictions.append({"source": rows[i]["source"], "image_id": rows[i]["image_id"],
                    "patient_id": rows[i]["patient_id"], "task": name, "truth": int(labels[i]),
                    "score": float(probabilities[j, pos])})
        metrics["by_source"] = by_source
        result["tasks"][name] = metrics

    result["elapsed_seconds_total"] = round(time.time()-started, 2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "aggregate_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output_dir / "private_validation_predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "image_id", "patient_id", "task", "truth", "score"])
        writer.writeheader(); writer.writerows(private_predictions)
    np.save(args.output_dir / "private_embeddings.npy", features.astype(np.float16))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
