#!/usr/bin/env python3
"""Patient-balanced partial ConvNeXt fine-tuning for Zhejiang disease-vs-all.

The script reads train/validation rows only (filters out test before resolving
image paths), trains six independent one-vs-rest logits, and keeps checkpoints
and all row-level outputs on the authorized VM. Validation is used for early
stopping and threshold selection, so results are development-only.
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
                             confusion_matrix, roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

LABELS = ["AMD", "DR", "PM", "RD", "RVO", "Uveitis"]
NAMES = {"AMD": "AMD", "DR": "Diabetic retinopathy", "PM": "Pathological myopia",
         "RD": "Retinal detachment", "RVO": "RVO", "Uveitis": "Uveitis"}


class EyeRows(Dataset):
    def __init__(self, rows, transform, training=False):
        self.rows, self.transform, self.training = rows, transform, training

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")
            if self.training and random.random() < 0.5:
                image = ImageOps.mirror(image)
            x = self.transform(image)
        y = torch.tensor([float(row["source_label"] == label) for label in LABELS])
        return x, y, i


def pool_eyes(rows, scores):
    pooled = {}
    for row, score in zip(rows, scores):
        key = (row["patient_group"], row["eye_id"])
        item = pooled.setdefault(key, {"label": row["source_label"], "scores": score.copy()})
        if item["label"] != row["source_label"]:
            raise ValueError("Conflicting labels within an eye")
        item["scores"] = np.maximum(item["scores"], score)
    return list(pooled.values())


def choose_threshold(y, score):
    candidates = np.r_[np.unique(score), np.nextafter(score.max(), np.inf)]
    feasible = []
    for threshold in candidates:
        pred = score >= threshold
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        spec = tn / max(tn + fp, 1)
        sens = tp / max(tp + fn, 1)
        if spec >= 0.90 - 1e-12:
            feasible.append((sens, spec, float(threshold)))
    return max(feasible)[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--max-seconds", type=int, default=3600)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--resume-checkpoint", type=Path,
                    help="Warm-start from the best saved model state; optimizer state is reset.")
    ap.add_argument("--prior-log", type=Path,
                    help="Optional previous stdout log whose epoch JSON rows should be retained in the summary.")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    with args.manifest.open(newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] in {"train", "validation"}]
    if not rows or {r["split"] for r in rows} != {"train", "validation"}:
        raise ValueError("Expected only nonempty train and validation rows")
    for row in rows:
        row["image_path"] = str(args.image_root / row["relative_image_path"])
        if not Path(row["image_path"]).is_file():
            raise FileNotFoundError("Unresolved image path in train/validation manifest")
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "validation"]
    if set(r["source_label"] for r in tr) != set(LABELS + ["Healthy"]):
        raise ValueError("Unexpected source-label set")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    transform = weights.transforms()
    model = convnext_tiny(weights=weights)
    model.classifier[2] = nn.Linear(768, len(LABELS))
    for p in model.parameters(): p.requires_grad_(False)
    for idx in (6, 7):
        for p in model.features[idx].parameters(): p.requires_grad_(True)
    for p in model.classifier.parameters(): p.requires_grad_(True)
    resume_state = None
    if args.resume_checkpoint:
        if not args.resume_checkpoint.is_file():
            raise FileNotFoundError(args.resume_checkpoint)
        resume_state = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        if resume_state.get("labels") != LABELS:
            raise ValueError("Resume checkpoint label order does not match this run")
        model.load_state_dict(resume_state["state_dict"], strict=True)
    model.to(device)

    # Equal sampling mass per source label, then equal mass per patient and image.
    pcounts = defaultdict(set); image_counts = Counter(r["patient_group"] for r in tr)
    for r in tr: pcounts[r["source_label"]].add(r["patient_group"])
    sample_weights = [1.0 / (len(pcounts[r["source_label"]]) * image_counts[r["patient_group"]]) for r in tr]
    sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), len(tr), replacement=True)
    train_loader = DataLoader(EyeRows(tr, transform, True), batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(EyeRows(va, transform), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW([
        {"params": list(model.features[6].parameters()) + list(model.features[7].parameters()), "lr": 5e-6},
        {"params": model.classifier.parameters(), "lr": 2e-4},
    ], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float(resume_state.get("validation_loss", float("inf"))) if resume_state else float("inf")
    best_epoch = int(resume_state.get("epoch", 0)) if resume_state else 0
    stale = 0
    best_path = args.output_dir / "convnext_partial_finetune_best_private.pt"
    history = []
    if args.prior_log and args.prior_log.is_file():
        for line in args.prior_log.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and "epoch" in record and "validation_loss" in record:
                history.append(record)
    start = time.monotonic()
    first_epoch = best_epoch + 1
    for epoch in range(first_epoch, first_epoch + args.epochs):
        model.train(); losses=[]
        for x, y, _ in train_loader:
            if time.monotonic() - start > args.max_seconds: break
            x=x.to(device, non_blocking=True); y=y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits=model(x); loss=nn.functional.binary_cross_entropy_with_logits(logits, y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach().cpu()))
        model.eval(); val_logits=[]; val_labels=[]
        with torch.inference_mode():
            for x,y,_ in val_loader:
                x=x.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    z=model(x)
                val_logits.extend(z.float().cpu().numpy()); val_labels.extend(y.numpy())
        logits=np.asarray(val_logits); targets=np.asarray(val_labels)
        val_loss=float(nn.functional.binary_cross_entropy_with_logits(torch.tensor(logits),torch.tensor(targets)).item())
        row={"epoch":epoch,"train_loss":float(np.mean(losses)) if losses else None,"validation_loss":val_loss,"elapsed_seconds":int(time.monotonic()-start)}
        history.append(row); print(json.dumps(row),flush=True)
        if val_loss < best_loss:
            best_loss,best_epoch,stale=val_loss,epoch,0
            torch.save({"state_dict":model.state_dict(),"epoch":epoch,"validation_loss":val_loss,
                        "labels":LABELS,"seed":args.seed},best_path)
        else:
            stale+=1
        if stale >= args.patience or time.monotonic()-start > args.max_seconds: break

    # Reload the validation-loss checkpoint, then select thresholds on validation eyes.
    ckpt=torch.load(best_path,map_location=device,weights_only=False); model.load_state_dict(ckpt["state_dict"]); model.eval()
    image_scores=[]
    with torch.inference_mode():
        for x,_,_ in val_loader:
            x=x.to(device,non_blocking=True)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=device.type=="cuda"):
                image_scores.extend(torch.sigmoid(model(x)).float().cpu().numpy())
    eyes=pool_eyes(va,np.asarray(image_scores)); metrics={}
    for j,label in enumerate(LABELS):
        y=np.asarray([int(e["label"]==label) for e in eyes]); score=np.asarray([e["scores"][j] for e in eyes])
        threshold=choose_threshold(y,score); pred=score>=threshold
        tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
        metrics[label]={"positive_eyes":int(y.sum()),"validation_eyes":int(len(y)),
          "sensitivity":float(tp/max(tp+fn,1)),"specificity":float(tn/max(tn+fp,1)),
          "false_negatives":int(fn),"false_positives":int(fp),
          "accuracy":float(accuracy_score(y,pred)),"auroc":float(roc_auc_score(y,score)),
          "auprc":float(average_precision_score(y,score)),"threshold_validation_only":float(threshold)}
    digest=hashlib.sha256(best_path.read_bytes()).hexdigest()
    summary={"status":"PARTIAL_CONVNEXT_FINE_TUNING_DISEASE_VS_ALL_VALIDATION_ONLY",
      "encoder":"ImageNet ConvNeXt-Tiny; features blocks 6 and 7 partially unfrozen",
      "task":"Six independent one-vs-all sigmoid outputs; seven mutually exclusive source labels used as training classes; co-occurrence not evaluated.",
      "split":"Patient-grouped Zhejiang split; only train and validation image rows loaded; test rows filtered before paths are resolved.",
      "checkpoint_epoch":int(ckpt["epoch"]),"best_validation_loss":float(ckpt["validation_loss"]),
      "history":history,"metrics":metrics,
      "resume_mode":("Warm-started from the saved best model state; optimizer/scaler state was not saved and was reset." if resume_state else "Fresh partial fine-tuning run."),
      "boundary":"Small single-source validation, used for early stopping and threshold selection. Metrics are optimistic exploratory development values; no clinical or external validation.",
      "checkpoint_sha256":digest,"checkpoint_path_private":best_path.name}
    (args.output_dir/"aggregate_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__": main()
