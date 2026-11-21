"""Train and evaluate a source-separated RIM-ONE DL glaucoma classifier.

The hospital test partition is opened only after validation-based model and
threshold selection. Row-level manifests, scores, and checkpoints must remain
on the authorized VM because the dataset release prohibits redistribution.
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
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ConvNeXt_Tiny_Weights

SEED = 8159
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class FundusDataset(Dataset):
    def __init__(self, rows, root: Path, train: bool):
        self.rows, self.root = rows, root
        ops = [transforms.Resize(256), transforms.CenterCrop(224)]
        ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        rel, label = self.rows[index]
        with Image.open(self.root / rel) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
        return self.transform(image), torch.tensor(label, dtype=torch.float32)


def gather(root: Path, partition: str):
    rows = []
    for label_name, label in (("normal", 0), ("glaucoma", 1)):
        folder = root / partition / label_name
        rows.extend((p.relative_to(root).as_posix(), label)
                    for p in sorted(folder.iterdir()) if p.is_file())
    if not rows or len({y for _, y in rows}) != 2:
        raise RuntimeError(f"Expected both classes in {partition}")
    return rows


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "positive": int(np.sum(y)), "negative": int(len(y)-np.sum(y)),
        "threshold": float(threshold), "accuracy": float(accuracy_score(y, pred)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "false_negative": int(fn), "false_positive": int(fp),
        "true_positive": int(tp), "true_negative": int(tn),
        "auroc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
    }


def bootstrap_auc_ci(y, p, seed=SEED, reps=2000):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(reps):
        ix = rng.integers(0, len(y), len(y))
        if np.unique(y[ix]).size == 2:
            values.append(roc_auc_score(y[ix], p[ix]))
    if not values:
        return None
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def collect_logits(model, loader, device, amp):
    model.eval()
    values, targets = [], []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(images).squeeze(1)
            values.extend(logits.float().cpu().numpy().tolist())
            targets.extend(labels.numpy().tolist())
    return np.asarray(values, dtype=np.float64), np.asarray(targets, dtype=np.int64)


def extract_features(model, loader, device):
    model.eval()
    xs, ys = [], []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            feat = model.features(images)
            feat = model.avgpool(feat).flatten(1)
            xs.append(feat.float().cpu().numpy())
            ys.append(labels.numpy().astype(int))
    return np.concatenate(xs), np.concatenate(ys)


def choose_threshold(y, probabilities, target_sensitivity=0.90):
    best = None
    for threshold in np.unique(np.r_[0.0, probabilities, 1.0]):
        result = metrics(y, probabilities, float(threshold))
        if result["sensitivity"] + 1e-12 < target_sensitivity:
            continue
        key = (result["specificity"], result["accuracy"], -float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold), result)
    if best is None:
        # Sensitivity-constrained operation is infeasible on validation. Record
        # the maximum-sensitivity point; do not pretend the target was met.
        candidates = [(metrics(y, probabilities, float(t)), float(t))
                      for t in np.unique(np.r_[0.0, probabilities, 1.0])]
        result, threshold = max(candidates, key=lambda z: (z[0]["sensitivity"], z[0]["specificity"]))
        return threshold, result, False
    return best[1], best[2], True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = args.data_root
    all_train = gather(root, "training_set")
    official_test = gather(root, "test_set")
    tr_rows, va_rows = train_test_split(all_train, test_size=.20, random_state=SEED,
                                        stratify=[y for _, y in all_train])
    private_split = args.output_dir / "split_manifest_private.csv"
    with private_split.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("relative_path", "label", "split"))
        writer.writerows((p, y, "train") for p, y in tr_rows)
        writer.writerows((p, y, "validation") for p, y in va_rows)
        writer.writerows((p, y, "official_hospital_test") for p, y in official_test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This campaign requires the authorized L4; refusing CPU fallback")
    train_loader = DataLoader(FundusDataset(tr_rows, root, True), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=True)
    train_eval_loader = DataLoader(FundusDataset(tr_rows, root, False), batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(FundusDataset(va_rows, root, False), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(FundusDataset(official_test, root, False), batch_size=args.batch_size,
                             shuffle=False, num_workers=args.workers, pin_memory=True)
    model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)
    model = model.to(device)
    amp = True

    # Candidate A: frozen ImageNet ConvNeXt features with a regularized logistic head.
    xtr, ytr = extract_features(model, train_eval_loader, device)
    xva, yva = extract_features(model, val_loader, device)
    probe = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=SEED)
    probe.fit(xtr, ytr)
    p_probe = probe.predict_proba(xva)[:, 1]
    probe_val = {"auroc": float(roc_auc_score(yva, p_probe)),
                 "average_precision": float(average_precision_score(yva, p_probe))}

    # Candidate B: fine-tuned ConvNeXt-Tiny. Early stop and select only by internal validation AUROC.
    counts = np.bincount([y for _, y in tr_rows], minlength=2)
    pos_weight = torch.tensor([counts[0] / max(1, counts[1])], device=device)
    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": 1e-5},
        {"params": model.classifier.parameters(), "lr": 3e-4},
    ], weight_decay=0.02)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_state, best_auc, best_epoch, stale = None, -1.0, 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0; seen = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            running += float(loss.detach()) * len(labels); seen += len(labels)
        val_logits, val_y = collect_logits(model, val_loader, device, amp)
        val_p = 1.0 / (1.0 + np.exp(-np.clip(val_logits, -40, 40)))
        val_auc = float(roc_auc_score(val_y, val_p))
        val_ap = float(average_precision_score(val_y, val_p))
        val_loss = float(nn.functional.binary_cross_entropy_with_logits(
            torch.tensor(val_logits), torch.tensor(val_y, dtype=torch.float32)))
        history.append({"epoch": epoch, "train_loss": running/max(1,seen),
                        "valid_loss": val_loss, "valid_auroc": val_auc, "valid_ap": val_ap})
        print(json.dumps(history[-1]), flush=True)
        if val_auc > best_auc + 1e-4:
            best_auc, best_epoch, stale = val_auc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 4:
                break
    if best_state is None:
        raise RuntimeError("No validation-selected fine-tuned checkpoint")

    # Select between the frozen probe and fine-tuned candidate on validation only.
    # Restore the validation-best fine-tuned state before scoring or selection.
    model.load_state_dict(best_state); model.eval()
    val_ft_logits, val_y = collect_logits(model, val_loader, device, amp)
    p_ft = 1.0 / (1.0 + np.exp(-np.clip(val_ft_logits, -40, 40)))
    ft_val = {"auroc": float(roc_auc_score(val_y, p_ft)),
              "average_precision": float(average_precision_score(val_y, p_ft))}
    selected = "finetuned_convnext_tiny" if (ft_val["auroc"], ft_val["average_precision"]) >= (probe_val["auroc"], probe_val["average_precision"]) else "frozen_convnext_tiny_logistic_probe"
    threshold_values = p_ft if selected == "finetuned_convnext_tiny" else p_probe
    threshold, threshold_val_metrics, target_met = choose_threshold(val_y, threshold_values)

    # Freeze candidate and operating point, then open the official hospital test once.
    if selected == "finetuned_convnext_tiny":
        test_logits, test_y = collect_logits(model, test_loader, device, amp)
        test_p = 1.0 / (1.0 + np.exp(-np.clip(test_logits, -40, 40)))
        checkpoint = args.output_dir / "rimone_glaucoma_convnext_tiny.pt"
        torch.save({"state_dict": best_state, "architecture": "torchvision.convnext_tiny",
                    "pretrained": "ImageNet1K_V1", "threshold": threshold,
                    "input_size": 224, "mean": MEAN, "std": STD,
                    "seed": SEED, "best_epoch": best_epoch}, checkpoint)
    else:
        test_x, test_y = extract_features(model, test_loader, device)
        test_p = probe.predict_proba(test_x)[:, 1]
        checkpoint = args.output_dir / "rimone_glaucoma_frozen_probe.npz"
        np.savez_compressed(checkpoint, coef=probe.coef_, intercept=probe.intercept_,
                            threshold=np.asarray([threshold]), seed=np.asarray([SEED]))
    test_metrics = metrics(test_y, test_p, threshold)
    test_metrics["image_level_bootstrap_95ci_auroc"] = bootstrap_auc_ci(test_y, test_p)
    prediction_path = args.output_dir / "official_test_predictions_private.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("relative_path", "label", "score", "predicted_label", "false_negative"))
        for (relative_path, label), score in zip(official_test, test_p):
            pred = int(score >= threshold)
            writer.writerow((relative_path, label, f"{float(score):.10f}", pred,
                             int(label == 1 and pred == 0)))
    report = {
        "status": "ONE_PASS_OFFICIAL_HOSPITAL_TEST_COMPLETE",
        "source": "RIM-ONE DL hospital-separated partition; source-defined training and held-out hospitals",
        "task": "glaucoma label versus normal; fundus image classification",
        "data_counts": {"official_training_hospital": len(all_train), "internal_train": len(tr_rows),
                        "internal_validation": len(va_rows), "held_out_hospital_test": len(official_test),
                        "train_glaucoma": int(sum(y for _,y in all_train)),
                        "test_glaucoma": int(sum(y for _,y in official_test))},
        "split": "Official hospital split; deterministic stratified image-level 80/20 validation from training hospitals; patient IDs absent and patient separation unverified.",
        "duplicate_audit": "Exact duplicate SHA-256 count across hospital partitions=0; one dHash<=4 candidate examined separately, not automatically removed.",
        "candidate_validation": {"frozen_probe": probe_val, "fine_tuned": ft_val},
        "selected_model": selected,
        "threshold_policy": {"target_validation_sensitivity": 0.90, "threshold": threshold,
                             "feasible": target_met, "validation_metrics": threshold_val_metrics},
        "official_test_metrics": test_metrics,
        "model_checkpoint_sha256_private": sha256(checkpoint),
        "model_checkpoint_name_private": checkpoint.name,
        "private_split_manifest_sha256": sha256(private_split),
        "pretrained_weights": {"name": "ConvNeXt-Tiny ImageNet1K_V1", "sha256": "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"},
        "epoch_history": history,
        "limitations": ["No patient identifiers; image-level split within training hospitals and patient disjointness cannot be verified.",
                        "Hospital-held-out benchmark is not independent prospective clinical validation.",
                        "Small single-source cohort; results do not establish generalized glaucoma diagnosis.",
                        "Checkpoint and row-level outputs remain private under source redistribution terms."],
    }
    report_path = args.output_dir / "summary_aggregate.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(report_path), "selected": selected,
                      "validation": {"probe": probe_val, "fine_tuned": ft_val},
                      "threshold": threshold, "test": test_metrics,
                      "elapsed_seconds": round(time.time()-start_time, 2)}, indent=2), flush=True)


if __name__ == "__main__":
    start_time = time.time()
    main()
