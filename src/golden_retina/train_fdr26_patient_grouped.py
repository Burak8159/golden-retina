#!/usr/bin/env python3
"""Patient-grouped FDR26 DR grading experiment; all image outputs remain private.

The first eight characters of FDR26 image_id form 1,918 groups across 3,000
rows, matching the dataset descriptor's patient count. This derived key is
used for leakage control and explicitly recorded in the run report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from openpyxl import load_workbook
from PIL import Image, ImageOps
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, mean_absolute_error, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED = 8159
GRADE_STRINGS = ("DR0 - Pas de RD", "DR1 - Minime", "DR2 - Moderee", "DR3 - Severe", "DR4 - Proliferante")
GRADE_MAP = {name: i for i, name in enumerate(GRADE_STRINGS)}
ARCHES = ("convnext_tiny", "resnet50", "efficientnet_b0")
MEAN = (.485, .456, .406)
STD = (.229, .224, .225)
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows_from_source(metadata: Path, image_root: Path):
    wb = load_workbook(metadata, read_only=True, data_only=True)
    ws = wb["Toutes"]
    rows, missing = [], 0
    for image_id, _class, grade, sex, age, diabetes_type in ws.iter_rows(min_row=2, values_only=True):
        if not image_id or grade not in GRADE_MAP:
            continue
        image_id = str(image_id).strip()
        path = image_root / f"{image_id}.jpg"
        if not path.exists():
            missing += 1
            continue
        patient_key = image_id[:8]
        with Image.open(path) as im:
            im.verify()
        rows.append({"image_id": image_id, "patient_key": patient_key, "path": path,
                     "label": GRADE_MAP[grade], "grade_text": grade})
    return rows, missing


def split_patient_groups(rows):
    y = np.asarray([r["label"] for r in rows])
    groups = np.asarray([r["patient_key"] for r in rows])
    # Choose a deterministic outer fold nearest a 20% test share by patient.
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    candidates = list(outer.split(np.zeros(len(y)), y, groups))
    tr_idx, te_idx = min(candidates, key=lambda pair: abs(len(set(groups[pair[1]])) / len(set(groups)) - .20))
    remain_y, remain_g = y[tr_idx], groups[tr_idx]
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
    candidates2 = list(inner.split(np.zeros(len(remain_y)), remain_y, remain_g))
    rel_tr, rel_va = min(candidates2, key=lambda pair: abs(len(set(remain_g[pair[1]])) / len(set(remain_g)) - .20))
    train_idx, val_idx = tr_idx[rel_tr], tr_idx[rel_va]
    train, val, test = [rows[i] for i in train_idx], [rows[i] for i in val_idx], [rows[i] for i in te_idx]
    tg = [{r["patient_key"] for r in part} for part in (train, val, test)]
    if tg[0] & tg[1] or tg[0] & tg[2] or tg[1] & tg[2]:
        raise RuntimeError("patient group crossed a frozen split")
    return train, val, test


class FundusRows(Dataset):
    def __init__(self, rows, size, train=False):
        self.rows = rows
        aug = [transforms.Resize((size, size))]
        if train:
            aug += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(8),
                    transforms.ColorJitter(.05, .05, .04, .01)]
        aug += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        self.tf = transforms.Compose(aug)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        with Image.open(r["path"]) as im:
            x = self.tf(ImageOps.exif_transpose(im).convert("RGB"))
        return x, r["label"], r["patient_key"], r["image_id"]


def make_model(name):
    if name == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, 5)
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, 5)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 5)
    else:
        raise ValueError(name)
    return m


def predict(model, loader):
    model.eval()
    probs, labels, groups, ids = [], [], [], []
    with torch.inference_mode():
        for x, y, patient, image_id in loader:
            x = x.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
            probs.extend(torch.softmax(logits.float(), 1).cpu().numpy())
            labels.extend(y.numpy())
            groups.extend(patient)
            ids.extend(image_id)
    return np.asarray(probs), np.asarray(labels), np.asarray(groups), ids


def summarize(y, probs):
    pred = probs.argmax(1)
    cm = confusion_matrix(y, pred, labels=range(5))
    per_grade = {}
    for k, name in enumerate(GRADE_STRINGS):
        tp = int(cm[k, k]); fn = int(cm[k].sum() - tp); fp = int(cm[:, k].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        per_grade[name] = {"support": int(cm[k].sum()), "sensitivity": tp / max(1, tp + fn),
                           "specificity": tn / max(1, tn + fp), "false_negatives": fn,
                           "false_positives": fp, "one_vs_rest_auroc": float(roc_auc_score((y == k).astype(int), probs[:, k]))}
    referable = y >= 2
    ref_pred = pred >= 2
    tn, fp, fn, tp = confusion_matrix(referable, ref_pred, labels=[False, True]).ravel()
    return {"n_images": int(len(y)), "n_patient_groups": int(len(set())),
            "accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(np.mean([cm[i, i] / max(1, cm[i].sum()) for i in range(5)])),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "quadratic_weighted_kappa": float(cohen_kappa_score(y, pred, labels=range(5), weights="quadratic")),
            "mean_absolute_grade_error": float(mean_absolute_error(y, pred)),
            "confusion_matrix_rows_true_DR0_to_DR4_cols_pred_DR0_to_DR4": cm.tolist(),
            "per_grade": per_grade,
            "grade2_or_worse_operational_proxy": {"support": int(referable.sum()), "sensitivity": float(tp / max(1, tp + fn)),
                "specificity": float(tn / max(1, tn + fp)), "false_negatives": int(fn), "false_positives": int(fp),
                "definition": "ALFEDIAM grade >= 2; dataset-specific operating analysis, not a validated referral rule"}}


def choose_threshold(y_true, scores, target_specificity=0.90):
    """Highest-sensitivity threshold meeting a validation-only specificity floor."""
    candidates = np.r_[np.unique(scores), np.nextafter(float(np.max(scores)), np.inf)]
    feasible = []
    for threshold in candidates:
        pred = scores >= threshold
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[False, True]).ravel()
        spec = float(tn / max(1, tn + fp))
        sens = float(tp / max(1, tp + fn))
        if spec >= target_specificity:
            feasible.append((sens, spec, float(threshold)))
    return max(feasible, key=lambda x: (x[0], x[1], -x[2]))


def binary_summary(y_true, scores, threshold):
    pred = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[False, True]).ravel()
    return {"n_images": int(len(y_true)), "positive_support": int(np.sum(y_true)),
            "threshold": float(threshold), "sensitivity": float(tp / max(1, tp + fn)),
            "specificity": float(tn / max(1, tn + fp)), "accuracy": float((tp + tn) / max(1, len(y_true))),
            "false_negatives": int(fn), "false_positives": int(fp)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exclude-root", type=Path, action="append", default=[],
                    help="Previously used source roots for exact cross-source exclusion")
    ap.add_argument("--image-size", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seconds-per-model", type=int, default=900)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    a.image_root.mkdir(parents=True, exist_ok=True)
    # Extract only images; archive paths are constrained to a flat images folder.
    with ZipFile(a.zip) as z:
        for name in z.namelist():
            p = Path(name)
            if p.suffix.lower() not in IMAGE_EXTS or p.parent.name != "images":
                continue
            target = a.image_root / p.name
            if not target.exists():
                with z.open(name) as src, target.open("wb") as dst:
                    while True:
                        buf = src.read(1 << 20)
                        if not buf: break
                        dst.write(buf)
    rows, missing = rows_from_source(a.metadata, a.image_root)
    matched_rows = len(rows)
    source_hashes = set()
    source_root_counts = {}
    for root in a.exclude_root:
        count = 0
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                try:
                    source_hashes.add(sha256(p)); count += 1
                except OSError:
                    continue
        source_root_counts[str(root)] = count
    # Collapse identical images; preserve patient cluster linkage in aggregate provenance.
    seen, hash_counts, duplicate_rows, conflicting_hashes, conflict_pairs, cross_source_rows = {}, Counter(), 0, set(), Counter(), 0
    for r in rows:
        h = sha256(r["path"])
        if h in source_hashes:
            cross_source_rows += 1
            continue
        if h in seen:
            duplicate_rows += 1
            hash_counts[h] += 1
            if seen[h]["label"] != r["label"]:
                conflicting_hashes.add(h)
                conflict_pairs[" / ".join(sorted((GRADE_STRINGS[seen[h]["label"]], GRADE_STRINGS[r["label"]])))] += 1
            continue
        r["sha256"] = h
        seen[h] = r
        hash_counts[h] = 1
    unique = [r for h, r in seen.items() if h not in conflicting_hashes]
    rows = unique
    tr, va, te = split_patient_groups(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    split = {"dataset": "FDR26 Zenodo v3", "archive_md5_expected": "be97fb330cb06cadf3926de2ff8be7c9",
             "metadata_rows_total": matched_rows + missing, "metadata_rows_with_image": matched_rows, "missing_images": missing,
             "cross_source_exact_overlap_rows_removed": cross_source_rows, "known_source_roots_scanned": source_root_counts,
             "exact_duplicate_rows_removed": duplicate_rows,
             "conflicting_duplicate_hash_groups_removed": len(conflicting_hashes),
             "rows_in_conflicting_hash_groups_excluded": int(sum(hash_counts[h] for h in conflicting_hashes)),
             "conflicting_grade_pair_counts": dict(conflict_pairs),
             "unique_hash_images_before_label_conflict_exclusion": len(seen),
             "unique_images_used_after_dedup_and_conflict_exclusion": len(rows),
             "patient_key": "first 8 characters of image_id; 1,918 unique keys across 3,000 metadata rows, matching published patient count; derived key, not a separately documented clinical identifier",
             "images_by_grade": dict(Counter(GRADE_STRINGS[r["label"]] for r in rows)),
             "groups_by_grade": {g: len({r["patient_key"] for r in rows if r["label"] == i}) for i, g in enumerate(GRADE_STRINGS)},
             "splits": {n: {"images": len(part), "patients": len({r["patient_key"] for r in part}),
                            "grades": dict(Counter(GRADE_STRINGS[r["label"]] for r in part))}
                        for n, part in (("train", tr), ("validation", va), ("test", te))},
             "split_method": "nested deterministic StratifiedGroupKFold by derived patient key; 64/16/20 target; test untouched for candidate selection",
             "boundary": "Single-center Moroccan source. Patient grouping is derived from anonymized image IDs and cross-checked against the source paper's patient count; same-eye/time semantics are not asserted. No independent external or clinical validation."}
    split["source_scan_note"] = "Exact hashes checked across extracted image files in the listed known-source roots only. Mendeley expansion files in ZIP archives were not traversed and are not covered by this cross-source scan."
    (a.out / "split_audit_aggregate.json").write_text(json.dumps(split, indent=2) + "\n")
    loaders = {"train": DataLoader(FundusRows(tr, a.image_size, True), batch_size=a.batch_size, shuffle=True, num_workers=a.workers, pin_memory=True),
               "validation": DataLoader(FundusRows(va, a.image_size), batch_size=a.batch_size, shuffle=False, num_workers=a.workers, pin_memory=True),
               "test": DataLoader(FundusRows(te, a.image_size), batch_size=a.batch_size, shuffle=False, num_workers=a.workers, pin_memory=True)}
    results = []
    for arch in ARCHES:
        model = make_model(arch).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=2e-4)
        best, best_epoch, stale, started = float("inf"), 0, 0, time.time()
        checkpoint = a.out / f"{arch}_validation_best.pt"
        class_counts = np.asarray([sum(r["label"] == i for r in tr) for i in range(5)], dtype=float)
        weights = torch.tensor((class_counts.mean() / class_counts) ** .5, dtype=torch.float32, device="cuda")
        for epoch in range(1, a.epochs + 1):
            model.train(); loss_total = 0.; count = 0
            for x, y, _, _ in loaders["train"]:
                x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True); opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(x); loss = nn.functional.cross_entropy(logits, y, weight=weights, label_smoothing=.03)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.); opt.step()
                loss_total += float(loss.detach()) * len(y); count += len(y)
            vp, vy, vg, _ = predict(model, loaders["validation"])
            val_loss = float(nn.functional.nll_loss(torch.tensor(np.log(np.maximum(vp, 1e-8)), dtype=torch.float32), torch.tensor(vy)).item())
            val = summarize(vy, vp); val["n_patient_groups"] = len(set(vg))
            print(json.dumps({"architecture": arch, "epoch": epoch, "train_loss": loss_total / max(1, count), "validation_loss": val_loss, "validation": val}), flush=True)
            if val_loss < best - 1e-4:
                best, best_epoch, stale = val_loss, epoch, 0
                torch.save({"state_dict": model.state_dict(), "architecture": arch, "epoch": epoch,
                            "image_size": a.image_size, "grades": list(GRADE_STRINGS)}, checkpoint)
            else:
                stale += 1
            if stale >= 3 or time.time() - started >= a.seconds_per_model:
                break
        results.append({"architecture": arch, "best_epoch": best_epoch, "best_validation_loss": best,
                        "checkpoint_sha256": sha256(checkpoint)})
        del model; torch.cuda.empty_cache()
    chosen = min(results, key=lambda x: x["best_validation_loss"])
    checkpoint = a.out / f"{chosen['architecture']}_validation_best.pt"
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = make_model(chosen["architecture"]); model.load_state_dict(ck["state_dict"]); model.cuda()
    # Freeze two disease-screening operating points using validation only.
    vp, vy, vg, _ = predict(model, loaders["validation"])
    val_any = 1.0 - vp[:, 0]
    val_ref = vp[:, 2:].sum(axis=1)
    any_thr = choose_threshold(vy > 0, val_any)
    ref_thr = choose_threshold(vy >= 2, val_ref)
    probs, y, groups, ids = predict(model, loaders["test"])
    test = summarize(y, probs); test["n_patient_groups"] = len(set(groups))
    test["any_dr_validation_threshold_target_90pct_specificity"] = binary_summary(y > 0, 1.0 - probs[:, 0], any_thr[2])
    test["grade2plus_validation_threshold_target_90pct_specificity"] = binary_summary(y >= 2, probs[:, 2:].sum(axis=1), ref_thr[2])
    # Patient-cluster bootstrap; all images from sampled patients travel together.
    unique_groups = np.unique(groups); rng = np.random.default_rng(SEED); boot = defaultdict(list)
    group_indices = {g: np.flatnonzero(groups == g) for g in unique_groups}
    for _ in range(2000):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_indices[g] for g in sampled])
        if len(np.unique(y[idx])) < 5: continue
        sm = summarize(y[idx], probs[idx])
        boot["accuracy"].append(sm["accuracy"]); boot["macro_f1"].append(sm["macro_f1"])
        boot["quadratic_weighted_kappa"].append(sm["quadratic_weighted_kappa"])
        boot["grade2plus_sensitivity"].append(sm["grade2_or_worse_operational_proxy"]["sensitivity"])
        b_any = binary_summary(y[idx] > 0, 1.0 - probs[idx, 0], any_thr[2])
        b_ref = binary_summary(y[idx] >= 2, probs[idx, 2:].sum(axis=1), ref_thr[2])
        boot["any_dr_sensitivity"].append(b_any["sensitivity"]); boot["any_dr_specificity"].append(b_any["specificity"])
        boot["grade2plus_sensitivity_locked_threshold"].append(b_ref["sensitivity"])
        boot["grade2plus_specificity_locked_threshold"].append(b_ref["specificity"])
    intervals = {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for k, v in boot.items()}
    # Per-image test outputs remain on the VM and are not intended for publication.
    with (a.out / "private_test_predictions.csv").open("w") as f:
        f.write("image_id,patient_key,true_grade,predicted_grade,probabilities_json\n")
        for image_id, group, yi, pi in zip(ids, groups, y, probs):
            f.write(f"{image_id},{group},{int(yi)},{int(pi.argmax())},\"{json.dumps(pi.tolist())}\"\n")
    report = {"status": "complete", "dataset": "FDR26 Zenodo v3", "selected_architecture": chosen["architecture"],
              "selected_epoch": chosen["best_epoch"], "selection": "lowest validation cross entropy among the three prespecified candidates",
              "candidate_summaries": results, "split_audit": split, "test_metrics": test,
              "validation_selected_thresholds": {"selection_rule":"maximize sensitivity subject to validation specificity >= 90%",
                  "any_dr_threshold_and_validation_sensitivity_specificity": {"threshold":any_thr[2],"sensitivity":any_thr[0],"specificity":any_thr[1]},
                  "grade2plus_threshold_and_validation_sensitivity_specificity": {"threshold":ref_thr[2],"sensitivity":ref_thr[0],"specificity":ref_thr[1]},
                  "grade2plus_definition":"ALFEDIAM grade >=2; a dataset-specific proxy, not a validated clinical referral definition"},
              "patient_cluster_bootstrap_95_ci": intervals, "boundary": split["boundary"],
              "redistribution": "raw images, patient key, and per-image test predictions stay on the authorized VM; only safe aggregates/code may be pulled."}
    (a.out / "aggregate_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "complete", "selected_architecture": chosen["architecture"],
                      "test_metrics": test, "patient_cluster_bootstrap_95_ci": intervals,
                      "boundary": report["boundary"]}), flush=True)


if __name__ == "__main__":
    main()
