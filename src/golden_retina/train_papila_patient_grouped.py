#!/usr/bin/env python3
"""Patient-grouped PAPILA glaucoma/suspect fundus experiment.

PAPILA diagnosis is eye-specific and uses 0=healthy, 1=glaucoma,
2=suspect. The model keeps those classes distinct. A secondary, explicitly
screening-oriented threshold treats glaucoma OR suspect as positive and is
chosen on validation only. All source data, split rows, predictions, and
checkpoints remain on the authorized cloud VM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED = 8159
CLASSES = ["healthy", "glaucoma", "suspect"]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_rows(root: Path):
    rows, issues = [], Counter()
    for side in ("OD", "OS"):
        sheet = root / "ClinicalData" / f"patient_data_{side.lower()}.xlsx"
        table = pd.read_excel(sheet, header=1)
        id_col = table.columns[0]
        for _, record in table.iterrows():
            raw_id = str(record[id_col]).strip()
            if not raw_id.startswith("#") or not raw_id[1:].isdigit():
                continue
            patient = raw_id[1:].zfill(3)
            path = root / "FundusImages" / f"RET{patient}{side}.jpg"
            if not path.is_file():
                issues["missing_expected_image"] += 1
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception:
                issues["decode_error"] += 1
                continue
            diagnosis = int(record["Diagnosis"])
            if diagnosis not in (0, 1, 2):
                issues["unknown_diagnosis"] += 1
                continue
            rows.append({"path": path, "patient": patient, "eye": side,
                         "label": diagnosis, "sha": sha256(path)})
    if len(rows) != 488 or len({r["patient"] for r in rows}) != 244:
        raise ValueError(f"Unexpected PAPILA map: {len(rows)} rows, {len({r['patient'] for r in rows})} patients")
    return rows, {"images_before_cross_source_filter": len(rows), "patients_before_cross_source_filter": len({r["patient"] for r in rows}),
                  "eye_label_counts": {CLASSES[k]: int(v) for k, v in Counter(r["label"] for r in rows).items()},
                  "mapping": "patient_data_od/os.xlsx header row 2; ID #NNN maps to RETNNNOD/OS.jpg; diagnosis 0 healthy, 1 glaucoma, 2 suspect",
                  "issues": dict(issues)}


def grouped_split(rows):
    y = np.asarray([r["label"] for r in rows], dtype=int)
    groups = np.asarray([r["patient"] for r in rows])
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    trainval, test = next(outer.split(np.zeros(len(y)), y, groups))
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + 1)
    tr, va = next(inner.split(np.zeros(len(trainval)), y[trainval], groups[trainval]))
    parts = {"train": [rows[trainval[i]] for i in tr],
             "validation": [rows[trainval[i]] for i in va],
             "test": [rows[i] for i in test]}
    gs = {k: {r["patient"] for r in v} for k, v in parts.items()}
    if gs["train"] & gs["validation"] or gs["train"] & gs["test"] or gs["validation"] & gs["test"]:
        raise RuntimeError("Patient groups crossed partitions")
    for name, part in parts.items():
        if set(r["label"] for r in part) != {0, 1, 2}:
            raise RuntimeError(f"A class is missing from {name}")
    return parts


class PadFundus(Dataset):
    def __init__(self, rows, train=False, size=320):
        self.rows, self.size, self.train = rows, size, train
        ops = []
        if train:
            ops = [transforms.RandomHorizontalFlip(), transforms.RandomRotation(5),
                   transforms.ColorJitter(brightness=.05, contrast=.06, saturation=.04)]
        ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        self.tf = transforms.Compose(ops)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        with Image.open(row["path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        scale = self.size / max(image.size)
        w, h = max(1, round(image.width * scale)), max(1, round(image.height * scale))
        image = image.resize((w, h), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), (0, 0, 0))
        canvas.paste(image, ((self.size - w) // 2, (self.size - h) // 2))
        return self.tf(canvas), torch.tensor(row["label"], dtype=torch.long)


def make_model(name):
    if name == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, 3)
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, 3)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 3)
    else:
        raise ValueError(name)
    return m


def predict(model, loader):
    model.eval(); probs, ys = [], []
    with torch.inference_mode():
        for x, y in loader:
            x = x.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                z = model(x)
            probs.extend(torch.softmax(z.float(), 1).cpu().numpy())
            ys.extend(y.numpy())
    return np.asarray(probs), np.asarray(ys, dtype=int)


def binary_at(y, score, threshold):
    positive = y > 0
    pred = score >= threshold
    tn, fp, fn, tp = confusion_matrix(positive, pred, labels=[False, True]).ravel()
    return {"n_eyes": int(len(y)), "positive_glaucoma_or_suspect_eyes": int(positive.sum()),
            "sensitivity": float(tp / max(1, tp + fn)), "specificity": float(tn / max(1, tn + fp)),
            "accuracy": float((tp + tn) / max(1, len(y))), "false_negatives": int(fn),
            "false_positives": int(fp), "threshold": float(threshold)}


def choose_specificity_threshold(y, scores, target=.90):
    cand = np.unique(np.r_[scores, np.nextafter(float(scores.max()), np.inf)])
    feasible = []
    for t in cand:
        m = binary_at(y, scores, float(t))
        if m["specificity"] >= target:
            feasible.append((m["sensitivity"], m["specificity"], float(t)))
    return max(feasible, key=lambda z: (z[0], z[1], -z[2]))


def summarize(y, p):
    pred = p.argmax(1)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    per = {}
    aucs, aps = [], []
    for k, name in enumerate(CLASSES):
        yy = (y == k).astype(int)
        auc = float(roc_auc_score(yy, p[:, k]))
        ap = float(average_precision_score(yy, p[:, k]))
        aucs.append(auc); aps.append(ap)
        per[name] = {"support": int(yy.sum()), "recall": float(cm[k, k] / max(1, cm[k].sum())),
                     "one_vs_rest_auroc": auc, "one_vs_rest_average_precision": ap}
    return {"n_eyes": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "macro_ovr_auroc": float(np.mean(aucs)), "macro_ovr_average_precision": float(np.mean(aps)),
            "confusion_matrix_rows_true_healthy_glaucoma_suspect": cm.tolist(), "per_class": per}


def patient_bootstrap(rows, y, p, threshold, reps=2000):
    groups = np.asarray([r["patient"] for r in rows]); unique = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(SEED + 2); vals = defaultdict(list)
    for _ in range(reps):
        samp = rng.choice(unique, len(unique), replace=True)
        ix = np.concatenate([idx_by[g] for g in samp])
        try:
            s = summarize(y[ix], p[ix]); b = binary_at(y[ix], 1 - p[ix, 0], threshold)
        except ValueError:
            continue
        for key in ("accuracy", "macro_f1", "macro_ovr_auroc"):
            vals[key].append(s[key])
        vals["glaucoma_or_suspect_sensitivity"].append(b["sensitivity"])
        vals["glaucoma_or_suspect_specificity"].append(b["specificity"])
    return {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
            for k, v in vals.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--image-size", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seconds-per-model", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--audit-only", action="store_true",
                    help="build and report the patient/duplicate/overlap audit without fitting models")
    ap.add_argument("--exclude-root", type=Path, action="append", default=[],
                    help="Prior source image roots; if either eye matches, exclude the whole PAPILA patient group")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing CPU fallback")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    rows, audit = load_rows(a.data)
    source_hashes, source_counts = set(), {}
    for root in a.exclude_root:
        n_source = 0
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                try:
                    source_hashes.add(sha256(path)); n_source += 1
                except OSError:
                    continue
        source_counts[str(root)] = n_source
    n_before_filter = len(rows)
    cross_source_rows = [r for r in rows if r["sha"] in source_hashes]
    overlapping_patients = {r["patient"] for r in cross_source_rows}
    hash_counts = Counter(r["sha"] for r in rows)
    repeated_hashes = {h for h, n in hash_counts.items() if n > 1}
    repeated_hash_patients = {r["patient"] for r in rows if r["sha"] in repeated_hashes}
    overlapping_patients |= repeated_hash_patients
    rows = [r for r in rows if r["patient"] not in overlapping_patients]
    if not rows:
        raise RuntimeError("Cross-source exact-hash screening removed all PAPILA patients")
    audit.update({"known_source_roots_scanned": source_counts,
                  "cross_source_exact_image_matches": len(cross_source_rows),
                  "within_dataset_repeated_hash_groups": len(repeated_hashes),
                  "within_dataset_repeated_hash_patient_groups_excluded": len(repeated_hash_patients)})
    # Count removed eyes without recording any row identifiers.
    removed_rows = n_before_filter - len(rows)
    audit["cross_source_exact_match_patient_groups_excluded"] = len(overlapping_patients)
    audit["cross_source_exact_match_eye_rows_excluded"] = removed_rows
    audit["images_after_cross_source_filter"] = len(rows)
    audit["patients_after_cross_source_filter"] = len({r["patient"] for r in rows})
    parts = grouped_split(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    split = {k: {"images": len(v), "patients": len({r['patient'] for r in v}),
                 "eye_labels": {CLASSES[i]: int(sum(r['label'] == i for r in v)) for i in range(3)}}
             for k, v in parts.items()}
    split_summary = {"dataset": "PAPILA v2", **audit, "split": split,
                     "split_method": "Nested stratified group split on explicit patient IDs; fixed seed 8159; 60/20/20 target.",
                     "cross_source_overlap": "Exact-hash scan must be completed and recorded before any test evaluation.",
                     "boundary": "244-patient single-center cohort. Three distinct eye labels; screening threshold groups glaucoma and suspect as positive, and is selected on validation only. Small test and no external clinical validation."}
    (a.out / "split_audit_aggregate.json").write_text(json.dumps(split_summary, indent=2) + "\n")
    if a.audit_only:
        print(json.dumps({"status": "audit_complete", "split_audit": split_summary}, indent=2), flush=True)
        return
    loaders = {k: DataLoader(PadFundus(v, k == "train", a.image_size), batch_size=a.batch_size,
                             shuffle=k == "train", num_workers=a.workers, pin_memory=True)
               for k, v in parts.items()}
    candidates = []
    for arch in ("convnext_tiny", "resnet50", "efficientnet_b0"):
        model = make_model(arch).cuda()
        counts = np.asarray([sum(r["label"] == i for r in parts["train"]) for i in range(3)], dtype=float)
        cw = torch.tensor((counts.mean() / counts) ** .5, dtype=torch.float32, device="cuda")
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=2e-4)
        best, best_epoch, stale, start = float("inf"), 0, 0, time.time()
        ckpt = a.out / f"{arch}_validation_best_private.pt"
        for epoch in range(1, a.epochs + 1):
            model.train(); total = 0.; n = 0
            for x, y in loaders["train"]:
                x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True); opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = nn.functional.cross_entropy(model(x), y, weight=cw, label_smoothing=.02)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.); opt.step()
                total += float(loss.detach()) * len(y); n += len(y)
            vp, vy = predict(model, loaders["validation"])
            val_loss = float(nn.functional.nll_loss(torch.tensor(np.log(np.maximum(vp, 1e-8)), dtype=torch.float32), torch.tensor(vy)).item())
            vm = summarize(vy, vp)
            print(json.dumps({"architecture": arch, "epoch": epoch, "train_loss": total/max(1,n),
                              "validation_loss": val_loss, "validation": vm}), flush=True)
            if val_loss < best - 1e-4:
                best, best_epoch, stale = val_loss, epoch, 0
                torch.save({"state_dict": model.state_dict(), "architecture": arch, "epoch": epoch,
                            "input_size": a.image_size, "classes": CLASSES, "seed": SEED}, ckpt)
            else:
                stale += 1
            if stale >= 3 or time.time() - start >= a.seconds_per_model:
                break
        candidates.append({"architecture": arch, "best_epoch": best_epoch,
                           "best_validation_loss": best, "checkpoint_sha256": sha256(ckpt)})
        del model; torch.cuda.empty_cache()
    chosen = min(candidates, key=lambda x: x["best_validation_loss"])
    ck = torch.load(a.out / f"{chosen['architecture']}_validation_best_private.pt", map_location="cpu", weights_only=False)
    model = make_model(chosen["architecture"]); model.load_state_dict(ck["state_dict"]); model.cuda()
    vp, vy = predict(model, loaders["validation"])
    threshold = choose_specificity_threshold(vy, 1 - vp[:, 0], .90)
    probs, y = predict(model, loaders["test"])
    test_multi = summarize(y, probs)
    test_binary = binary_at(y, 1 - probs[:, 0], threshold[2])
    test_groups = {r["patient"] for r in parts["test"]}
    ci = patient_bootstrap(parts["test"], y, probs, threshold[2])
    report = {"status": "complete", "dataset": "PAPILA v2 · Figshare GPL-3.0+",
              "task": "Eye-level healthy/glaucoma/suspect classification; secondary glaucoma-or-suspect screening threshold",
              "selected_architecture": chosen["architecture"], "selected_epoch": chosen["best_epoch"],
              "candidate_validation": candidates, "split_audit": split_summary,
              "validation_threshold": {"target_specificity": .90, "threshold": threshold[2],
                  "observed_validation_sensitivity": threshold[0], "observed_validation_specificity": threshold[1],
                  "positive_definition": "clinical diagnosis glaucoma or suspect; suspect is separately retained in 3-class results"},
              "test_three_class_metrics": test_multi, "test_screen_operating_metrics": test_binary,
              "test_patient_groups": len(test_groups), "patient_cluster_bootstrap_95_ci": ci,
              "boundary": "Single-center, 244-patient cohort and small held-out patient sample. Performance is exploratory and not independent external or clinical validation. Suspect is not diagnosed glaucoma; screening-positive grouping is explicitly a sensitivity-oriented proxy. Raw data, checkpoints and eye-level predictions remain on VM."}
    (a.out / "aggregate_report.json").write_text(json.dumps(report, indent=2) + "\n")
    # Keep row-level outputs private on the VM.
    with (a.out / "private_eye_predictions.csv").open("w") as f:
        f.write("patient_id,eye,true_label,predicted_label,probabilities_json\n")
        for row, yi, pi in zip(parts["test"], y, probs):
            f.write(f"{row['patient']},{row['eye']},{CLASSES[int(yi)]},{CLASSES[int(pi.argmax())]},\"{json.dumps(pi.tolist())}\"\n")
    print(json.dumps({"status": "complete", "selected_architecture": chosen["architecture"],
                      "test_three_class_metrics": test_multi, "test_screen_metrics": test_binary,
                      "patient_cluster_bootstrap_95_ci": ci, "test_patients": len(test_groups)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
