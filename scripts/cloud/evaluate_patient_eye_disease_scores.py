#!/usr/bin/env python3
"""Evaluate disease-vs-Healthy scores at eye level with patient-cluster CIs.

Inputs are private VM-only CSVs. Output is an aggregate JSON report; no image,
patient, eye, or row-level score data are written by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, roc_auc_score)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def choose_threshold(y: np.ndarray, score: np.ndarray, target_specificity: float) -> float:
    candidates = np.r_[np.unique(score), np.nextafter(np.max(score), np.inf)]
    eligible = []
    for threshold in candidates:
        pred = score >= threshold
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        specificity = tn / max(tn + fp, 1)
        sensitivity = tp / max(tp + fn, 1)
        if specificity + 1e-12 >= target_specificity:
            eligible.append((sensitivity, specificity, float(threshold)))
    if not eligible:
        return float(np.nextafter(np.max(score), np.inf))
    return max(eligible)[2]


def summarize(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n_eyes": int(len(y)), "positive_eyes": int(y.sum()), "negative_eyes": int(len(y) - y.sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
        "auroc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
        "auprc": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else None,
        "brier_score": float(brier_score_loss(y, score)),
        "threshold": float(threshold),
    }


def cluster_bootstrap(eyes: list[dict], threshold: float, replicates: int, seed: int) -> dict:
    by_patient: dict[str, list[dict]] = defaultdict(list)
    for row in eyes:
        by_patient[row["patient_id"]].append(row)
    patient_class = {}
    for patient, rows in by_patient.items():
        labels = {r["target"] for r in rows}
        if len(labels) != 1:
            raise ValueError("A patient has conflicting target/control labels")
        patient_class[patient] = labels.pop()
    pos = [p for p, label in patient_class.items() if label == 1]
    neg = [p for p, label in patient_class.items() if label == 0]
    rng = np.random.default_rng(seed)
    values = defaultdict(list)
    for _ in range(replicates):
        sample_patients = list(rng.choice(pos, len(pos), replace=True)) + list(rng.choice(neg, len(neg), replace=True))
        sample_rows = [r for patient in sample_patients for r in by_patient[patient]]
        y = np.asarray([r["target"] for r in sample_rows], dtype=int)
        score = np.asarray([r["score"] for r in sample_rows], dtype=float)
        metrics = summarize(y, score, threshold)
        for key in ("accuracy", "sensitivity", "specificity", "auroc", "auprc", "brier_score"):
            if metrics[key] is not None:
                values[key].append(metrics[key])
    return {
        "unit": "patient-cluster stratified bootstrap; all eyes from a sampled patient travel together",
        "replicates_requested": replicates,
        "interval": {key: [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]
                     for key, vals in values.items() if vals},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True, help="Private CSV with unique image_id,score columns")
    parser.add_argument("--disease", required=True)
    parser.add_argument("--target-specificity", type=float, default=0.90)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.target_specificity < 1:
        raise ValueError("target-specificity must be between 0 and 1")
    manifest, score_rows = load_csv(args.manifest), load_csv(args.scores)
    required_m = {"image_id", "patient_id", "eye_id", "target", "split"}
    if not manifest or not required_m.issubset(manifest[0]):
        raise ValueError(f"Manifest must contain {sorted(required_m)}")
    if not score_rows or not {"image_id", "score"}.issubset(score_rows[0]):
        raise ValueError("Scores must contain image_id and score")
    if len({r["image_id"] for r in manifest}) != len(manifest):
        raise ValueError("Duplicate image IDs in task manifest")
    if len({r["image_id"] for r in score_rows}) != len(score_rows):
        raise ValueError("Duplicate image IDs in score file")
    manifest_by_id = {r["image_id"]: r for r in manifest}
    score_by_id = {r["image_id"]: float(r["score"]) for r in score_rows}
    if set(score_by_id) != set(manifest_by_id):
        raise ValueError("Score file IDs do not exactly match task manifest")
    if any(not np.isfinite(v) or v < 0 or v > 1 for v in score_by_id.values()):
        raise ValueError("Scores must be finite probabilities in [0,1]")
    for row in manifest:
        if row["split"] not in {"train", "validation", "test"} or row["target"] not in {"0", "1"}:
            raise ValueError("Unexpected split or target label")
    # Eye is the classification unit; multiple images from the same eye are max-pooled.
    eyes: dict[tuple[str, str, str], dict] = {}
    for row in manifest:
        key = (row["patient_id"], row["eye_id"], row["split"])
        rec = eyes.setdefault(key, {"patient_id": row["patient_id"], "eye_id": row["eye_id"],
                                    "split": row["split"], "target": int(row["target"]), "score": -np.inf})
        if rec["target"] != int(row["target"]):
            raise ValueError("Conflicting target labels within eye")
        rec["score"] = max(rec["score"], score_by_id[row["image_id"]])
    eye_rows = list(eyes.values())
    by_patient_split: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in eye_rows:
        by_patient_split[(row["patient_id"], row["split"])].add(row["target"])
    assignments: dict[str, set[str]] = defaultdict(set)
    for patient, split in by_patient_split:
        assignments[patient].add(split)
    if any(len(s) != 1 for s in assignments.values()):
        raise ValueError("Patient leakage across train/validation/test detected")
    valid = [r for r in eye_rows if r["split"] == "validation"]
    test = [r for r in eye_rows if r["split"] == "test"]
    for split_name, rows in (("validation", valid), ("test", test)):
        if not rows or len({r["target"] for r in rows}) != 2:
            raise ValueError(f"Both classes are required in {split_name}")
    yv = np.asarray([r["target"] for r in valid], dtype=int)
    sv = np.asarray([r["score"] for r in valid], dtype=float)
    threshold = choose_threshold(yv, sv, args.target_specificity)
    yt = np.asarray([r["target"] for r in test], dtype=int)
    st = np.asarray([r["score"] for r in test], dtype=float)
    summary = {
        "status": "INTERNAL_PATIENT_GROUPED_DEVELOPMENT_EVALUATION",
        "disease": args.disease,
        "task": "disease vs Healthy; other diagnosis classes excluded",
        "analysis_unit": "eye; multiple images within an eye aggregated by maximum score",
        "threshold_policy": f"validation-selected maximum sensitivity subject to specificity >= {args.target_specificity:.3f}",
        "validation": summarize(yv, sv, threshold),
        "test": summarize(yt, st, threshold),
        "test_patient_cluster_ci": cluster_bootstrap(test, threshold, args.bootstrap_replicates, args.seed),
        "patient_counts": {
            split: len({r["patient_id"] for r in eye_rows if r["split"] == split})
            for split in ("train", "validation", "test")
        },
        "caveat": "Single-source retrospective development estimate. This is not independent external validation or clinical validation. The Zhejiang DR cohort has prior project model outputs.",
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "score_file_sha256": hashlib.sha256(args.scores.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
