#!/usr/bin/env python3
"""Nested patient-grouped CV for candidate selection on development TRAIN rows.

No validation or test rows are consumed. DDR is omitted because its patient
identifiers are not verified. Row-level predictions remain on the private VM.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


CANDIDATES = {
    "FLAIR": "flair/private_embeddings.npy",
    "DINOv2": "dinov2/private_embeddings.npy",
    "RETFound-MAE": "retfound_mae/private_embeddings.npy",
    "VisionFM": "visionfm/private_embeddings.npy",
    "UrFound": "urf/private_embeddings.npy",
}
C_GRID = (0.1, 1.0, 10.0)


def grouped_auc(y, score, groups, source):
    out = {}
    for s in sorted(set(source)):
        idx = np.asarray([i for i, x in enumerate(source) if x == s], dtype=int)
        if len(np.unique(y[idx])) == 2:
            out[s] = {"n_images": int(len(idx)), "n_patients": int(len(set(groups[idx]))),
                      "auroc": float(roc_auc_score(y[idx], score[idx]))}
    return out


def nested_oof(X, y, groups, sources, seed):
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    selected_cs = []
    fold_records = []
    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y, groups), start=1):
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed + fold)
        c_scores = {}
        for c in C_GRID:
            fold_aucs = []
            for a, b in inner.split(X[train_idx], y[train_idx], groups[train_idx]):
                ti, vi = train_idx[a], train_idx[b]
                scaler = StandardScaler()
                xtr = scaler.fit_transform(X[ti])
                xva = scaler.transform(X[vi])
                model = LogisticRegression(C=c, class_weight="balanced", max_iter=3000,
                                           solver="lbfgs", random_state=seed)
                model.fit(xtr, y[ti])
                fold_aucs.append(float(roc_auc_score(y[vi], model.predict_proba(xva)[:, 1])))
            c_scores[c] = float(np.mean(fold_aucs))
        best_c = max(C_GRID, key=lambda c: c_scores[c])
        selected_cs.append(best_c)
        scaler = StandardScaler()
        xtr = scaler.fit_transform(X[train_idx])
        xte = scaler.transform(X[test_idx])
        model = LogisticRegression(C=best_c, class_weight="balanced", max_iter=3000,
                                   solver="lbfgs", random_state=seed)
        model.fit(xtr, y[train_idx])
        p = model.predict_proba(xte)[:, 1]
        oof[test_idx] = p
        fold_records.append({
            "outer_fold": fold,
            "n_train_images": int(len(train_idx)),
            "n_holdout_images": int(len(test_idx)),
            "n_holdout_patients": int(len(set(groups[test_idx]))),
            "selected_C": float(best_c),
            "inner_mean_auroc_by_C": {str(k): v for k, v in c_scores.items()},
            "holdout_auroc": float(roc_auc_score(y[test_idx], p)),
            "holdout_accuracy_at_0_5": float(accuracy_score(y[test_idx], p >= 0.5)),
        })
    assert np.isfinite(oof).all()
    return oof, selected_cs, fold_records


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    train = [r for r in rows if r["split"] == "train" and r["patient_group_verified"] == "True"
             and r["source"] in {"deepdrid", "drtid_paired", "hygd_v1_1_0"}]
    return rows, train


def run_task(X, rows, task, seed):
    if task == "referable_dr_ge2":
        idx = [i for i, r in enumerate(rows) if r["source"] in {"deepdrid", "drtid_paired"}
               and r["dr_grade"] not in {"", "-1"}]
        y = np.asarray([int(rows[i]["dr_grade"]) >= 2 for i in idx], dtype=np.int64)
    else:
        idx = [i for i, r in enumerate(rows) if r["source"] == "hygd_v1_1_0"
               and r["gon"] not in {"", "-1"}]
        y = np.asarray([int(rows[i]["gon"]) == 1 for i in idx], dtype=np.int64)
    selected = [rows[i] for i in idx]
    groups = np.asarray([r["patient_id"] for r in selected], dtype=object)
    sources = np.asarray([r["source"] for r in selected], dtype=object)
    oof, cs, folds = nested_oof(X[idx], y, groups, sources, seed)
    return {
        "n_images": int(len(idx)), "positive_images": int(y.sum()),
        "n_patients": int(len(set(groups))),
        "outer_oof_auroc": float(roc_auc_score(y, oof)),
        "outer_oof_accuracy_at_0_5": float(accuracy_score(y, oof >= 0.5)),
        "outer_oof_sensitivity_at_0_5": float(((y == 1) & (oof >= 0.5)).sum() / max(1, y.sum())),
        "outer_oof_specificity_at_0_5": float(((y == 0) & (oof < 0.5)).sum() / max(1, (y == 0).sum())),
        "source_auroc": grouped_auc(y, oof, groups, sources),
        "selected_C_by_outer_fold": [float(x) for x in cs],
        "folds": folds,
        "private_oof": {"labels": y.tolist(), "scores": oof.tolist(),
                        "patient_ids": groups.tolist(), "sources": sources.tolist()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=20261108)
    args = ap.parse_args()
    base = Path(args.results_root)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    all_rows, train_rows = load_rows(args.manifest)
    # Each cached matrix follows the complete frozen manifest order; subset to
    # the verified-patient training rows and never read validation/test targets.
    manifest_rows = all_rows
    source_positions = {id(r): i for i, r in enumerate(manifest_rows)}
    keep = np.asarray([source_positions[id(r)] for r in train_rows], dtype=int)
    model_summaries = {}
    private_rows = []
    for name, rel in CANDIDATES.items():
        X_all = np.load(base / rel, mmap_mode="r")
        if X_all.shape[0] != len(manifest_rows):
            raise RuntimeError(f"{name}: embedding rows {X_all.shape[0]} != frozen manifest rows {len(manifest_rows)}")
        X = np.asarray(X_all[keep], dtype=np.float32)
        task_results = {}
        # Each task's subset indexes the same filtered train_rows in order.
        for task in ("referable_dr_ge2", "glaucoma_hygd"):
            task_results[task] = run_task(X, train_rows, task, args.seed)
            priv = task_results[task].pop("private_oof")
            for j, (y, s, patient, source) in enumerate(zip(priv["labels"], priv["scores"], priv["patient_ids"], priv["sources"])):
                private_rows.append({"model": name, "task": task, "patient_id": patient,
                                     "source": source, "label": y, "oof_score": s})
        aucs = [task_results[t]["outer_oof_auroc"] for t in task_results]
        task_results["macro_disease_auroc"] = float(np.mean(aucs))
        model_summaries[name] = task_results
        print(json.dumps({"model": name, "macro_disease_auroc": task_results["macro_disease_auroc"],
                          "DR_AUROC": task_results["referable_dr_ge2"]["outer_oof_auroc"],
                          "glaucoma_AUROC": task_results["glaucoma_hygd"]["outer_oof_auroc"]}), flush=True)
    ranked = sorted(model_summaries, key=lambda n: model_summaries[n]["macro_disease_auroc"], reverse=True)
    summary = {
        "status": "NESTED_PATIENT_GROUPED_DEVELOPMENT_CV_TRAIN_ROWS_ONLY",
        "outer_folds": 5, "inner_folds": 3, "C_grid": list(C_GRID),
        "seed": args.seed, "n_manifest_rows_total": len(all_rows),
        "n_verified_train_rows": len(train_rows), "validation_rows_read": 0,
        "test_rows_read": 0,
        "excluded_sources": {"ddr_hf_specificity300": "patient grouping is unverified; excluded from patient-grouped model selection"},
        "candidate_ranking_by_mean_outer_oof_disease_auroc": ranked,
        "models": model_summaries,
        "interpretation": "Internal development-only model selection estimate. These nested folds are not an external test or clinical validation. Metrics are image-level; patient IDs define fold grouping.",
    }
    (out / "nested_patient_cv_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with open(out / "nested_patient_cv_oof_private.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(private_rows[0])); w.writeheader(); w.writerows(private_rows)
    print(json.dumps({"ranking": ranked, "summary": str(out / "nested_patient_cv_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
