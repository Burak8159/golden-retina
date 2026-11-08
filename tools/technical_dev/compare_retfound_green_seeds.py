#!/usr/bin/env python3
"""Compare two RETFound-Green validation runs at the patient level.

Reads private validation CSVs and writes aggregate-only metrics. The patient
unit is (source, patient_id); patient scores are the maximum image score and
labels are positive if any image is positive. This is internal validation,
not external confirmation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


def read_patients(path: Path) -> dict[tuple[str, str], tuple[int, float]]:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            truth = int(row["referable_truth"])
            if truth < 0:
                continue
            key = (row["source"], row["patient_id"])
            item = grouped.setdefault(key, {"labels": [], "scores": []})
            item["labels"].append(truth)
            item["scores"].append(float(row["referable_score"]))
    return {key: (int(max(v["labels"])), float(max(v["scores"])))
            for key, v in grouped.items()}


def metric_vector(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    pred = scores >= 0.5
    return {
        "accuracy_at_fixed_0_5": float(accuracy_score(labels, pred)),
        "sensitivity_at_fixed_0_5": float(pred[labels == 1].mean()) if np.any(labels == 1) else float("nan"),
        "specificity_at_fixed_0_5": float((~pred[labels == 0]).mean()) if np.any(labels == 0) else float("nan"),
        "auroc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan"),
        "auprc": float(average_precision_score(labels, scores)) if np.any(labels == 1) else float("nan"),
    }


def bootstrap(a: dict, b: dict, keys: list[tuple[str, str]], draws: int, seed: int):
    y = np.asarray([a[k][0] for k in keys], dtype=int)
    pa = np.asarray([a[k][1] for k in keys], dtype=float)
    pb = np.asarray([b[k][1] for k in keys], dtype=float)
    names = ("accuracy_at_fixed_0_5", "sensitivity_at_fixed_0_5",
             "specificity_at_fixed_0_5", "auroc", "auprc")
    point_a, point_b = metric_vector(y, pa), metric_vector(y, pb)
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in names}
    for _ in range(draws):
        idx = rng.integers(0, len(keys), size=len(keys))
        ma, mb = metric_vector(y[idx], pa[idx]), metric_vector(y[idx], pb[idx])
        for name in names:
            diff = mb[name] - ma[name]
            if np.isfinite(diff):
                samples[name].append(float(diff))
    return {
        "n_patients": len(keys),
        "positive_patients": int(y.sum()),
        "seed_1": point_a,
        "seed_2": point_b,
        "delta_seed2_minus_seed1": {
            name: point_b[name] - point_a[name]
            for name in names
        },
        "paired_patient_bootstrap_95ci_for_delta": {
            name: [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]
            if vals else None for name, vals in samples.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed1", type=Path, required=True)
    parser.add_argument("--seed2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    a, b = read_patients(args.seed1), read_patients(args.seed2)
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("The two runs have no common patient-source groups")
    mismatched = [k for k in common if a[k][0] != b[k][0]]
    if mismatched:
        raise ValueError(f"Label mismatch in {len(mismatched)} patient-source groups")
    out = {
        "status": "PATIENT_GROUPED_INTERNAL_VALIDATION_SEED_STABILITY",
        "cohort": "fixed exposed validation manifest; no test rows",
        "patient_aggregation": "source-patient group; patient positive if any labeled image is positive; patient score is max image score",
        "classification_threshold": 0.5,
        "common_patient_source_groups": len(common),
        "seed1_groups": len(a),
        "seed2_groups": len(b),
        "metrics": bootstrap(a, b, common, args.bootstrap_draws, 20261109),
        "warning": "Seed stability on the same exposed validation cohort is not independent validation or clinical certification.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
