#!/usr/bin/env python3
"""Patient-cluster bootstrap audit for the scoped ConvNeXt baseline report.

All inputs are local, row-level development artifacts. Outputs are exploratory
and must not be represented as independent external validation.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def auc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = sorted(range(len(scores)), key=scores.__getitem__)
    rank_sum = 0.0
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2
        rank_sum += rank * sum(labels[order[k]] for k in range(i, j))
        i = j
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def binary_metrics(rows: list[dict], label_key: str, score_key: str) -> dict:
    y = [int(float(r[label_key])) for r in rows]
    s = [float(r[score_key]) for r in rows]
    p = [v >= 0.5 for v in s]
    tp = sum(a == 1 and b for a, b in zip(y, p))
    tn = sum(a == 0 and not b for a, b in zip(y, p))
    fp = sum(a == 0 and b for a, b in zip(y, p))
    fn = sum(a == 1 and not b for a, b in zip(y, p))
    return {
        "accuracy": (tp + tn) / len(y),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "auroc": auc(y, s),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "positive": sum(y), "negative": len(y) - sum(y),
    }


def grade_metrics(rows: list[dict]) -> dict:
    y = [int(r["dr_grade_true"]) for r in rows]
    p = [int(r["dr_grade_pred"]) for r in rows]
    present = sorted(set(y))
    recalls = [sum(a == c and b == c for a, b in zip(y, p)) / sum(a == c for a in y)
               for c in present]
    return {"accuracy": sum(a == b for a, b in zip(y, p)) / len(y),
            "macro_recall": sum(recalls) / len(recalls), "classes_present": present}


def ci_for(rows_by_cluster: dict[str, list[dict]], metric_fn, n_boot: int,
            seed: int) -> tuple[dict, dict]:
    clusters = list(rows_by_cluster)
    observed_rows = [r for group in rows_by_cluster.values() for r in group]
    observed = metric_fn(observed_rows)
    interval_keys = {"accuracy", "sensitivity", "specificity", "auroc", "macro_recall"}
    draws: dict[str, list[float]] = {k: [] for k in interval_keys if k in observed}
    rng = random.Random(seed)
    for _ in range(n_boot):
        selected = [rng.choice(clusters) for _ in clusters]
        sample = [r for group in selected for r in rows_by_cluster[group]]
        values = metric_fn(sample)
        for key in draws:
            value = values.get(key)
            if value is not None:
                draws[key].append(float(value))
    intervals = {}
    for key, values in draws.items():
        values.sort()
        if not values:
            intervals[key] = None
            continue
        lo = values[int(0.025 * (len(values) - 1))]
        hi = values[int(0.975 * (len(values) - 1))]
        intervals[key] = [lo, hi]
    return observed, intervals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--run-summary", type=Path,
                    help="Training summary used to record the validation-selected checkpoint epoch.")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261107)
    args = ap.parse_args()
    with args.predictions.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    best_epoch = None
    if args.run_summary:
        summary = json.loads(args.run_summary.read_text(encoding="utf-8"))
        best_epoch = summary.get("best_epoch")
    result = {
        "status": "EXPOSED_INTERNAL_DEVELOPMENT_TEST_DIAGNOSTIC_NOT_EXTERNAL_VALIDATION",
        "checkpoint_selection": (f"validation loss only; best epoch {best_epoch}"
                                  if best_epoch is not None else
                                  "checkpoint-selection epoch not supplied"),
        "bootstrap_replicates": args.bootstrap,
        "seed": args.seed,
        "interval": "percentile 95% cluster bootstrap; resample patients where IDs are verified, otherwise image IDs",
        "sources": {},
    }
    for source, selected in sorted(by_source.items()):
        verified = all(r["patient_group_verified"].lower() == "true" for r in selected)
        cluster_key = "patient_id" if verified else "image_id"
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in selected:
            groups[row[cluster_key]].append(row)
        tasks = {}
        dr = [r for r in selected if int(r["dr_grade_true"]) >= 0]
        if dr:
            drgroups: dict[str, list[dict]] = defaultdict(list)
            for row in dr:
                drgroups[row[cluster_key]].append(row)
            tasks["dr_grade_0_to_4"] = dict(zip(("estimate", "ci95"), ci_for(
                drgroups, grade_metrics, args.bootstrap, args.seed + len(tasks) + 1)))
            ref_fn = lambda rs: binary_metrics(
                [{"y": int(int(r["dr_grade_true"]) >= 2), "score": float(r["dr_probability_referable_ge_2"])} for r in rs], "y", "score")
            tasks["referable_dr_grade_ge_2"] = dict(zip(("estimate", "ci95"), ci_for(
                drgroups, ref_fn, args.bootstrap, args.seed + len(tasks) + 1)))
        quality = [r for r in selected if int(r["quality_ungradable_true"]) >= 0]
        if quality:
            qgroups: dict[str, list[dict]] = defaultdict(list)
            for row in quality:
                qgroups[row[cluster_key]].append(row)
            tasks["ungradable_detection"] = dict(zip(("estimate", "ci95"), ci_for(
                qgroups, lambda rs: binary_metrics(rs, "quality_ungradable_true", "quality_probability"),
                args.bootstrap, args.seed + len(tasks) + 1)))
        glaucoma = [r for r in selected if int(r["gon_true"]) >= 0]
        if glaucoma:
            ggroups: dict[str, list[dict]] = defaultdict(list)
            for row in glaucoma:
                ggroups[row[cluster_key]].append(row)
            tasks["glaucoma_gon"] = dict(zip(("estimate", "ci95"), ci_for(
                ggroups, lambda rs: binary_metrics(rs, "gon_true", "gon_probability"),
                args.bootstrap, args.seed + len(tasks) + 1)))
        result["sources"][source] = {
            "n_images": len(selected), "n_clusters": len(groups),
            "cluster_unit": cluster_key,
            "patient_ids_verified": verified,
            "tasks": tasks,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sources": {
        k: {"n_images": v["n_images"], "n_clusters": v["n_clusters"], "cluster_unit": v["cluster_unit"]}
        for k, v in result["sources"].items()}}, indent=2))


if __name__ == "__main__":
    main()
