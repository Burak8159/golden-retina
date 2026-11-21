#!/usr/bin/env python3
"""Split-separated aggregate audit for DeepDRiD paired-view predictions.

Reads existing prediction CSVs, preserves patient clusters, separates OOF
development partitions from the historical held-out partition, and writes
aggregate metrics only (no patient/eye identifiers or row-level scores).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from scipy.stats import beta

REQUIRED = {"patient_id", "eye_id", "split", "view_set", "grade",
            "referable_dr", "referable_score", "fold_id", "prediction_scope"}
DEV_SPLITS = {"train", "calibration", "validation"}
TEST_SPLIT = "sealed_test"
FIRST_VIEW = "macula"
TWO_VIEWS = "macula+optic_disc"


def summarize(y: np.ndarray, s: np.ndarray) -> dict:
    if len(np.unique(y)) < 2:
        return {"n_eyes": int(len(y)), "auroc": None, "auprc": None,
                "brier": float(brier_score_loss(y, s)) if len(y) else None}
    return {"n_eyes": int(len(y)), "positive_eyes": int(y.sum()),
            "auroc": float(roc_auc_score(y, s)),
            "auprc": float(average_precision_score(y, s)),
            "brier": float(brier_score_loss(y, s))}


def patient_bootstrap(paired: list[dict], reps: int, seed: int) -> dict:
    by_patient = defaultdict(list)
    for row in paired:
        by_patient[row["patient"]].append(row)
    patient_label = {p: int(any(r["y"] for r in rows)) for p, rows in by_patient.items()}
    positive = [p for p, y in patient_label.items() if y]
    negative = [p for p, y in patient_label.items() if not y]
    if not positive or not negative:
        return {"status": "not_estimable_both_patient_classes_required"}
    rng = np.random.default_rng(seed)
    metrics = defaultdict(list)
    for _ in range(reps):
        ps = list(rng.choice(positive, len(positive), replace=True))
        ps += list(rng.choice(negative, len(negative), replace=True))
        rows = [r for p in ps for r in by_patient[p]]
        y = np.asarray([r["y"] for r in rows], dtype=int)
        first = np.asarray([r["first"] for r in rows], dtype=float)
        both = np.asarray([r["both"] for r in rows], dtype=float)
        oracle = np.where((y - both) ** 2 < (y - first) ** 2, both, first)
        pairs = {
            "auroc_delta_two_view_minus_first": roc_auc_score(y, both) - roc_auc_score(y, first),
            "auprc_delta_two_view_minus_first": average_precision_score(y, both) - average_precision_score(y, first),
            "brier_delta_two_view_minus_first": brier_score_loss(y, both) - brier_score_loss(y, first),
            "oracle_brier_delta_vs_first": brier_score_loss(y, oracle) - brier_score_loss(y, first),
        }
        for k, v in pairs.items():
            if np.isfinite(v):
                metrics[k].append(float(v))
    return {"method": "patient-stratified cluster bootstrap; both eyes/views kept with patient",
            "replicates": reps,
            "interval_95": {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
                            for k, v in metrics.items() if v}}


def evaluate_risk_gated_policy(calibration: list[dict], validation: list[dict],
                               reps: int, seed: int) -> dict:
    """Calibrate a zero-observed-miss acquisition rule on calibration only."""
    cal_pos = [r for r in calibration if r["y"] == 1]
    if not cal_pos:
        return {"status": "not_estimable_no_calibration_positives"}
    first_threshold = min(r["first"] for r in cal_pos)
    two_view_threshold = min(r["both"] for r in cal_pos)

    def decide(rows):
        output = []
        for r in rows:
            acquire = r["first"] >= first_threshold
            predicted_positive = (r["both"] >= two_view_threshold) if acquire else False
            output.append({**r, "acquire": acquire, "pred": predicted_positive})
        return output

    cal_decisions = decide(calibration)
    val_decisions = decide(validation)

    def summarize_policy(rows):
        positives = sum(r["y"] for r in rows)
        negatives = len(rows) - positives
        fn = sum(r["y"] == 1 and not r["pred"] for r in rows)
        fp = sum(r["y"] == 0 and r["pred"] for r in rows)
        tp = positives - fn
        tn = negatives - fp
        by_patient = defaultdict(list)
        for r in rows:
            by_patient[r["patient"]].append(r)
        positive_patients = [p for p, rs in by_patient.items() if any(x["y"] for x in rs)]
        missed_patients = sum(any(x["y"] and not x["pred"] for x in by_patient[p])
                              for p in positive_patients)
        npp = len(positive_patients)
        upper = float(beta.ppf(.95, missed_patients + 1, npp - missed_patients)) if npp > missed_patients else 1.0
        return {
            "eyes": len(rows), "patients": len(by_patient), "positive_eyes": positives,
            "negative_eyes": negatives, "true_positives": tp, "false_negatives": fn,
            "true_negatives": tn, "false_positives": fp,
            "sensitivity": tp / positives if positives else None,
            "specificity": tn / negatives if negatives else None,
            "binary_action_coverage": 1.0,
            "secondary_view_acquisition_rate": sum(r["acquire"] for r in rows) / len(rows),
            "mean_views_per_eye": 1.0 + sum(r["acquire"] for r in rows) / len(rows),
            "positive_patients": npp, "missed_positive_patients": int(missed_patients),
            "one_sided_95pct_exact_upper_bound_patient_miss_risk": upper,
        }

    def bootstrap_policy(rows):
        by_patient = defaultdict(list)
        for r in rows:
            by_patient[r["patient"]].append(r)
        strata = {p: int(any(r["y"] for r in rr)) for p, rr in by_patient.items()}
        pos = [p for p, z in strata.items() if z]
        neg = [p for p, z in strata.items() if not z]
        if not pos or not neg:
            return {"status": "not_estimable_both_patient_classes_required"}
        rng = np.random.default_rng(seed)
        vals = defaultdict(list)
        for _ in range(reps):
            sample = list(rng.choice(pos, len(pos), replace=True))
            sample += list(rng.choice(neg, len(neg), replace=True))
            rr = [r for p in sample for r in by_patient[p]]
            np_ = sum(r["y"] for r in rr)
            nn_ = len(rr) - np_
            fn_ = sum(r["y"] == 1 and not r["pred"] for r in rr)
            fp_ = sum(r["y"] == 0 and r["pred"] for r in rr)
            vals["sensitivity"].append((np_ - fn_) / np_ if np_ else 0.)
            vals["specificity"].append((nn_ - fp_) / nn_ if nn_ else 0.)
            vals["secondary_view_acquisition_rate"].append(sum(r["acquire"] for r in rr) / len(rr))
        return {"method": "patient-cluster stratified bootstrap", "replicates": reps,
                "interval_95": {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
                                for k, v in vals.items()}}

    return {
        "status": "DEVELOPMENT_ONLY_CALIBRATED_ON_CALIBRATION_SPLIT",
        "rule": "Acquire a second view unless the first-view score is below its minimum among calibration positives; after acquisition, mark positive when the two-view score is at least its minimum among calibration positives. Thresholds are frozen from calibration only.",
        "thresholds": {"first_view_clearance_upper_exclusive": float(first_threshold),
                       "two_view_positive_lower_inclusive": float(two_view_threshold)},
        "calibration": summarize_policy(cal_decisions),
        "validation": summarize_policy(val_decisions),
        "validation_patient_bootstrap": bootstrap_policy(val_decisions),
        "boundary": "Empirical zero calibration misses are not a safety guarantee. Validation is small and same-source. Policy covers only a second view in DeepDRiD; it is not a clinical stop/acquire/defer system.",
    }


def run_file(path: Path, mapping_branch: str, reps: int, seed: int) -> dict:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty prediction file")
    keys = set()
    patient_splits = defaultdict(set)
    grouped = defaultdict(dict)
    for row in rows:
        split = row["split"].strip()
        scope = row["prediction_scope"].strip().lower()
        allowed = {"crossfit", "cross_fitted", "out_of_fold", "oof"} if split in DEV_SPLITS else {"heldout", "heldout_test", "sealed_test"}
        if split not in DEV_SPLITS | {TEST_SPLIT} or scope not in allowed:
            raise ValueError(f"Unexpected split/scope pair: {split!r}/{scope!r}")
        y = int(row["referable_dr"])
        score = float(row["referable_score"])
        if y not in (0, 1) or not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid binary label or score")
        patient, eye, view = row["patient_id"], row["eye_id"], row["view_set"]
        k = (split, patient, eye, view)
        if k in keys:
            raise ValueError("Duplicate patient/eye/split/view prediction")
        keys.add(k)
        patient_splits[patient].add(split)
        if view not in {FIRST_VIEW, "optic_disc", TWO_VIEWS}:
            raise ValueError(f"Unexpected view_set: {view!r}")
        grouped[(split, patient, eye)][view] = {
            "y": y, "score": score, "fold": row["fold_id"], "scope": scope}
    if any(len(s) != 1 for s in patient_splits.values()):
        raise ValueError("Patient appears across partitions")

    partitions = {}
    for split in sorted(DEV_SPLITS | {TEST_SPLIT}):
        paired = []
        incomplete = 0
        for (row_split, patient, eye), views in grouped.items():
            if row_split != split:
                continue
            if FIRST_VIEW not in views or TWO_VIEWS not in views:
                incomplete += 1
                continue
            f, b = views[FIRST_VIEW], views[TWO_VIEWS]
            if f["y"] != b["y"]:
                raise ValueError("Inconsistent paired labels")
            paired.append({"patient": patient, "eye": eye, "y": f["y"],
                           "first": f["score"], "both": b["score"]})
        if not paired:
            partitions[split] = {"status": "no_complete_pairs", "incomplete_eyes": incomplete}
            continue
        y = np.asarray([r["y"] for r in paired], dtype=int)
        first = np.asarray([r["first"] for r in paired], dtype=float)
        both = np.asarray([r["both"] for r in paired], dtype=float)
        oracle = np.where((y - both) ** 2 < (y - first) ** 2, both, first)
        partitions[split] = {
            "scope": "cross-fitted development; not a final test" if split in DEV_SPLITS else "previously designated held-out partition; descriptive only, not used for current selection",
            "n_paired_eyes": len(paired),
            "n_patients": len({r["patient"] for r in paired}),
            "positive_eyes": int(y.sum()),
            "incomplete_eyes_excluded": incomplete,
            "macula_only": summarize(y, first),
            "macula_plus_disc": summarize(y, both),
            "label_aware_hindsight_oracle": summarize(y, oracle),
            "paired_differences": {
                "auroc_two_view_minus_first": (summarize(y, both)["auroc"] - summarize(y, first)["auroc"]),
                "auprc_two_view_minus_first": (summarize(y, both)["auprc"] - summarize(y, first)["auprc"]),
                "brier_two_view_minus_first": (summarize(y, both)["brier"] - summarize(y, first)["brier"]),
                "oracle_brier_minus_first": summarize(y, oracle)["brier"] - summarize(y, first)["brier"],
            },
            "patient_cluster_bootstrap": patient_bootstrap(paired, reps, seed),
        }
    def paired_rows(split):
        out = []
        for (row_split, patient, eye), views in grouped.items():
            if row_split != split or FIRST_VIEW not in views or TWO_VIEWS not in views:
                continue
            f, b = views[FIRST_VIEW], views[TWO_VIEWS]
            out.append({"patient": patient, "eye": eye, "y": f["y"],
                        "first": f["score"], "both": b["score"]})
        return out
    calibration, validation = paired_rows("calibration"), paired_rows("validation")
    return {"prediction_file": path.name, "prediction_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "mapping_branch": mapping_branch,
            "mapping_boundary": "Filename suffix-to-anatomical-field mapping is not officially confirmed; branch comparison is a sensitivity analysis, not an averaged estimate.",
            "row_counts_by_split": {s: sum(1 for r in rows if r["split"] == s) for s in sorted(DEV_SPLITS | {TEST_SPLIT})},
            "partitions": partitions,
            "risk_gated_second_view_policy": evaluate_risk_gated_policy(
                calibration, validation, reps, seed)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", nargs=2, required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--bootstrap-replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261121)
    args = ap.parse_args()
    branches = ["inferred_suffix_1_disc", "swapped_sensitivity"]
    out = {"status": "SPLIT_SEPARATED_PAIRED_VIEW_AGGREGATE_AUDIT",
           "dataset": "DeepDRiD", "task": "referable DR, paired macula-only vs macula+optic-disc score",
           "analysis_unit": "eye; uncertainty resampled by patient cluster",
           "bootstrap_replicates": args.bootstrap_replicates,
           "policy_boundary": "This compares score quality and a label-aware retrospective oracle; it does not learn or certify STOP/ACQUIRE/DEFER, and no rows are used to tune an operating threshold.",
           "branches": [run_file(p, b, args.bootstrap_replicates, args.seed+i) for i,(p,b) in enumerate(zip(args.predictions,branches))]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
