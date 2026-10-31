"""Evaluate a frozen MFIDDR checkpoint with validation-set patient thresholds."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from golden_retina.train_mixed_view_bag_autonomy import (
    EyeBags,
    MixedViewModel,
    aggregate_predictions_by_patient,
    predictions,
)
from golden_retina.train_quality_gated_autonomy import (
    evaluate_policy,
    read_rows,
    select_accuracy_policy,
    select_accuracy_floor_sensitivity_policy,
    select_policy,
)


def _metrics(scores: dict[str, np.ndarray], policy: dict[str, Any]) -> dict[str, Any]:
    base = {key: scores[key] for key in ("grade", "risk", "quality", "quality_known", "ungradable")}
    metrics = evaluate_policy(base, policy)
    y = (scores["grade"].astype(int) >= 2) & (scores["grade"].astype(int) <= 4)
    predicted = (scores["risk"].astype(float) >= float(policy["risk_threshold"])) | (
        scores["quality"].astype(float) >= float(policy["quality_threshold"])
    )
    positive = y.astype(bool)
    negative = ~positive
    tp = int((predicted & positive).sum())
    tn = int((~predicted & negative).sum())
    fp = int((predicted & negative).sum())
    fn = int((~predicted & positive).sum())
    sensitivity = tp / int(positive.sum()) if positive.any() else float("nan")
    specificity = tn / int(negative.sum()) if negative.any() else float("nan")
    metrics.update({
        "accuracy": float((predicted == positive).mean()),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive_referrals": fp,
        "false_negative_clearances": fn,
        "n_referable_dr": int(positive.sum()),
        "n_nonreferable": int(negative.sum()),
        "n_patients": int(len(y)),
    })
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        metrics["auroc"] = float(roc_auc_score(y, scores["risk"].astype(float)))
        metrics["average_precision"] = float(average_precision_score(y, scores["risk"].astype(float)))
    except ValueError:
        metrics["auroc"] = None
        metrics["average_precision"] = None
    return metrics


def _patient_bootstrap(
    scores: dict[str, np.ndarray], policy: dict[str, Any], replicates: int, seed: int
) -> dict[str, dict[str, float]]:
    labels = ((scores["grade"].astype(int) >= 2) & (scores["grade"].astype(int) <= 4)).astype(int)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if not len(positive) or not len(negative):
        return {}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        key: [] for key in ("sensitivity", "specificity", "accuracy", "balanced_accuracy", "auroc")
    }
    for _ in range(replicates):
        sample = np.concatenate((
            rng.choice(positive, size=len(positive), replace=True),
            rng.choice(negative, size=len(negative), replace=True),
        ))
        resampled = {key: value[sample] for key, value in scores.items()}
        result = _metrics(resampled, policy)
        for key in values:
            value = result.get(key)
            if value is not None and np.isfinite(value):
                values[key].append(float(value))
    return {
        key: {
            "lower_95": float(np.quantile(items, 0.025)),
            "upper_95": float(np.quantile(items, 0.975)),
        }
        for key, items in values.items() if items
    }


def _upper_miss_bound(misses: int, positives: int, confidence: float = 0.95) -> float | None:
    if positives <= 0:
        return None
    if misses == 0:
        return float(1.0 - (1.0 - confidence) ** (1.0 / positives))
    from scipy.stats import beta

    return float(beta.ppf(confidence, misses + 1, positives - misses))


def _write_patient_predictions(path: Path, scores: dict[str, np.ndarray], policy: dict[str, Any]) -> None:
    risk_threshold = float(policy["risk_threshold"])
    quality_threshold = float(policy["quality_threshold"])
    referred = (scores["risk"].astype(float) >= risk_threshold) | (
        scores["quality"].astype(float) >= quality_threshold
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("patient_id", "max_eye_grade", "referable", "risk", "action"))
        for i, patient in enumerate(scores["patient_id"]):
            writer.writerow((
                patient,
                int(scores["grade"][i]),
                int(scores["referable"][i]),
                float(scores["risk"][i]),
                "REFER" if referred[i] else "CLEAR",
            ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--budgets", nargs="+", type=int, default=[1, 2, 3, 4])
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = checkpoint["args"]
    max_views = int(saved["max_views"])
    if any(budget < 1 or budget > max_views for budget in args.budgets):
        raise ValueError(f"Budgets must lie within 1..{max_views}")

    model = MixedViewModel(
        checkpoint["model_name"],
        pretrained=False,
        image_size=int(saved["image_size"]),
        preserve_pretrained_risk=bool(saved["preserve_pretrained_risk"]),
        risk_pooling=str(saved["risk_pooling"]),
        risk_pool_temperature=float(saved["risk_pool_temperature"]),
        max_views=max_views,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()

    root = Path.cwd()
    splits = read_rows(Path(args.manifest))
    allowed_fields = set(saved["fields"] or []) if saved.get("fields") else None
    if allowed_fields:
        for split_name, rows in splits.items():
            splits[split_name] = [row for row in rows if str(row.get("field", "")) in allowed_fields]
    valid_data = EyeBags(
        splits["valid"], model.eval_transform, root, pad_to_views=max_views
    )
    test_data = EyeBags(
        splits["test"], model.eval_transform, root, pad_to_views=max_views
    )
    pin = device.type == "cuda"
    valid_loader = DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    budget_results: dict[str, Any] = {}
    for budget in args.budgets:
        valid_eye = predictions(model, valid_loader, device, view_budget=budget)
        test_eye = predictions(model, test_loader, device, view_budget=budget)
        valid_patient = aggregate_predictions_by_patient(valid_eye)
        test_patient = aggregate_predictions_by_patient(test_eye)
        validation_input = {
            "grade": valid_patient["grade"],
            "risk": valid_patient["risk"],
            "quality": valid_patient["quality"],
            "quality_known": valid_patient["quality_known"],
            "ungradable": valid_patient["ungradable"],
        }
        accuracy_policy = select_accuracy_policy(validation_input)
        accuracy_floor_policy = select_accuracy_floor_sensitivity_policy(validation_input, 0.90)
        safety_policy = select_policy(validation_input)
        test_metrics = {}
        validation_metrics = {}
        for name, policy in (
            ("accuracy", accuracy_policy),
            ("sensitivity_at_90pct_accuracy", accuracy_floor_policy),
            ("zero_miss_specificity", safety_policy),
        ):
            metrics = _metrics(test_patient, policy)
            misses = int(metrics["missed_referable_dr"])
            positives = int(metrics["n_referable_dr"])
            metrics["one_sided_95_upper_missed_risk"] = _upper_miss_bound(misses, positives)
            metrics["bootstrap_95_ci"] = _patient_bootstrap(
                test_patient, policy, args.bootstrap_replicates, args.seed + budget + (0 if name == "accuracy" else 1000)
            )
            test_metrics[name] = metrics
            validation_metrics[name] = _metrics(valid_patient, policy)
        _write_patient_predictions(
            output_dir / f"patient_predictions_test_budget_{budget}_accuracy.csv", test_patient, accuracy_policy
        )
        _write_patient_predictions(
            output_dir / f"patient_predictions_test_budget_{budget}_sensitivity_at_90pct_accuracy.csv",
            test_patient,
            accuracy_floor_policy,
        )
        _write_patient_predictions(
            output_dir / f"patient_predictions_test_budget_{budget}_zero_miss_specificity.csv", test_patient, safety_policy
        )
        budget_results[str(budget)] = {
            "view_budget": budget,
            "fields_used": ["F1", "F2", "F3", "F4"][:budget],
            "primary_objective": "binary_accuracy",
            "accuracy_policy_selected_on_validation_patients": accuracy_policy,
            "sensitivity_at_90pct_accuracy_policy_selected_on_validation_patients": accuracy_floor_policy,
            "zero_miss_specificity_policy_selected_on_validation_patients": safety_policy,
            "validation_patient_metrics": validation_metrics,
            "test_patient_metrics": test_metrics,
            "patient_disjoint_split": True,
        }

    summary = {
        "status": "FROZEN_CHECKPOINT_PATIENT_LEVEL_EVALUATION",
        "primary_target": "maximize patient-level binary referable-DR accuracy on validation; report held-out official test accuracy",
        "claim_boundary": "Retrospective MFIDDR benchmark evaluation; not clinical validation or a safety certificate.",
        "checkpoint": str(checkpoint_path),
        "manifest": args.manifest,
        "model_name": checkpoint["model_name"],
        "training_field_filter": saved.get("fields"),
        "test_was_used_for_model_selection": False,
        "unit_of_analysis": "patient; either-eye referable disease defines a positive patient",
        "bootstrap": {
            "unit": "patient",
            "replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "interval": "stratified patient percentile bootstrap, conditional on frozen model and validation-selected policy",
        },
        "budgets": budget_results,
    }
    summary_path = output_dir / "aggregate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
