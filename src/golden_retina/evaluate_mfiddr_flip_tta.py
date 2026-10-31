"""Evaluate a horizontal-flip TTA policy with patient-level MFIDDR splits.

The threshold is selected on validation patients only. The script also computes
an explicitly exploratory test diagnostic; it is not independent confirmation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from golden_retina.train_mixed_view_bag_autonomy import (
    EyeBags,
    MixedViewModel,
    aggregate_predictions_by_patient,
    grade_boundary_margin,
    max_grade_margin,
    ordinal_referable_score,
)
from golden_retina.train_quality_gated_autonomy import (
    evaluate_policy,
    read_rows,
    select_accuracy_floor_sensitivity_policy,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = checkpoint["args"]
    max_views = int(saved["max_views"])
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

    splits = read_rows(Path(args.manifest))
    root = Path.cwd()
    fields = set(saved["fields"] or []) if saved.get("fields") else None
    if fields:
        for split_name in splits:
            splits[split_name] = [
                row for row in splits[split_name]
                if str(row.get("field", "")) in fields
            ]

    @torch.inference_mode()
    def run(split_name: str) -> dict[str, np.ndarray]:
        dataset = EyeBags(
            splits[split_name], model.eval_transform, root, pad_to_views=max_views
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
        )
        keys = (
            "grade", "referable", "quality_known", "ungradable", "risk",
            "ordinal_risk", "max_grade_margin", "grade_boundary_margin",
            "quality", "eye_id", "patient_id", "source_dataset", "image_ids",
        )
        columns = {key: [] for key in keys}
        for batch in loader:
            images, mask, grade, ref, known, ungradable, eye, patient, source, ids, _, _ = batch
            images = images.to(device)
            mask = mask.to(device)
            grade_logits, risk_logits, quality_logits = model(images, mask)
            flip_grade_logits, flip_risk_logits, flip_quality_logits = model(
                images.flip(-1), mask
            )

            # Sensitivity-oriented flip TTA: take the higher risk and quality
            # score across the original and horizontally flipped inputs.
            risk = torch.maximum(
                torch.sigmoid(risk_logits.float()),
                torch.sigmoid(flip_risk_logits.float()),
            )
            ordinal_risk = torch.maximum(
                ordinal_referable_score(grade_logits),
                ordinal_referable_score(flip_grade_logits),
            )
            mean_grade_logits = (grade_logits.float() + flip_grade_logits.float()) / 2
            quality = torch.maximum(
                torch.sigmoid(quality_logits.float()),
                torch.sigmoid(flip_quality_logits.float()),
            )
            values = {
                "grade": grade.numpy(),
                "referable": ref.numpy(),
                "quality_known": known.numpy(),
                "ungradable": ungradable.numpy(),
                "risk": risk.cpu().numpy(),
                "ordinal_risk": ordinal_risk.cpu().numpy(),
                "max_grade_margin": max_grade_margin(mean_grade_logits).cpu().numpy(),
                "grade_boundary_margin": grade_boundary_margin(mean_grade_logits).cpu().numpy(),
                "quality": quality.cpu().numpy(),
                "eye_id": eye,
                "patient_id": patient,
                "source_dataset": source,
                "image_ids": ids,
            }
            for key, value in values.items():
                columns[key].extend(
                    value.tolist() if isinstance(value, np.ndarray) else list(value)
                )
        return aggregate_predictions_by_patient(
            {key: np.asarray(value) for key, value in columns.items()}
        )

    def metrics(scores: dict[str, np.ndarray], policy: dict) -> dict:
        policy_input = {
            key: scores[key]
            for key in ("grade", "risk", "quality", "quality_known", "ungradable")
        }
        result = evaluate_policy(policy_input, policy)
        positive = (scores["grade"].astype(int) >= 2) & (scores["grade"].astype(int) <= 4)
        referred = (scores["risk"] >= float(policy["risk_threshold"])) | (
            scores["quality"] >= float(policy["quality_threshold"])
        )
        tp = int((referred & positive).sum())
        fn = int((~referred & positive).sum())
        tn = int((~referred & ~positive).sum())
        fp = int((referred & ~positive).sum())
        result.update({
            "n": int(len(positive)),
            "accuracy": float((referred == positive).mean()),
            "misses": fn,
            "sensitivity": tp / max(1, int(positive.sum())),
            "specificity": tn / max(1, int((~positive).sum())),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "positive": int(positive.sum()),
        })
        return result

    validation = run("valid")
    test = run("test")
    validation_input = {
        key: validation[key]
        for key in ("grade", "risk", "quality", "quality_known", "ungradable")
    }
    policy = select_accuracy_floor_sensitivity_policy(validation_input, 0.90)
    output = {
        "method": "Horizontal-flip TTA using the maximum risk across original/flipped inputs; threshold selected on patient-level validation.",
        "validation": metrics(validation, policy),
        "already_exposed_test_diagnostic": metrics(test, policy),
        "policy": policy,
        "n_validation": int(len(validation["grade"])),
        "n_test": int(len(test["grade"])),
        "test_independent_confirmation": False,
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
