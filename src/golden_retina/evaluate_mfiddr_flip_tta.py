"""Evaluate a horizontal-flip TTA policy with patient-level MFIDDR splits.

The threshold is selected on validation patients only. The script also computes
an explicitly exploratory test diagnostic; it is not independent confirmation.
"""
from __future__ import annotations

import argparse
import csv
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
    parser.add_argument(
        "--orientation-tta",
        choices=("horizontal", "dihedral4"),
        default="horizontal",
        help="Use horizontal flip or identity plus horizontal/vertical/180-degree flips.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Also compute the already-exposed test diagnostic after selecting on validation.",
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=0,
        help="Optional diagnostic cap; nonzero runs are not full evaluation results.",
    )
    parser.add_argument(
        "--validation-predictions-out",
        type=Path,
        default=None,
        help="Optional CSV path for full patient-level validation scores and labels.",
    )
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
        for batch_index, batch in enumerate(loader):
            if args.limit_batches and batch_index >= args.limit_batches:
                break
            images, mask, grade, ref, known, ungradable, eye, patient, source, ids, _, _ = batch
            images = images.to(device)
            mask = mask.to(device)
            variants = [images, images.flip(-1)]
            if args.orientation_tta == "dihedral4":
                variants.extend((images.flip(-2), images.flip((-1, -2))))
            outputs = [model(variant, mask) for variant in variants]
            grade_logits = torch.stack([item[0].float() for item in outputs])
            risk_logits = torch.stack([item[1].float() for item in outputs])
            quality_logits = torch.stack([item[2].float() for item in outputs])

            # Sensitivity-oriented TTA: take the maximum disease/quality risk
            # across deterministic orientation variants.
            risk = torch.sigmoid(risk_logits).amax(dim=0)
            ordinal_risk = torch.stack(
                [ordinal_referable_score(item[0]) for item in outputs]
            ).amax(dim=0)
            mean_grade_logits = grade_logits.mean(dim=0)
            quality = torch.sigmoid(quality_logits).amax(dim=0)
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
    test = run("test") if args.evaluate_test else None
    if args.validation_predictions_out is not None:
        args.validation_predictions_out.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(validation)
        with args.validation_predictions_out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for index in range(len(validation["grade"])):
                writer.writerow({
                    key: (
                        validation[key][index].item()
                        if isinstance(validation[key][index], np.generic)
                        else validation[key][index]
                    )
                    for key in keys
                })
    validation_input = {
        key: validation[key]
        for key in ("grade", "risk", "quality", "quality_known", "ungradable")
    }
    policy = select_accuracy_floor_sensitivity_policy(validation_input, 0.90)
    output = {
        "method": f"{args.orientation_tta} orientation TTA using maximum risk across variants; threshold selected on patient-level validation.",
        "validation": metrics(validation, policy),
        "already_exposed_test_diagnostic": metrics(test, policy) if test is not None else None,
        "policy": policy,
        "n_validation": int(len(validation["grade"])),
        "n_test": int(len(test["grade"])) if test is not None else 0,
        "test_independent_confirmation": False,
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
