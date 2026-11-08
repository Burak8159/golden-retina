"""One-shot exploratory DR-risk ranking on the reserved Zhejiang UWF cohort.

This deliberately reports patient-level ranking only. Zhejiang labels indicate
source-defined DR diagnosis versus healthy controls, not referable-DR grade, and
the UWF camera domain differs from training. It is not a clinical validation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from golden_retina.train_mixed_view_bag_autonomy import MixedViewModel


class SingleImageRows(Dataset):
    def __init__(self, rows: list[dict[str, str]], transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image_path"]) as image:
            image = self.transform(image.convert("RGB"))
        # Model expects two padded slots; the mask ensures only the observed
        # UWF image contributes. This is a single-view domain-transfer probe.
        return torch.stack((image, image)), torch.tensor((True, False)), index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="CSV: image_path,patient_id,diagnosis")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.manifest, newline="", encoding="utf-8-sig")))
    required = {"image_path", "patient_id", "diagnosis", "quality_overall_quality"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must contain {sorted(required)}")
    for row in rows:
        if row["diagnosis"] not in {"DR", "Healthy"}:
            continue
        if not Path(row["image_path"]).is_file():
            raise FileNotFoundError(row["image_path"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint["args"]
    model = MixedViewModel(
        model_name=checkpoint["model_name"], pretrained=False,
        image_size=int(ckpt_args["image_size"]),
        preserve_pretrained_risk=bool(ckpt_args.get("preserve_pretrained_risk", False)),
        risk_pooling=ckpt_args.get("risk_pooling", "fused"),
        risk_pool_temperature=float(ckpt_args.get("risk_pool_temperature", 0.25)),
        lesion_presence_classes=0, max_views=int(ckpt_args.get("max_views", 2)),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    dataset = SingleImageRows(rows, model.eval_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                        pin_memory=device.type == "cuda")
    risk = np.empty(len(rows), dtype=np.float64)
    quality = np.empty(len(rows), dtype=np.float64)
    grades = np.empty((len(rows), 5), dtype=np.float64)
    with torch.inference_mode():
        for images, mask, indices in loader:
            images, mask = images.to(device), mask.to(device)
            grade_logits, logits, quality_logits = model(images, mask)
            idx = indices.numpy()
            risk[idx] = torch.sigmoid(logits.float()).cpu().numpy()
            quality[idx] = torch.sigmoid(quality_logits.float()).cpu().numpy()
            grades[idx] = torch.softmax(grade_logits.float(), dim=-1).cpu().numpy()
    for i, row in enumerate(rows):
        row["risk_score"] = float(risk[i])
        row["quality_score"] = float(quality[i])
        row["grade_probabilities"] = json.dumps(grades[i].tolist())

    # Disease-risk summaries collapse multiple images/eyes to one patient using
    # max score. Quality assessment remains image-level.
    patient_groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        patient_groups.setdefault(row["patient_id"], []).append(i)
    patients = []
    for patient_id, indices in patient_groups.items():
        labels = {rows[i]["diagnosis"] for i in indices}
        if len(labels) != 1:
            raise ValueError(f"Patient has conflicting DR/Healthy labels: {patient_id}")
        patients.append({"patient_id": patient_id, "diagnosis": labels.pop(),
                         "risk_score": max(risk[i] for i in indices),
                         "quality_score": max(quality[i] for i in indices),
                         "n_images": len(indices)})

    from sklearn.metrics import average_precision_score, roc_auc_score
    categories = sorted({p["diagnosis"] for p in patients})
    healthy = [p for p in patients if p["diagnosis"] == "Healthy"]
    observed = {"n_patients_total": len(patients), "n_images_total": len(rows),
                "patients_by_diagnosis": {d: sum(p["diagnosis"] == d for p in patients) for d in categories},
                "disease_vs_healthy_patient_ranking": {}}
    for category in categories:
        if category == "Healthy":
            continue
        cohort = [p for p in patients if p["diagnosis"] in {category, "Healthy"}]
        y_cat = np.asarray([p["diagnosis"] == category for p in cohort], dtype=int)
        score_cat = np.asarray([p["risk_score"] for p in cohort], dtype=float)
        observed["disease_vs_healthy_patient_ranking"][category] = {
            "disease_patients": int(y_cat.sum()), "healthy_patients": int(len(healthy)),
            "auroc": float(roc_auc_score(y_cat, score_cat)),
            "auprc": float(average_precision_score(y_cat, score_cat)),
        }
    # Quality head was trained against DDR grade-5 ungradable labels. Zhejiang
    # marks overall quality 0=poor, 1=good; evaluate poor-quality ranking at
    # image level and apply the previously frozen validation threshold once.
    q_y = np.asarray([1-int(r["quality_overall_quality"]) for r in rows], dtype=int)
    q_score = quality
    q_thresh = float(checkpoint["best_epoch"]["validation_policy"]["quality_threshold"])
    q_pred = q_score >= q_thresh
    observed["overall_quality_poor_vs_good_image_level"] = {
        "poor_images": int(q_y.sum()), "good_images": int((1-q_y).sum()),
        "auroc": float(roc_auc_score(q_y, q_score)),
        "auprc": float(average_precision_score(q_y, q_score)),
        "fixed_validation_threshold": q_thresh,
        "sensitivity_at_fixed_threshold": float((q_pred & (q_y == 1)).sum()/max(1, int(q_y.sum()))),
        "specificity_at_fixed_threshold": float((~q_pred & (q_y == 0)).sum()/max(1, int((1-q_y).sum()))),
        "threshold_semantics_caveat": "Frozen threshold was selected for DDR gradability, not Zhejiang overall quality; these values are exploratory only.",
    }

    dr_healthy = [p for p in patients if p["diagnosis"] in {"DR", "Healthy"}]
    y = np.asarray([p["diagnosis"] == "DR" for p in dr_healthy], dtype=int)
    scores = np.asarray([p["risk_score"] for p in dr_healthy], dtype=float)
    observed["primary_dr_vs_healthy"] = {
        "n_patients": int(len(dr_healthy)), "dr_patients": int(y.sum()),
        "healthy_patients": int((1-y).sum()),
        "auroc": float(roc_auc_score(y, scores)),
        "auprc": float(average_precision_score(y, scores)),
    }
    rng = np.random.default_rng(20261108)
    boot = []
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for _ in range(2000):
        sample = np.r_[rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        boot.append((roc_auc_score(y[sample], scores[sample]), average_precision_score(y[sample], scores[sample])))
    boot = np.asarray(boot)
    observed["primary_dr_vs_healthy"]["patient_bootstrap_95_percentile_ci"] = {
        "replicates": 2000,
        "auroc": np.quantile(boot[:, 0], [0.025, 0.975]).tolist(),
        "auprc": np.quantile(boot[:, 1], [0.025, 0.975]).tolist(),
        "unit": "patient, stratified by source-defined DR/Healthy class",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "EXPLORATORY_SOURCE_TRANSFER_ONLY_NOT_REFERABLE_DR_VALIDATION",
        "checkpoint_sha256": __import__("hashlib").sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "manifest_sha256": __import__("hashlib").sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "label_semantics": "Zhejiang source-defined DR diagnosis versus healthy control; not referable-DR grade.",
        "domain_limit": "Single-center Optos ultra-widefield images differ from the training image sources.",
        "patient_aggregation": "Maximum image risk score within patient.",
        "metrics": observed,
        "decision_threshold_metrics": "No DR-vs-healthy binary decision metrics: the frozen referable-DR threshold is not semantically valid for DR-diagnosis-versus-healthy labels. The quality threshold result is reported only as exploratory because it was chosen for DDR gradability.",
        "clinical_claim": "None; retrospective exploratory source-transfer rankings only. The risk head is a referable-DR score, not heads for AMD, RVO, PM, uveitis, or retinal detachment.",
        "row_predictions_path": str(out),
    }
    Path(str(out)+".summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
