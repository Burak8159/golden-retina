"""Patient-clustered validation comparison for paired-view and macula-only arms."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


SOURCES = ("drtid_paired", "deepdrid")


def patient_rows(path: Path, source: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("split") != "valid" or row.get("source_dataset") != source:
                continue
            patient = str(row["patient_id"])
            label = int(row["referable"])
            score = float(row["risk"])
            item = grouped.setdefault(patient, {"label": 0.0, "score": float("-inf")})
            # The analysis plan defines a patient positive when either eye is
            # referable. Eye-level labels can legitimately differ by laterality.
            item["label"] = float(max(item["label"], label))
            item["score"] = max(item["score"], score)
    if not grouped:
        raise ValueError(f"No validation patients for source {source} in {path}")
    return grouped


def score(labels: np.ndarray, values: np.ndarray) -> dict[str, float]:
    if len(np.unique(labels)) != 2:
        raise ValueError("A bootstrap sample lacks a positive or negative patient")
    return {
        "auroc": float(roc_auc_score(labels, values)),
        "auprc": float(average_precision_score(labels, values)),
    }


def compare(pair_csv: Path, macula_csv: Path, replicates: int, seed: int) -> dict:
    source_data = {}
    per_source = {}
    for source in SOURCES:
        pair, macula = patient_rows(pair_csv, source), patient_rows(macula_csv, source)
        if pair.keys() != macula.keys():
            raise ValueError(f"Patient sets differ between arms for {source}")
        ids = sorted(pair)
        labels = np.asarray([pair[x]["label"] for x in ids], dtype=np.int8)
        if not np.array_equal(labels, np.asarray([macula[x]["label"] for x in ids], dtype=np.int8)):
            raise ValueError(f"Labels differ between arms for {source}")
        pair_scores = np.asarray([pair[x]["score"] for x in ids], dtype=np.float64)
        macula_scores = np.asarray([macula[x]["score"] for x in ids], dtype=np.float64)
        source_data[source] = (labels, pair_scores, macula_scores)
        p, m = score(labels, pair_scores), score(labels, macula_scores)
        per_source[source] = {
            "n_patients": len(ids),
            "positive_patients": int(labels.sum()),
            "negative_patients": int(len(labels) - labels.sum()),
            "paired_view": p,
            "macula_only": m,
            "difference_paired_minus_macula": {
                key: p[key] - m[key] for key in ("auroc", "auprc")
            },
        }

    rng = np.random.default_rng(seed)
    boot_pair, boot_macula, boot_delta = [], [], []
    for _ in range(replicates):
        p_auc, m_auc = [], []
        for source in SOURCES:
            labels, p_scores, m_scores = source_data[source]
            indices = rng.integers(0, len(labels), size=len(labels))
            if len(np.unique(labels[indices])) != 2:
                continue
            p_auc.append(roc_auc_score(labels[indices], p_scores[indices]))
            m_auc.append(roc_auc_score(labels[indices], m_scores[indices]))
        if len(p_auc) == len(SOURCES):
            boot_pair.append(float(np.mean(p_auc)))
            boot_macula.append(float(np.mean(m_auc)))
            boot_delta.append(float(np.mean(np.asarray(p_auc) - np.asarray(m_auc))))
    if len(boot_delta) < int(replicates * 0.9):
        raise RuntimeError(f"Too many invalid bootstrap samples: {len(boot_delta)}/{replicates}")

    def interval(values: list[float]) -> list[float]:
        return [float(x) for x in np.quantile(values, [0.025, 0.975])]

    return {
        "status": "INTERNAL_VALIDATION_COMPARISON_NOT_EXTERNAL_VALIDATION",
        "primary_endpoint": "equal-weight macro patient-level AUROC across DRTiD and DeepDRiD validation cohorts",
        "paired_view_csv": str(pair_csv),
        "macula_only_csv": str(macula_csv),
        "bootstrap": {"unit": "patient, resampled within source", "replicates": replicates, "seed": seed, "valid_replicates": len(boot_delta)},
        "macro_patient_auroc": {
            "paired_view": float(np.mean([v["paired_view"]["auroc"] for v in per_source.values()])),
            "macula_only": float(np.mean([v["macula_only"]["auroc"] for v in per_source.values()])),
            "paired_difference": float(np.mean([v["difference_paired_minus_macula"]["auroc"] for v in per_source.values()])),
            "paired_view_ci95": interval(boot_pair),
            "macula_only_ci95": interval(boot_macula),
            "paired_difference_ci95": interval(boot_delta),
        },
        "per_source": per_source,
        "limitations": [
            "Validation data were used for checkpoint and operating-point selection; the bootstrap intervals are post-selection descriptive intervals, not confirmatory inferential guarantees.",
            "The selected checkpoints were chosen using validation data spanning all included training sources, including DDR; the primary comparison excludes DDR but is therefore still selection-exposed.",
            "The comparison does not establish clinical or independent external validation.",
            "DDR is omitted from the primary endpoint because it has single-image cases and patient identity is unverified.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-view", type=Path, required=True)
    parser.add_argument("--macula-only", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args()
    result = compare(args.paired_view, args.macula_only, args.replicates, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
