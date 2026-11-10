#!/usr/bin/env python3
"""Summarize error patterns without using them for threshold/model tuning."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    rows = list(csv.DictReader(a.predictions.open(newline="", encoding="utf-8")))
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    report = {"status": "EXPOSED_INTERNAL_DEVELOPMENT_DIAGNOSTIC_ONLY_NO_TUNING",
              "by_source": {}}
    for source, selected in sorted(by_source.items()):
        dr = [r for r in selected if int(r["dr_grade_true"]) >= 0]
        grade_counts = Counter(int(r["dr_grade_true"]) for r in dr)
        grade_errors = Counter(int(r["dr_grade_true"]) for r in dr
                               if int(r["dr_grade_pred"]) != int(r["dr_grade_true"]))
        confusion = {str(i): {str(j): 0 for j in range(5)} for i in range(5)}
        for r in dr:
            confusion[str(int(r["dr_grade_true"]))][str(int(r["dr_grade_pred"]))] += 1
        positives = [r for r in dr if int(r["dr_grade_true"]) >= 2]
        false_negative = [r for r in positives
                          if float(r["dr_probability_referable_ge_2"]) < 0.5]
        false_positive = [r for r in dr if int(r["dr_grade_true"]) < 2
                          and float(r["dr_probability_referable_ge_2"]) >= 0.5]
        patient_fn = defaultdict(list)
        for r in false_negative:
            patient_fn[r["patient_id"]].append(r["image_id"])
        quality = [r for r in selected if int(r["quality_ungradable_true"]) >= 0]
        quality_counts = Counter(int(r["quality_ungradable_true"]) for r in quality)
        qfn = sum(int(r["quality_ungradable_true"]) == 1 and float(r["quality_probability"]) < 0.5
                  for r in quality)
        qfp = sum(int(r["quality_ungradable_true"]) == 0 and float(r["quality_probability"]) >= 0.5
                  for r in quality)
        report["by_source"][source] = {
            "images": len(selected),
            "patient_grouping_verified": all(r["patient_group_verified"].lower() == "true" for r in selected),
            "dr_grade_support": {str(k): grade_counts[k] for k in sorted(grade_counts)},
            "dr_grade_error_counts_by_true_grade": {str(k): grade_errors[k] for k in sorted(grade_counts)},
            "dr_grade_confusion_matrix_rows_true_columns_pred": confusion,
            "referable_dr": {
                "n_positive_images": len(positives),
                "false_negative_images_at_0_5": len(false_negative),
                "false_negative_counts_by_true_grade": dict(sorted(Counter(int(r["dr_grade_true"]) for r in false_negative).items())),
                "unique_patients_with_at_least_one_false_negative": len(patient_fn),
                "false_negative_patient_count_by_true_grade": {
                    str(grade): len({r["patient_id"] for r in false_negative
                                    if int(r["dr_grade_true"]) == grade})
                    for grade in sorted({int(r["dr_grade_true"]) for r in false_negative})
                },
                "false_positive_images_at_0_5": len(false_positive),
                "positive_patient_support": len({r["patient_id"] for r in positives}),
            },
            "ungradable_head": ({
                "n_images": len(quality),
                "gradable_images": quality_counts.get(0, 0),
                "ungradable_images": quality_counts.get(1, 0),
                "false_negative_ungradable_at_0_5": qfn,
                "false_positive_ungradable_at_0_5": qfp,
            } if quality else None),
        }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for s, v in report["by_source"].items():
        e = v["referable_dr"]
        print(s, json.dumps({k: e[k] for k in (
            "n_positive_images", "false_negative_images_at_0_5",
            "false_negative_counts_by_true_grade", "unique_patients_with_at_least_one_false_negative",
            "false_positive_images_at_0_5")}, sort_keys=True))


if __name__ == "__main__":
    main()
