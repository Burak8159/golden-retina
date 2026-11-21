# Cross-track uncertainty and evaluation-boundary closeout

## Patient-grouped Zhejiang internal test

The current campaign’s six-output ConvNeXt-Tiny model was evaluated on one internal patient-grouped Zhejiang UWF partition (136 eyes / 120 patients). Thresholds were selected on validation only. Sensitivity intervals below are patient-cluster bootstrap 95% intervals; small positive-patient counts make them wide.

| Target | Sensitivity | False negatives | Patient-cluster 95% CI for sensitivity | Interpretation |
|---|---:|---:|---:|---|
| DR | 94.7% | 1/19 | 82.4–100.0% | Internal, source-exposed development estimate |
| AMD | 94.1% | 1/17 | 78.6–100.0% | Internal, source-exposed development estimate |
| Pathological myopia | 85.0% | 3/20 | 70.0–100.0% | Misses positives; not reassurance-safe |
| Retinal detachment | 95.0% | 1/20 | 85.0–100.0% | Misses positives; internal only |
| RVO | 95.0% | 1/20 | 85.0–100.0% | Aggregate RVO target; internal only |
| Uveitis | 35.0% | 13/20 | 15.0–55.1% | Clear no-go for screening |

These intervals quantify sampling uncertainty conditional on this cohort and protocol. They do not account for dataset shift, label error, source selection, repeated project development, or clinical workflow effects. The prior DR analysis used the same source, so the test is not project-wide untouched external validation.

## RFMiD 1.0 shared multi-label model

The 24-output ConvNeXt-Tiny run used an image-level partition with 640 test images. Patient identifiers were unavailable, the test partition was exposed in project work, and rare positive supports are too small for reliable patient-level uncertainty. The aggregate macro AUROC is 94.24% and macro average precision is 56.96%; element-wise accuracy is 75.30% and mean per-label accuracy is 95.15%. Exact-set accuracy is 0%. These metrics are descriptive only and do not measure clinical screening utility. No patient-bootstrap interval is reported because patient resampling units cannot be established.

Every disease label must be read with support and false-negative counts. Examples of severe failure include ERM 0/5 detected, AION 1/4 detected, and CRVO 5/9 detected in the expanded RFMiD run. High accuracy in these sparse-label rows is dominated by negatives.

## RIM-ONE DL glaucoma-suspect track

RIM-ONE DL’s hospital partition contains 174 images (56 glaucoma). The source-only scratch model achieved 87.5% sensitivity and 50.85% specificity, missing 7/56 positive images. The hospital test was already opened by an earlier protocol-deviation run; patient identifiers were absent. The first ConvNeXt model’s image-level AUROC interval (0.849–0.942) does not repair that protocol issue and is not a confirmatory estimate. No patient-level confidence interval or clinical-performance claim is made for the source-only rerun.

## Cross-track conclusion

No track establishes autonomous diagnosis or safe negative clearance. All rows remain research development/benchmark evidence. A future confirmatory evaluation needs a prospectively frozen version and threshold, a new patient-identified cohort from an independent source, blinded reference grading, prespecified operating points, and patient-level uncertainty analysis. The report intentionally does not pool unlike diseases, labels, modalities, or cohorts into a single “all retinal disease” accuracy.
