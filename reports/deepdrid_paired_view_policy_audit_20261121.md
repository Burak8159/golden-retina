# DeepDRiD paired-view policy audit

**Status:** development-only audit completed from saved cross-fit predictions.
**Task:** compare a fixed first-view fundus score with a paired macula-plus-optic-disc score, then evaluate a simple second-view acquisition rule calibrated on calibration patients and applied once to validation patients.
**No raw images were fetched to the workstation.** The aggregate report and scoring utility are reproducible; row-level prediction files remain restricted to their existing private location.

## Data and integrity findings

Both visual-suffix and swapped-suffix prediction files exist; a prior status JSON incorrectly said the required predictions were missing. Each contains cross-fitted train/calibration/validation rows and held-out-model scores for the historical test partition, with patient, eye, split, view set, grade, fold and prediction-scope fields. Patient partitions are disjoint and paired labels agree. The validation pair count is 78 eyes / 39 patients, two fewer than the nominal 80 / 40; calibration has 80 / 40.

The filename suffix-to-anatomical-field mapping remains unverified from an official source. The two branches are therefore reported separately. Never average them or choose the branch based on a previously accessed test partition.

## Paired-score comparison

| Mapping branch | Partition | Paired eyes / patients | AUROC: first → two views | AUPRC: first → two views | Brier: first → two views |
|---|---|---:|---:|---:|---:|
| Inferred suffix 1 = disc | Calibration | 80 / 40 | 0.776 → 0.799 | 0.668 → 0.712 | 0.180 → 0.170 |
| Inferred suffix 1 = disc | Validation | 78 / 39 | 0.849 → 0.834 | 0.781 → 0.772 | 0.167 → 0.171 |
| Swapped sensitivity branch | Calibration | 80 / 40 | 0.751 → 0.799 | 0.679 → 0.712 | 0.207 → 0.170 |
| Swapped sensitivity branch | Validation | 78 / 39 | 0.766 → 0.834 | 0.642 → 0.772 | 0.195 → 0.171 |

Patient-cluster intervals for validation AUROC differences include zero under the inferred branch (−0.068 to 0.038) but not under the swapped branch (+0.015 to +0.133). The conclusion about first-view value changes with the unresolved view mapping; this is an explicit provenance blocker.

## Calibration-only second-view policy

A simple two-threshold rule was calibrated on the 40-patient calibration partition: clear an eye after the first image only when its score is below the minimum first-view score among calibration positives; otherwise acquire a second view and label positive when the paired score is at least the minimum paired score among calibration positives. This yields zero *observed* calibration misses by construction and is not a learned or risk-certified policy.

| Mapping branch | Calibration: misses / positive patients | Calibration specificity | Validation: misses / positive patients | Validation specificity (patient-bootstrap 95% CI) | Validation acquire-second-view rate (95% CI) | Binary action coverage |
|---|---:|---:|---:|---:|---:|---:|
| Inferred suffix 1 = disc | 0 / 21 | 29.3% | 0 / 18 | 32.6% (16.7–50.0%) | 83.3% (74.3–92.3%) | 100% |
| Swapped sensitivity branch | 0 / 21 | 26.8% | 0 / 18 | 26.1% (12.5–40.9%) | 94.9% (88.5–100%) | 100% |

For the validation branch with zero missed positive patients out of 18, the one-sided 95% exact upper bound on patient-level miss risk is 15.3%; on calibration it is 13.3% from 21 positive patients. Thus “zero observed misses” is not evidence of zero risk. The rule preserves binary action coverage by construction, but achieves low specificity at a very high second-image rate. It is not useful as a deployment policy.

## Decision and limits

- Keep the DeepDRiD result as a two-view development audit only. It does not demonstrate useful adaptive acquisition, clinically acceptable miss risk, or multi-step non-myopic planning.
- Do not promote either view mapping until the official suffix mapping is confirmed; the value conclusion is mapping-sensitive.
- The historical `sealed_test` rows appear in the saved prediction files and this project has previously used DeepDRiD for development. Their metrics are excluded from this decision report and must not be described as untouched external confirmation.
- A real `STOP / ACQUIRE_SECOND_VIEW / DEFER` policy needs more positive calibration patients, a prespecified risk target, patient-level risk control, adequate specificity, and a genuinely unexposed confirmation cohort. MFIDDR/mBRSET could provide additional views only after access and patient/view metadata are verified; neither can be assumed to fix this result.

Aggregate file: [`deepdrid_paired_view_crossfit_aggregate_20261121.json`](deepdrid_paired_view_crossfit_aggregate_20261121.json). Reproducible utility: [`../scripts/audit_deepdrid_paired_crossfit_aggregate.py`](../scripts/audit_deepdrid_paired_crossfit_aggregate.py).
