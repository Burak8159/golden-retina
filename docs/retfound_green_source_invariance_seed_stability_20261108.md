# RETFound-Green source-invariance seed stability

## Protocol

The source-adversarial adaptation was repeated with seed 20261112 under the same frozen train/validation manifest, initialization, batch size, loss weights, and 16-epoch selection rule as seed 20261111. Each seed selected its checkpoint by validation referable-DR AUROC. Evaluation grouped by `(source, patient_id)` and used the same patient-level aggregation and fixed 0.5 threshold. The validation cohort is exposed internal development data; no test rows were used.

## Results

Across the same 314 patient-source groups (150 positive):

| Metric | Seed 1 (20261111) | Seed 2 (20261112) | Seed 2 minus seed 1 (paired 95% CI) |
|---|---:|---:|---:|
| Accuracy | 89.81% | 82.17% | -7.64 pp (-10.83, -4.14) |
| Sensitivity | 90.00% | 92.67% | +2.67 pp (+0.62, +5.44) |
| Specificity | 89.63% | 72.56% | -17.07 pp (-22.88, -11.46) |
| AUROC | 0.94911 | 0.94681 | -0.23 pp (-0.94, +0.49) |
| AUPRC | 0.95460 | 0.95431 | -0.03 pp (-0.66, +0.67) |

The paired bootstrap used 5,000 patient-source resamples. Seed-specific AUROC and AUPRC are similar, with intervals for their differences crossing zero; the fixed-threshold accuracy and specificity differences are substantial. The high-specificity fixed-threshold point from seed 1 did not reproduce in seed 2. Source-head validation balanced accuracy also varied across epochs and seeds rather than showing consistent source suppression.

## Interpretation and freeze decision

This does not support source-invariance as a reproducible disease-transfer improvement. It shows that an intervention can materially change the operating point while leaving ranking metrics essentially unchanged, and that the apparent specificity gain is seed-sensitive. Freeze the RETFound-Green seed-1 checkpoint only as the current research comparator selected by the prespecified validation AUROC rule; retain the fixed 0.5 threshold and record the seed sensitivity. Do not describe the 89.63% specificity as independently validated, calibrated, or clinical performance. Do not use the seed-1 operating point as a deployment guarantee.

Aggregate-only artifacts are `results/technical_dev_20261108_rerun/retfound_green_source_invariance_01_v2/patient_level_summary.json` and `results/technical_dev_20261108_rerun/retfound_green_source_invariance_02/source_invariance_seed_stability_patient.json`. Checkpoints and row-level predictions remain private on the VM.
