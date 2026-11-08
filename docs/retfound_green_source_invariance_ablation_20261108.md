# RETFound-Green source-invariance ablation

## Question

Does a gradient-reversal source classifier improve disease prediction across the three sources present in both training and validation?

## Prespecified separation

- Frozen manifest: `results/technical_dev_20261108_rerun/train_valid_manifest.csv`.
- Same RETFound-Green initialization, training seed (20261111), disease objectives, sampler, batch size, validation cohort, epoch limit, and checkpoint rule as the matched no-quality control.
- Gradient-reversal coefficient: 0.1; source-classification cross-entropy is applied only to DeepDRiD, DRTiD, and HYGD samples. These three sources occur in both train and validation.
- DDR remains eligible for disease supervision, but it is excluded from the source-adversarial head because it is train-only and has no validation rows. This avoids scoring the source head on an unseen validation domain.
- No quality auxiliary objective is used in this ablation.

## Readout and stop rule

Select checkpoints by the same validation referable-DR AUROC. Evaluate source predictability and disease metrics by source, then use source-patient-group paired bootstrap against the matched control. Keep the component only if patient-level disease transfer improves without material source-specific regression. The validation cohort is exposed internal development data; no test rows are loaded and no independent external or clinical claim is supported.

The first implementation stopped after epoch 1: a batch containing only train-only DDR ungradable rows had neither disease labels nor a source-class target, and the trainer did not skip it. That run is invalid and excluded. The trainer was corrected to skip minibatches with no active disease, quality, or source supervision; the fixed 16-epoch run and patient-level comparison below are the valid results. Checkpoints and row-level predictions remain private.

## Completed run and patient-level comparison

The corrected run completed 16 epochs. The selected checkpoint was epoch 8 (validation AUROC 0.93240). At the fixed 0.5 image threshold, epoch 8 had sensitivity 0.8293 and specificity 0.9212 on 753 validation images; later epochs did not improve AUROC. The source-head validation balanced accuracy fluctuated from 0.204 to 0.467 across epochs, so source confusion was not stable.

Patient-level aggregation used the evaluator's source-patient grouping rule. Across 314 common patient-source groups (150 positive), the source-invariance candidate had accuracy 0.8981, sensitivity 0.9000, specificity 0.8963, AUROC 0.9491, and AUPRC 0.9546. The matched no-quality control had accuracy 0.8089, sensitivity 0.9467, specificity 0.6829, AUROC 0.9487, and AUPRC 0.9537, at the same fixed 0.5 patient threshold.

A paired 5,000-draw patient-source bootstrap found candidate-minus-control changes of +8.92 percentage points in accuracy (95% CI +5.10 to +13.06), -4.67 points in sensitivity (-8.23 to -1.48), +21.34 points in specificity (+15.29 to +28.00), +0.04 points in AUROC (-0.61 to +0.70), and +0.09 points in AUPRC (-0.55 to +0.73). Thus, this operating point materially trades sensitivity for specificity while ranking performance is statistically indistinguishable. It does not establish improved disease transfer or a clinically useful source-invariance mechanism. Keep as an ablation finding, not a selected clinical model improvement.

Aggregates are stored in `results/technical_dev_20261108_rerun/retfound_green_source_invariance_01_v2/patient_level_summary.json` and `results/technical_dev_20261108_rerun/source_invariance_vs_control_patient_paired.json`. Row-level predictions and weights remain private on the VM. These are internal development-validation results, not independent external validation.
