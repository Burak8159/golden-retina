# RETFound-Green quality-supervision ablation

## Question

Does training an auxiliary image-gradability head using DDR labels change disease-task transfer on the fixed internal validation cohort?

This experiment is not a validation of image-quality classification: the validation cohort has no quality labels.

## Data and split boundary

- Frozen manifest: `results/technical_dev_20261108_rerun/train_valid_manifest.csv`.
- Disease validation is unchanged: 878 images from DeepDRiD, DRTiD and HYGD; no DDR validation rows and no test rows.
- DDR contributes 3,002 train-only images: 2,434 with DR grades and 568 labeled ungradable.
- The two runs use the same RETFound-Green checkpoint, seed, source/label sampler, sampler generator, data-loader generator, augmentations and patient-grouped validation. The matched control has no quality loss; the intervention adds masked DDR gradability BCE.
- Any validation result is an exposed internal development estimate, not independent external validation or clinical performance.

## Prespecified readout

Select each run's checkpoint by validation referable-DR AUROC. Compare both at the fixed 0.5 threshold, at image and patient-source-group levels. Report AUROC, AUPRC, accuracy, sensitivity, specificity, grade accuracy, glaucoma metrics and patient-bootstrap intervals. Keep disease-specific/source-specific outcomes visible.

The quality auxiliary loss uses `ungradable=1` as positive and `ungradable=0` as gradable. DDR quality labels are training-only. Do not report quality-head accuracy, sensitivity, specificity or calibration as validated performance because there are zero labeled quality-validation cases.

## Current run status

The first control attempt stopped after one epoch because a minibatch contained only unlabeled images and mixed-precision gradient scaling had no optimizer gradients to inspect. That attempt is invalid and excluded. The trainer now skips batches with no active supervision; a fresh matched control is running in `retfound_green_quality_control_v2`.

Results will be appended only after both matched runs finish and patient-level comparison is complete. Model weights and row-level predictions remain private; this file is an aggregate-only protocol note.


## Results

Both matched 16-epoch arms were selected by image-level validation referable-DR AUROC and then evaluated at patient-source-group level. The no-quality control had AUROC 0.9487 (95% patient-bootstrap CI 0.9241–0.9696), accuracy 80.89% (76.43–85.03), sensitivity 94.67% (91.02–97.99), and specificity 68.29% (60.82–75.32) at a fixed 0.5 threshold. The quality-auxiliary candidate had AUROC 0.9466 (0.9197–0.9690), accuracy 80.57% (76.43–85.03), sensitivity 96.00% (92.52–98.71), and specificity 66.46% (59.12–73.51).

Paired 5,000-resample patient-source bootstrap of candidate minus control gave AUROC −0.21 percentage points (95% CI −1.40 to +0.95), AUPRC −0.27 pp (−1.19 to +0.58), accuracy −0.32 pp (−3.82 to +3.18), sensitivity +1.33 pp (−1.28 to +4.20), and specificity −1.83 pp (−7.73 to +4.24). Every interval includes zero. The auxiliary objective therefore has no supported disease-transfer benefit on this exposed validation cohort; it is not retained as an improvement. No quality-classification metric is reported because validation quality labels are absent. These results are internal development evidence, not external or clinical validation. Aggregate-only readout is stored in `results/technical_dev_20261108_rerun/retfound_green_quality_ablation_patient_summary.json`; row-level predictions and checkpoints remain private on the L4.
