# AMD and glaucoma-suspect development evidence

**Purpose:** consolidate the current AMD result and the strongest usable glaucoma-suspect evidence before moving to the remaining disease tracks. All values are internal development results; neither task has independent external or clinical validation.

## AMD: Zhejiang UWF

The patient-grouped Zhejiang train/validation split contains 137 validation eyes, including 16 AMD-positive eyes. In the newly completed partial ConvNeXt-Tiny fine-tune, a validation-selected threshold targeting specificity ≥90% yielded sensitivity 93.75% (1/16 missed), specificity 94.21%, AUROC 0.951 and AUPRC 0.766. The patient-cluster bootstrap sensitivity interval was 81.25–100%, conditional on the same selected threshold. Frozen-feature DINOv2 and regularization sweeps had already been tried on this same small validation cohort, so candidate selection is multiply exposed. Do not call this external or clinical performance.

## Glaucoma-suspect label: HYGD GON+/GON−

HYGD contains 626 images and 244 patient groups. Its group-verified development partition has 501 training images and 125 validation images (95 GON+ and 30 GON−); the validation set represents 43 patients. This task predicts the dataset's GON label and is only a glaucoma-suspect/referral research task. It does not establish a definitive glaucoma diagnosis, and the same validation patients were used in previous project development.

| Frozen encoder / head | Accuracy at fixed 0.5 | Sensitivity | Specificity | AUROC | AUPRC | Development read |
|---|---:|---:|---:|---:|---:|---|
| RETFound-MAE ViT-L/16 | 96.8% | 95.8% | 100.0% | 0.991 | 0.998 | Highest frozen-feature AUROC among the compared foundation encoders; candidate only. |
| DINOv2 ViT-B/14 | 93.6% | 92.6% | 96.7% | 0.984 | 0.995 | Strong second candidate. |
| VisionFM ViT-Base/16 | 94.4% | 95.8% | 90.0% | 0.963 | 0.985 | Higher sensitivity at 0.5, lower specificity and discrimination than RETFound. |
| FLAIR | 84.8% | 85.3% | 83.3% | 0.944 | 0.982 | Weaker on this source/task. |
| RETFound-MAE partial supervised adaptation | 96.0% | not in preserved aggregate | not in preserved aggregate | 0.978 | not in preserved aggregate | Adaptation did not exceed the frozen model's AUROC in the saved comparison. |

The values are from previous train/validation-only probes on the same patient-grouped manifest. They are not a fresh rerun, not an untouched cohort, and not comparable at a clinically chosen operating point: the table uses a common 0.5 cutoff. The frozen RETFound checkpoint is licensed CC BY-NC 4.0; use is bounded to the project's stated noncommercial research purpose and preserve required attribution. Its availability does not authorize redistribution of the checkpoint.

## Decision

- Retain frozen RETFound-MAE as the leading *research candidate* for the HYGD GON task; do not claim confirmed clinical performance or use it to reassure a patient.
- Keep glaucoma-suspect separate from AMD and other disease heads. Do not translate GON+ into a definitive glaucoma diagnosis.
- Keep AMD and glaucoma evidence in separate source/task run cards. Both remain development-only.
- Further training on the already-used HYGD validation split would add little independent evidence. The next meaningful glaucoma step requires an unused patient-level cohort or a predeclared new evaluation source.

## Provenance

Source aggregates: `results/technical_dev_20261108_rerun/retfound_mae_aggregate_summary.json`, `dinov2_aggregate_summary.json`, `visionfm_aggregate_summary.json`, and `flair_aggregate_summary.json`; RETFound adaptation: `retfound_mae_adaptation/summary.json` and the dated project status record. The source patient IDs, image rows and model predictions are not reproduced in this report.
