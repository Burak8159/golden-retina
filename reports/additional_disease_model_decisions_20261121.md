# Additional retinal-disease model decisions

## Current patient-grouped Zhejiang UWF development cohort

The current campaign has six mutually exclusive source labels. A shared partially fine-tuned ConvNeXt-Tiny model was fit as six independent one-vs-rest heads and evaluated on the 137-eye validation split. At same-validation thresholds selected for specificity ≥90%, results were:

| Disease | Sensitivity / misses | Specificity | AUROC | Decision |
|---|---:|---:|---:|---|
| Pathological myopia | 85.0% (3/20) | 90.6% | 0.949 | Keep exploratory; insufficient sensitivity for reassurance; confirm the pathological-myopia label definition. |
| Retinal detachment | 90.0% (2/20) | 94.0% | 0.963 | Keep exploratory only; small sample and possible field-coverage limitations make a negative result unsafe. |
| Retinal vein occlusion | 100.0% (0/20) | 90.6% | 0.991 | Candidate signal only; 20/20 validation detections do not establish zero miss risk. Report subtype not assessed. |
| Uveitis | 40.0% (12/20) | 93.2% | 0.838 | Do not use as a reassuring-negative screen; source label may include disease not visible in fundus photographs. |

For context, a preceding three-encoder and C sweep on the same validation cohort selected ConvNeXt/DINOv2 candidates with 95% sensitivity for PM, 85% for RD, 95% for RVO and 70% for uveitis. Those choices and thresholds were selected on the same validation data and are optimistic development signals. The subsequent shared fine-tune improved validation loss but did not uniformly improve sensitivity. It must not be used to replace the disease-specific candidates wholesale.

## Targets excluded from this stage

- Hypertensive retinopathy and systemic hypertension: no verified disease-specific label cohort was staged for this campaign.
- Central serous retinopathy, retinitis/chorioretinitis, and epiretinal membrane: only sparse legacy evidence or no adequate current patient-linked cohort; do not claim a model from those rows.
- MMRDR lesion findings: the image archive/labels are not staged in the authorized VM workspace. No lesion model was trained.
- Optic-disc cupping/pallor and tessellation/scars: image findings, not standalone disease diagnoses; legacy class support and error rates do not justify a disease-screen claim.

The detailed numbers, caveats, and source references are in [`disease_by_disease_overview_20261121.md`](../docs/disease_by_disease_overview_20261121.md) and the fine-tune aggregate in [`zhejiang_convnext_end_to_end_validation_aggregate_20261121.md`](zhejiang_convnext_end_to_end_validation_aggregate_20261121.md). All current-campaign results are single-source development estimates, with validation used for early stopping, thresholding, and model selection. The prior exposed DR subset cannot serve as a fresh confirmation cohort.
