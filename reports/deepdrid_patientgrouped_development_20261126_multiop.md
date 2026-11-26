# DeepDRiD patient-grouped candidate comparison

**Status:** internal development analysis; sealed test was not accessed and sealed labels were not parsed.
**Task:** referable diabetic retinopathy (grade >=2), image/eye predictions evaluated with patient-cluster bootstrap.
**Thresholding:** each threshold was selected on the disjoint calibration patients to achieve 100% observed calibration sensitivity. This does not guarantee 100% sensitivity on validation or future cohorts.

## Validation results at calibration-set 100% sensitivity

| Encoder | View | Validation AUROC (patient-bootstrap 95% CI) | Sensitivity | Misses | Specificity | Accuracy |
|---|---|---:|---:|---:|---:|---:|
| resnet50 | macula | 0.841 (0.718–0.934) | 94.1% | 2/34 | 50.0% | 68.8% |
| resnet50 | optic disc | 0.874 (0.771–0.952) | 96.9% | 1/32 | 34.8% | 60.3% |
| resnet50 | macula+optic disc | 0.895 (0.807–0.960) | 100.0% | 0/32 | 34.8% | 61.5% |
| convnext | macula | 0.907 (0.820–0.973) | 97.1% | 1/34 | 63.0% | 77.5% |
| convnext | optic disc | 0.908 (0.835–0.966) | 100.0% | 0/32 | 37.0% | 62.8% |
| convnext | macula+optic disc | 0.911 (0.824–0.977) | 100.0% | 0/32 | 30.4% | 59.0% |
| dinov2 | macula | 0.896 (0.808–0.961) | 97.1% | 1/34 | 56.5% | 73.8% |
| dinov2 | optic disc | 0.861 (0.753–0.942) | 96.9% | 1/32 | 45.7% | 66.7% |
| dinov2 | macula+optic disc | 0.912 (0.829–0.973) | 93.8% | 2/32 | 65.2% | 76.9% |

## Interpretation

The highest specificity among these fixed validation operating points was DINOv2 with macula+optic-disc views (65.2%), with 2 of 32 referable eyes missed and 76.9% accuracy. ConvNeXt-Tiny with macula+optic-disc reached zero observed misses in this validation sample (32/32 sensitivity) but specificity was 30.4% and accuracy 59.0%. These are candidate-specific internal estimates; they are not a deployable threshold recommendation.

At the 95% calibration-sensitivity setting, paired-view results were: convnext: AUROC 0.911, sensitivity 93.8% (30/32; 2 misses), specificity 60.9%, accuracy 74.4%; dinov2: AUROC 0.912, sensitivity 93.8% (30/32; 2 misses), specificity 69.6%, accuracy 79.5%.

**Limitations:** dataset was previously used in the broader project; small internal validation (40 patients; paired views 39 patients); view suffix mapping is inferred rather than official; bootstrap intervals are wide; thresholds were calibrated on a small separate calibration set. No independent external or clinical validation is established. Do not interpret any score as a diagnosis.

Machine-readable aggregate: `deepdrid_patientgrouped_development_20261126_multiop.json`. Row-level predictions, identifiers and images remain on the RTX.
