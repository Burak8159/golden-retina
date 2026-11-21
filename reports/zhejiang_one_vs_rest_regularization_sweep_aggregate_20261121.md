# Zhejiang UWF disease-versus-all regularization sensitivity

**Date:** 2026-11-21
**Role:** exploratory development selection only.
**Data:** 425 train images and 137 validation eyes from a patient-grouped Zhejiang split; test partition not loaded.
**Models:** frozen ConvNeXt-Tiny, DINOv2 ViT-B/14, and FLAIR representations with patient/source-balanced logistic heads.
**Sweep:** logistic-regression `C` in {0.03, 0.1, 1, 10}. For each C, the validation threshold was separately selected to maximize sensitivity at specificity >=90%. The C and threshold are both chosen using the same small validation set, so reported selected values are strongly optimistic.

## Validation-selected candidate per disease

Candidate encoder/C was selected by maximum sensitivity, then specificity, then AUROC, subject to the validation threshold policy. The C=1 comparison is the original fixed-regularization reference for that selected encoder; its threshold was also selected on validation.

| Disease | Provisional encoder / C | Selected sensitivity / misses | Specificity | AUROC / AUPRC | Same encoder C=1 sensitivity / misses |
|---|---|---:|---:|---:|---:|
| AMD | DINOv2 / 10 | 93.8% (1/16) | 95.9% | 0.965 / 0.865 | 93.8% (1/16) |
| DR* | ConvNeXt / 10 | 95.2% (1/21) | 93.1% | 0.964 / 0.848 | 90.5% (2/21) |
| Pathological myopia | DINOv2 / 0.03 | 95.0% (1/20) | 90.6% | 0.943 / 0.844 | 90.0% (2/20) |
| Retinal detachment | DINOv2 / 0.1 | 85.0% (3/20) | 94.9% | 0.920 / 0.835 | 85.0% (3/20) |
| RVO | ConvNeXt / 0.1 | 95.0% (1/20) | 90.6% | 0.977 / 0.894 | 90.0% (2/20) |
| Uveitis | DINOv2 / 0.03 | 70.0% (6/20) | 94.0% | 0.866 / 0.598 | 55.0% (9/20) |

`*` DR remains exposed at source level through earlier project analysis. The earlier test partition was previously scored and is not a project-wide untouched test, even though this sweep itself did not load it.

## What changed and what did not

- The fixed-encoder linear heads remain unchanged in representation; only logistic regularization and the validation-selected threshold changed.
- On this validation sample, the chosen settings reduce misses by one for DR, pathological myopia, and RVO relative to C=1 on the chosen encoder, and by three for uveitis. RD does not improve. AMD sensitivity is unchanged.
- Uveitis still misses 6/20 positive validation eyes and RD misses 3/20; neither supports a reassuring negative result. These selected validation outcomes are not deployable performance estimates.
- C selection adds another layer of reuse of the same validation cohort. The apparent gains may not reproduce; do not describe them as confirmed performance improvement.
- This result supports keeping multiple disease-specific research heads, not one universal encoder. It does not establish clinical risk probabilities, diagnosis, or real-world accuracy.

## Artifact boundary

This report publishes aggregate metrics only. The selected private head weights, per-eye outputs, images, and manifests remain in the authorized private archive and are not included in the public repository. Source metrics and model-family details are in [`zhejiang_one_vs_rest_validation_aggregate_20261121.md`](zhejiang_one_vs_rest_validation_aggregate_20261121.md).
