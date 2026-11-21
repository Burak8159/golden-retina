# Zhejiang UWF disease-versus-all validation comparison

**Date:** 2026-11-21
**Role:** development candidate comparison only.
**Task:** six separate disease-specific heads; each target is positive and all other source labels, including Healthy, are negative.
**Split:** frozen patient-grouped split. The run loaded 425 training images and 137 validation images (121 validation patients); the 138-image test partition was never loaded.
**Unit:** eye, with image scores max-pooled within eye. Training weights balance each patient and distribute negative weight equally across the non-target source labels.
**Threshold:** selected on the same validation set to maximize sensitivity while achieving at least 90% specificity. Thresholded performance is therefore optimistic and for candidate selection only.

## Results

| Target (positive eyes / all validation eyes) | Encoder | Accuracy | Sensitivity | Specificity | Misses / positives | AUROC | AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|
| AMD (16/137) | ConvNeXt-Tiny | 93.4% | 93.8% | 93.4% | 1/16 | 0.959 | 0.682 |
| AMD (16/137) | DINOv2 ViT-B/14 | 92.7% | 93.8% | 92.6% | 1/16 | 0.966 | 0.849 |
| DR (21/137) | ConvNeXt-Tiny | 94.9% | 90.5% | 95.7% | 2/21 | 0.965 | 0.836 |
| DR (21/137) | DINOv2 ViT-B/14 | 91.2% | 85.7% | 92.2% | 3/21 | 0.960 | 0.771 |
| Pathological myopia (20/137) | ConvNeXt-Tiny | 90.5% | 90.0% | 90.6% | 2/20 | 0.962 | 0.874 |
| Pathological myopia (20/137) | DINOv2 ViT-B/14 | 94.2% | 90.0% | 94.9% | 2/20 | 0.953 | 0.870 |
| Retinal detachment (20/137) | ConvNeXt-Tiny | 90.5% | 85.0% | 91.5% | 3/20 | 0.915 | 0.813 |
| Retinal detachment (20/137) | DINOv2 ViT-B/14 | 92.0% | 85.0% | 93.2% | 3/20 | 0.943 | 0.881 |
| RVO (20/137) | ConvNeXt-Tiny | 93.4% | 90.0% | 94.0% | 2/20 | 0.977 | 0.889 |
| RVO (20/137) | DINOv2 ViT-B/14 | 89.8% | 75.0% | 92.3% | 5/20 | 0.942 | 0.795 |
| Uveitis (20/137) | ConvNeXt-Tiny | 85.4% | 40.0% | 93.2% | 12/20 | 0.844 | 0.484 |
| Uveitis (20/137) | DINOv2 ViT-B/14 | 85.4% | 55.0% | 90.6% | 9/20 | 0.866 | 0.640 |
| AMD (16/137) | FLAIR | 91.2% | 93.8% | 90.9% | 1/16 | 0.971 | 0.851 |
| DR (21/137) | FLAIR | 97.1% | 90.5% | 98.3% | 2/21 | 0.973 | 0.846 |
| Pathological myopia (20/137) | FLAIR | 89.8% | 85.0% | 90.6% | 3/20 | 0.926 | 0.776 |
| Retinal detachment (20/137) | FLAIR | 86.1% | 50.0% | 92.3% | 10/20 | 0.780 | 0.481 |
| RVO (20/137) | FLAIR | 89.8% | 65.0% | 94.0% | 7/20 | 0.876 | 0.721 |
| Uveitis (20/137) | FLAIR | 83.9% | 45.0% | 90.6% | 11/20 | 0.829 | 0.392 |

## Interpretation and limits

- Compared with the easier disease-versus-Healthy tasks, validation sensitivity falls for several heads once competing retinal labels are included. Disease-versus-Healthy results should not be treated as a realistic multi-disease screening estimate.
- At the validation-selected operating points, uveitis remains a clear failure for both encoders: 9–12 of 20 positive validation eyes are missed. RD also misses 3 of 20 for either encoder. Neither supports a reassuring negative result.
- DINOv2 gives higher AMD and RD AUPRC, while ConvNeXt is stronger on DR and RVO in these validation results. PM estimates are similar. This is not sufficient to establish a universal best encoder; disease-specific candidate selection is more appropriate.
- FLAIR has the strongest validation DR AUROC and similar AMD AUROC, but its RD, RVO, and uveitis sensitivities are poor at the selected operating points. The encoder results support disease-specific candidate comparisons, not one shared model ranking.
- Validation is small, threshold-selected and used repeatedly for model development. No confidence interval is reported here. Metrics are not clinical risks, diagnoses, or external validation.
- The source labels are mutually exclusive; this evaluation does not test co-occurring diseases or multi-label presentations. Zhejiang DR is additionally exposed by earlier project analysis.
- The initial source test partition was scored by the earlier disease-versus-Healthy probe and is excluded from all candidate selection here. It cannot now be called an untouched project-wide test.

## Competing-label false-positive review

The following are the two largest *counts* of false positives by source label at each model's validation-selected threshold. These are descriptive error-review counts, not evidence that the model learned a disease-specific mechanism. Small denominators make the rates unstable.

| Target head | ConvNeXt-Tiny | DINOv2 ViT-B/14 | FLAIR |
|---|---|---|---|
| AMD | DR 7/21; RD 1/20 | DR 9/21 | DR 10/21; RD 1/20 |
| DR | AMD 4/16; RVO 1/20 | AMD 5/16; RVO 2/20 | AMD 1/16; RD 1/20 |
| Pathological myopia | RVO 4/20; AMD 2/16 | RVO 4/20; RD 1/20 | RVO 7/20; RD 2/20 |
| Retinal detachment | Healthy 4/20; uveitis 4/20 | Uveitis 3/20; RVO 2/20 | Uveitis 4/20; PM 2/20 |
| RVO | Uveitis 5/20; PM 1/20 | Uveitis 5/20; RD 2/20 | PM 3/20; Healthy 2/20 |
| Uveitis | Healthy 7/20; RD 1/20 | Healthy 9/20; RD 1/20 | RD 5/20; Healthy 4/20 |

The pattern flags important class confusions for subsequent label and image-quality review: DR is the most frequent competing label called AMD by all three encoders; AMD is the leading competing label called DR by ConvNeXt and DINOv2; RVO is often called pathological myopia; and Healthy cases frequently cross the uveitis threshold. This does not establish why these errors occurred. Individual images and predictions remain private on the VM.

Only aggregate metrics are published here. Patient/image IDs, per-image scores, images, and checkpoints remain in the authorized private archive and are not part of the public repository.
