# RFMiD 1.0 shared ConvNeXt refresh

**Status:** exploratory image-split development evidence; not clinical validation.

## Run

- Dataset: [RFMiD 1.0 mirror](https://huggingface.co/datasets/ctmedtech/RFMID), revision `16ee0cf528e9cb6ec38a6d7b1f3e4580172bc1da`. The mirror card declares CC BY 4.0 and links the upstream [RFMiD data descriptor](https://doi.org/10.3390/data6020014).
- Model: one shared ImageNet-initialized `torchvision.convnext_tiny`, input size 320, seed 20261121; best epoch 14 by validation BCE.
- Training: AdamW, learning rate 2e-5, weight decay 2e-4, batch size 32, two data-loader workers, 30-epoch maximum with validation-BCE patience 5, and `BCEWithLogitsLoss` positive weights clipped to [1, 10]. The run stopped at epoch 19 after five stale epochs. PyTorch 2.9.1+cu129 / torchvision 0.24.1+cu129 / CUDA 12.9 on NVIDIA L4.
- Official image partitions: 1920 train / 640 validation / 640 test.
- Label inclusion: RFMiD's 45 source condition/finding columns are retained after excluding the dataset aggregate `Disease_Risk`; a label is trained only when it has at least 30 positive and 30 negative training examples. Fourteen labels met that floor; the rest are not trained heads.
- Per-label threshold: validation-only maximum-sensitivity threshold subject to a 90% validation-specificity target. Test labels were not used to choose thresholds.

## Aggregate held-out image-split metrics

- Macro AUROC across train-supported labels with both test classes: **94.38%**.
- Macro average precision across train-supported labels with test positives: **69.88%**.
- Mean per-label accuracy across the 14 train-supported labels: **92.83%**. This is an average of binary per-label accuracies, not conventional single-label accuracy.
- Mean elementwise accuracy across all 45 retained source labels: 69.65%; it includes 31 outputs excluded for inadequate training support and is not used as the headline metric.
- Exact-content hash audit: 3,200 image files hashed; zero identical-content groups crossed official splits; no missing/ambiguous images. This does not establish patient separation.
- The official test partition was used in earlier project analyses; this run's test figures are therefore not a project-wide untouched holdout, even though this run selected its checkpoint and thresholds using validation only.

## Per-label outcomes

| Source label | Interpretation | Test positives | Sensitivity | Specificity | Accuracy | AUROC | AP | FN | FP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DR | Diabetic retinopathy | 124 | 93.5% | 94.8% | 94.5% | 97.9% | 92.9% | 8 | 27 |
| ARMD | Age-related macular degeneration (RFMiD source label) | 31 | 87.1% | 92.6% | 92.3% | 96.3% | 65.5% | 4 | 45 |
| MH | Media haze | 104 | 94.2% | 88.4% | 89.4% | 97.4% | 90.6% | 6 | 62 |
| DN | Drusen | 46 | 52.2% | 92.9% | 90.0% | 85.7% | 42.8% | 22 | 42 |
| MYA | Myopic fundus appearance | 32 | 96.9% | 96.4% | 96.4% | 99.6% | 94.2% | 1 | 22 |
| BRVO | Branch retinal vein occlusion (RFMiD source label) | 23 | 87.0% | 90.8% | 90.6% | 92.9% | 78.8% | 3 | 57 |
| TSLN | Retinal tessellation | 53 | 88.7% | 91.1% | 90.9% | 96.1% | 71.3% | 6 | 52 |
| LS | Laser scar | 15 | 80.0% | 95.2% | 94.8% | 90.9% | 70.4% | 3 | 30 |
| CSR | Central serous retinopathy (RFMiD source label) | 13 | 84.6% | 98.2% | 98.0% | 99.1% | 74.7% | 2 | 11 |
| ODC | Optic-disc cupping | 91 | 68.1% | 90.9% | 87.7% | 91.7% | 72.7% | 29 | 50 |
| ODP | Optic-disc pallor | 24 | 54.2% | 91.9% | 90.5% | 85.5% | 36.0% | 11 | 50 |
| ODE | Optic-disc edema | 17 | 82.4% | 97.1% | 96.7% | 95.0% | 76.2% | 3 | 18 |
| RS | Retinitis (RFMiD source label) | 14 | 85.7% | 95.4% | 95.2% | 98.0% | 76.0% | 2 | 29 |
| CRS | Chorioretinitis (RFMiD source label) | 11 | 81.8% | 92.8% | 92.7% | 95.2% | 36.1% | 2 | 45 |

The model is a shared multi-label classifier, not a collection of disease-specific expert models. “Sensitivity” and “specificity” above are image-level results at per-label thresholds selected on validation; test specificity fell below 90% for some labels. Positive supports are small for several labels, so point estimates are unstable.

## Interpretation and limitations

The results support further research on these RFMiD source labels, not a generalized retinal disease detector, autonomous screening, diagnosis, or deployment claim. Patient IDs were unavailable; the split is image-level. The project had previously used/examined the official test partition, so these refreshed numbers are not an untouched external validation. No confidence intervals, calibration assessment, camera/source subgroup evaluation, or prospective workflow evaluation was completed. The source label “ARMD” does not by itself establish disease stage; “MYA” is fundus appearance rather than a refractive diagnosis; optic-disc cupping is not glaucoma; and media haze is an image-quality/visibility label.

The complete private run summary, training log, source script snapshot, and checkpoint are retained under the Git-ignored `private_vm_artifacts/20261121/rfmid1_l4_03/` archive; only aggregate metrics and reproducible code are intended for the public repository.
