# Golden Retina — primary-disease evidence closeout

**Date:** 22 November 2026
**Scope after user direction:** diabetic retinopathy (DR), age-related macular degeneration (AMD), glaucoma, and the most relevant adjacent common conditions with usable labels (cataract and retinal vein occlusion, RVO). Rare tumors and low-priority labels are no longer training priorities.

## What the current evidence says

These are source-specific research measurements, not calibrated patient risks or clinical validation. AUC, sensitivity, specificity, and accuracy are not interchangeable. Results from distinct cohorts should not be averaged.

| Primary condition / cohort | Held-out result | Interpretation |
|---|---|---|
| **DR — FDR26, patient-key-grouped internal test** | Five-grade accuracy **61.30%**, balanced accuracy **61.57%**, macro-F1 **60.81%**, QWK **0.862**; 553 images / 370 derived patient groups. At the dataset-specific grade-2+ proxy threshold: sensitivity **88.13%** (40/337 positive cases missed), specificity **85.65%**. | Best patient-grouped DR grading evidence in this campaign, but single-center and not external/clinical validation. The grade-2+ rule is a dataset proxy, not a validated referral standard. |
| **DR — CLARUS700, official patient-disjoint test** | 23 positives / 163 images; sensitivity **78.26%** (5 misses), specificity **92.14%**, accuracy **90.18%**, AUROC **0.958**. | Patient-disjoint, but one center/device and small positive support. |
| **DR — UWF Peng, reconstructed filename-group split** | 328 images; three-grade accuracy **92.07%**. Binary disease-vs-normal sensitivity **99.10%** (2 misses), specificity **94.29%**. | Wide-field images; filename grouping is not verified patient identity. Exploratory only. |
| **AMD — Mendeley multi-source compilation** | 4,491-image test: sensitivity **94.92%** (19/374 misses), specificity **97.30%**, AUROC **0.993**. | Not accepted as reliable disease performance: no patient/source-to-image map. A metadata/color-only model reached AMD AUROC **0.829**, indicating material acquisition/source predictability. |
| **AMD — Rajbari/BMRD lineage audit** | Nominal Rajbari image-group test was near-perfect; perceptual matching found substantial likely cross-source image lineage with BMRD. | The apparent cross-source result is withdrawn as independent evidence. Do not use its 100% figure as transfer performance. |
| **Glaucoma — RIM-ONE hospital-partition test** | 174 images; sensitivity **83.93%** (9 misses), specificity **75.42%**, accuracy **78.16%**, AUROC **0.899**. | Patient identifiers are absent; the hospital partition is not prospective clinical validation. |
| **Glaucoma — PAPILA patient-grouped test** | 96 eyes; glaucoma-or-suspect proxy sensitivity **36.67%** (19/30 missed), specificity **90.91%**, accuracy **73.96%**. | Small test; “suspect” is not confirmed glaucoma. Current threshold misses too many positives for a screening claim. |
| **Glaucoma — Mendeley threshold transferred to JustRAIGS internal test** | 6,000 images, 198 referable glaucoma: sensitivity **28.79%** (141/198 missed), specificity **96.05%**, accuracy **93.83%**, AUROC **0.720**. | High accuracy is class-imbalance driven; the transferred cutoff fails badly on sensitivity. Not the official challenge test. |
| **Cataract — Mendeley compilation** | Image-group test sensitivity **96.52%** (10/287 misses), specificity **96.91%**, AUROC **0.996**. | Not accepted as a robust disease estimate: the metadata/color-only baseline reached AUROC **0.944**; patient/source linkage is missing. |
| **RVO — CLARUS700 official patient-disjoint test** | 11 positives / 163 images; sensitivity **72.73%** (3 misses), specificity **98.68%**, AUROC **0.964**. | Very small positive support; interval uncertainty is substantial. |

### Central failure finding

The near-saturated Mendeley disease results are not trustworthy performance evidence. For DR, file metadata and coarse color statistics alone achieved AUROC **0.9995**. This is a direct warning that the compilation contains strong acquisition/source cues. The frozen Mendeley DR threshold also transferred to the previously exposed FDR26 test at sensitivity **80.30%** with **92 misses** and specificity **89.53%**; the frozen glaucoma cutoff transferred to JustRAIGS with **141 misses**. Repeated tuning on the Mendeley score cannot repair the missing source and patient provenance.

### Newly screened data sources

- The 2026 Figshare release contains 7,720 fundus-image IDs and vessel, artery/vein, and optic-disc/cup masks; it is useful for structural segmentation, but the described annotations are not disease-class labels. It was not downloaded for this disease-classification closeout. [Official Figshare record](https://figshare.com/articles/dataset/A_multi-disease_fundus_image_segmentation_dataset_with_pixel-level_structural_annotations/32646378).
- The 15-condition paper’s Hugging Face link resolves to a gated pretrained model card, not an image dataset. The linked card requires sharing the signed-in HF profile. No account details were shared and no gated file was accessed. The paper itself reports image-level upper-bound bias, a mean AUC decrease from 0.976 to 0.934 under pseudo-patient grouping, and only 68 cases in a preliminary single-site cohort. [Article](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2026.1778404/full) · [model card](https://huggingface.co/andromoon17/z-retina).

### Rare-tumor run archived, then removed from active scope

The UWF six-class run had already completed when closeout inspection began. It reported 97.91% image-split accuracy on 382 test images and macro-AUROC 0.9975, but the release has no patient IDs and documents repeat-eye images. The cloud-only perceptual audit found two zero-distance pHash cross-partition pairs and 21 train–test pairs at Hamming distance ≤2 (plus eight validation–test pairs). This result is retained only as a leakage-sensitive exploratory artifact; the user narrowed priorities to common primary diseases, so no further tumor training or interpretation was performed.

## Safe artifact handling

Twenty-eight allowlisted aggregate reports/logs were copied from the L4. The four UWF-tumor audit artifacts were individually matched to their remote SHA-256 checksums; the other copied reports have not yet had per-file checksum verification. No image files, model weights/checkpoints, per-image predictions, patient/eye manifests, split IDs, or raw hash-pair identities were pulled or committed. Those private/bulky files remain on the persistent cloud disk; stopping the VM does not delete that disk. The L4 is stopped after GitHub verification.
