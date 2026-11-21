# Golden Retina disease-by-disease scope and evidence

**Updated:** 2026-11-21
**Purpose:** research planning and transparent progress record; this is separate from the live project tracker.
**Current evidence class:** exploratory, single-source development evidence. No result below is clinical validation or a calibrated probability of disease.

## What the system is intended to assess

The current image pathway is color fundus photography, including ultra-widefield images where the source provides them. Each supported target is a distinct research classification task. A camera-only fundus image cannot replace slit-lamp examination, OCT, angiography, corneal topography, visual fields, or other specialized tests. “Not assessed” must remain distinct from “negative.”

The current Zhejiang experiments use one-vs-Healthy labels. Other disease categories are excluded from each task, rather than relabeled as healthy. The source categories are mutually exclusive and do not represent the comorbidity encountered in screening populations.

The first formulation was disease-versus-Healthy only, so it left competing diagnoses out of the negative class. The follow-up comparison uses disease-versus-all-other-labels on the same patient-grouped train/validation split and does not load the already-scored test partition.

## Disease-by-disease status

| Target | Observable with the planned fundus image? | Data and current evidence | Current decision / next work |
|---|---|---|---|
| **Diabetic retinopathy (DR)** | Yes; retinal lesions can be visible in fundus images. | Zhejiang UWF ConvNeXt-Tiny probe: 39 test eyes, 19 DR and 20 Healthy; 94.9% accuracy, 100% sensitivity (0/19 misses), 90% specificity, AUROC 0.995. Patient-cluster sensitivity interval is uninformative at this sample size. The cohort was previously used in project analysis. | Keep as a supported research track, but mark this result **exposed** and single-source. Compare encoders on train/validation only. Seek a genuinely unused, patient-linked source before any confirmatory claim. |
| **Age-related macular degeneration (AMD)** | Yes; macular changes can be visible, though disease stage and reference grading matter. | Zhejiang UWF ConvNeXt-Tiny probe: 37 test eyes, 17 AMD and 20 Healthy; 94.6% accuracy, 88.2% sensitivity (2/17 misses), 100% specificity, AUROC 0.997. Sensitivity 95% patient-cluster interval: 70.6–100%. | Retain as a research track. Small one-source estimate; validate with stage-aware labels and an independent patient cohort. |
| **Glaucoma / glaucoma-suspect finding** | Optic-disc appearance is visible, but diagnosis requires clinical context and often fields/OCT; fundus image alone supports only a suspect/referral task. | HYGD has 626 images / 244 groups with GON+ and GON− labels, but its patient groups were already used for development and no unused patient test partition is staged. No new glaucoma result in the current run. | Keep only as **glaucoma-suspect screening**, never a definitive glaucoma diagnosis. New independent patient-level evaluation is needed. |
| **Pathological myopia** | Posterior-pole and fundus changes may be visible; source label definitions must distinguish pathological myopia from ordinary myopic appearance. | Zhejiang UWF ConvNeXt-Tiny probe: 40 test eyes, 20 PM and 20 Healthy; 97.5% accuracy, 95.0% sensitivity (1/20 miss), 100% specificity, AUROC 1.000. Sensitivity interval: 85–100%. | Retain as a research track; confirm label criteria, severity coverage, and transfer on another source. |
| **Retinal detachment (RD)** | Some detachments are visible on fundus images, but peripheral extent and image protocol can limit detection; this cannot rule out RD. | Zhejiang UWF ConvNeXt-Tiny probe: 40 test eyes, 20 RD and 20 Healthy; 80.0% accuracy, 65.0% sensitivity (7/20 misses), 95.0% specificity, AUROC 0.943. Sensitivity interval: 45–85%. | **Current probe is inadequate.** Do not use as a reassuring negative. Investigate labeling, field coverage and training/validation errors; compare models on validation only. Escalation/referral wording must be conservative. |
| **Retinal vein occlusion (RVO)** | Often has visible retinal signs in fundus images. | Zhejiang UWF ConvNeXt-Tiny probe: 40 test eyes, 20 RVO and 20 Healthy; 90.0% accuracy, 95.0% sensitivity (1/20 miss), 85.0% specificity, AUROC 0.978. Source label is aggregate RVO, without BRVO/CRVO separation. | Retain as aggregate RVO research track only. Do not claim subtype performance. Need external patient-level replication and calibrated operating point. |
| **Uveitis** | Some posterior-segment inflammatory findings may be visible, but uveitis is heterogeneous and anterior disease may not be visible in a fundus photograph. | Zhejiang UWF ConvNeXt-Tiny probe: 40 test eyes, 20 uveitis and 20 Healthy; 52.5% accuracy, 25.0% sensitivity (15/20 misses), 80.0% specificity, AUROC 0.630. Sensitivity interval: 10–45%. | **Current probe fails.** The broad disease label may not match fundus-visible disease. Inspect source label definition and restrict any future task to a clearly visible phenotype; otherwise exclude from fundus-only claims. |

### Conditional targets, pending adequate patient-linked evidence

- **Central serous chorioretinopathy (CSC/CSR):** may show fundus changes, but OCT is central to confirmation. A fundus classifier could only be an explicitly limited referral flag.
- **Retinitis / chorioretinitis:** potentially visible when posterior lesions are present; etiologies are heterogeneous and image labels need clinical specificity.
- **Epiretinal membrane (ERM):** subtle cases are better characterized by OCT; retain only if the source has reliable fundus labels and adequate patient-level support.
- **Macular edema, retinal artery occlusion, retinal tears, optic-disc edema, and other urgent findings:** clinically important, but do not add a classifier unless a compatible image modality, reliable labels, patient linkage, and a defensible evaluation set are available.

### Excluded from the current fundus-only disease classifier

Keratoconus requires corneal imaging/topography; many corneal and anterior-segment disorders require slit-lamp or anterior-segment imaging. Refractive error, nephropathy, and systemic conditions cannot be diagnosed from a fundus photo alone. OCT-defined conditions cannot be treated as equivalent to fundus-image labels. These may belong in future device modules only if the acquisition hardware and reference standard are added.

## Model and evaluation record

The first current-campaign model is a frozen ImageNet ConvNeXt-Tiny encoder with separate patient-weighted L2 logistic heads. Each binary task is fit on the training patients; the operating threshold is selected using validation data to maximize sensitivity subject to at least 90% specificity. Eye scores are max-pooled across images. Uncertainty is estimated with patient-cluster bootstrap. The exact aggregate table and intervals are in [`zhejiang_convnext_linear_probe_20261121_aggregate.md`](../reports/zhejiang_convnext_linear_probe_20261121_aggregate.md).

The reported held-out partition has now been examined by this first probe. It must not be used to select later model families, thresholds, or training variants. Later comparisons use training/validation only; report the initial test result as an exploratory estimate and preserve the exposure warning for DR. Scores are uncalibrated model scores, not percentages of disease risk.

### Foundation-encoder comparison on validation (candidate selection only)

A frozen official DINOv2 ViT-B/14 CLS embedding probe was subsequently fit on training patients and evaluated on the same validation patients. The threshold was selected from eye-pooled validation scores to target at least 90% specificity. Because the same small validation set is used to compare candidates and choose thresholds, thresholded values below are optimistic and must not be reported as final performance. No test rows were loaded by this run.

| Target | Validation eyes | Accuracy at validation-selected threshold | Sensitivity | Specificity | AUROC | AUROC change vs frozen ConvNeXt-Tiny validation |
|---|---:|---:|---:|---:|---:|---:|
| AMD | 36 | 100.0% | 100.0% | 100.0% | 1.000 | +0.009 |
| DR* | 41 | 97.6% | 100.0% | 95.0% | 0.998 | +0.012 |
| Pathological myopia | 40 | 97.5% | 95.0% | 100.0% | 0.962 | −0.028 |
| Retinal detachment | 40 | 97.5% | 95.0% | 100.0% | 0.992 | +0.065 |
| Aggregate RVO | 40 | 97.5% | 95.0% | 100.0% | 0.975 | −0.022 |
| Uveitis | 40 | 75.0% | 60.0% | 90.0% | 0.800 | +0.135 |

*DR remains exposed at source level from earlier project analysis. The comparison is internal and does not restore independence. Validation AUROC improved for AMD, DR, RD, and uveitis relative to the frozen ConvNeXt probe, while PM and RVO were numerically lower. These small same-source differences are candidate-selection signals only; no inferential claim is justified. RD and uveitis remain weak enough that they are not suitable for a negative reassurance decision. The full aggregate, checkpoint hash, and selection caveat are in the VM-only `zhejiang_dinov2_validation_only_v2_20261121/validation_only_summary.json`.

### FLAIR validation-only comparison

The cached official FLAIR encoder was also evaluated with a frozen 512-dimensional image representation and patient/class-balanced logistic heads. It uses the same train and validation patients, but its official 512-pixel preprocessing differs from the 224-pixel ConvNeXt and DINOv2 inputs. Thresholded numbers are again optimistic validation selection estimates; compare these as exploratory encoder signals, not final performance.

| Target | Accuracy | Sensitivity | Specificity | AUROC | AUPRC | Misses / positive validation eyes |
|---|---:|---:|---:|---:|---:|---:|
| AMD | 100.0% | 100.0% | 100.0% | 1.000 | 1.000 | 0 / 16 |
| DR* | 100.0% | 100.0% | 100.0% | 1.000 | 1.000 | 0 / 21 |
| Pathological myopia | 92.5% | 95.0% | 90.0% | 0.963 | 0.969 | 1 / 20 |
| Retinal detachment | 82.5% | 70.0% | 95.0% | 0.905 | 0.907 | 6 / 20 |
| Aggregate RVO | 90.0% | 85.0% | 95.0% | 0.935 | 0.955 | 3 / 20 |
| Uveitis | 65.0% | 35.0% | 95.0% | 0.750 | 0.741 | 13 / 20 |

FLAIR did not resolve the RD or uveitis sensitivity problem. Its very high AMD/DR validation values are particularly prone to selection optimism on these small cohorts. FLAIR aggregate output remains on the VM at `zhejiang_flair_validation_20261121/validation_only_summary.json`; no row-level predictions or model heads were copied locally.

### Disease-versus-all candidate results

Three encoders were compared on the same task definition: frozen ImageNet ConvNeXt-Tiny, frozen DINOv2 ViT-B/14, and frozen FLAIR. Each task used one source disease label as positive and all six remaining labels, including Healthy, as negative. The frozen patient-grouped development split contributed 425 training images and 137 validation images from 121 patients. The test partition was never loaded by this comparison.

| Disease | Validation positives / eyes | ConvNeXt sensitivity (misses) | DINOv2 sensitivity (misses) | FLAIR sensitivity (misses) | Strongest candidate signal from AUROC/AUPRC |
|---|---:|---:|---:|---:|---|
| AMD | 16 / 137 | 93.8% (1) | 93.8% (1) | 93.8% (1) | DINOv2/FLAIR; AUROC 0.966/0.971, AUPRC 0.849/0.851 |
| DR* | 21 / 137 | 90.5% (2) | 85.7% (3) | 90.5% (2) | FLAIR; AUROC 0.973, AUPRC 0.846; cohort exposed at source level |
| Pathological myopia | 20 / 137 | 90.0% (2) | 90.0% (2) | 85.0% (3) | ConvNeXt/DINOv2; AUROC 0.962/0.953, AUPRC 0.874/0.870 |
| Retinal detachment | 20 / 137 | 85.0% (3) | 85.0% (3) | 50.0% (10) | DINOv2; AUROC 0.943, AUPRC 0.881 |
| RVO | 20 / 137 | 90.0% (2) | 75.0% (5) | 65.0% (7) | ConvNeXt; AUROC 0.977, AUPRC 0.889 |
| Uveitis | 20 / 137 | 40.0% (12) | 55.0% (9) | 45.0% (11) | DINOv2 highest but still poor; AUROC 0.866, AUPRC 0.640 |

Sensitivities are measured at thresholds selected on this same validation set to maximize sensitivity at specificity ≥90%, so they are optimistic. Accuracy is not the main selection criterion because negatives dominate (137 total eyes versus 16–21 positive eyes). AUROC/AUPRC are also same-validation candidate-selection estimates, not generalization estimates.

**Provisional decision:** continue with disease-specific candidates, not one universal encoder. ConvNeXt is a development candidate for DR/RVO; DINOv2 is promising for AMD/RD and is the least poor uveitis candidate; PM remains unresolved between ConvNeXt and DINOv2. These are validation-only choices. RD and especially uveitis remain unsuitable for a reassuring negative result. Full aggregates are in [`../reports/zhejiang_one_vs_rest_validation_aggregate_20261121.md`](../reports/zhejiang_one_vs_rest_validation_aggregate_20261121.md).

### Per-disease error review

Reviewing which *other source labels* generated the most false positives helps prioritize the next label and quality audit. Counts below are from the same small, repeatedly used validation cohort and its selected thresholds. They are not explanations of model behavior and do not identify image-level cases.

| Disease head | Main competing-label false positives (ConvNeXt / DINOv2 / FLAIR) | Development implication |
|---|---|---|
| AMD | DR 7/21 / 9/21 / 10/21 | Check shared appearance and label boundaries between DR and AMD; do not infer a biological shortcut from this count alone. |
| DR | AMD 4/16 / 5/16 / AMD 1/16 (FLAIR also RD 1/20) | Audit AMD/DR label definitions and errors; DR cohort is source-exposed. |
| Pathological myopia | RVO 4/20 / 4/20 / 7/20 | RVO is a recurrent confounder; verify PM phenotype criteria and image quality. |
| Retinal detachment | Healthy 4/20 and uveitis 4/20 / uveitis 3/20 / uveitis 4/20 | Pair this with the poor RD sensitivity; no negative reassurance. |
| RVO | Uveitis 5/20 / 5/20 / PM 3/20 | Resolve cross-label errors and retain the disease-specific head only as a development candidate. |
| Uveitis | Healthy 7/20 / 9/20 / RD 5/20 and Healthy 4/20 | Both misses and false alarms are substantial; the broad source label may not match fundus-visible disease. |

The completed aggregate error review and candidate metrics are recorded in [`../reports/zhejiang_one_vs_rest_validation_aggregate_20261121.md`](../reports/zhejiang_one_vs_rest_validation_aggregate_20261121.md). Any follow-on adaptation must use training rows for fitting and validation rows for model selection; no tuning may use the previously examined test partition.

### Regularization sensitivity and current development candidates

A four-value logistic-head regularization sweep (`C` = 0.03, 0.1, 1, 10) was run on frozen ConvNeXt, DINOv2 and FLAIR image features. For each value, its threshold was reselected on the same validation cohort to maximize sensitivity at specificity of at least 90%; the C choice was then made from that validation cohort as well. This is exploratory and multiply selected, so it overstates likely performance.

| Disease | Current validation-selected development head | Sensitivity / misses | Specificity | C=1 sensitivity for that encoder | Decision |
|---|---|---:|---:|---:|---|
| AMD | DINOv2, C=10 | 93.8% (1/16) | 95.9% | 93.8% (1/16) | Retain as a candidate; sensitivity did not improve. |
| DR* | ConvNeXt, C=10 | 95.2% (1/21) | 93.1% | 90.5% (2/21) | One fewer validation miss; source-level exposure remains. |
| Pathological myopia | DINOv2, C=0.03 | 95.0% (1/20) | 90.6% | 90.0% (2/20) | One fewer validation miss; phenotype boundary still matters. |
| Retinal detachment | DINOv2, C=0.1 | 85.0% (3/20) | 94.9% | 85.0% (3/20) | No sensitivity gain; fails a reassuring-negative standard. |
| RVO | ConvNeXt, C=0.1 | 95.0% (1/20) | 90.6% | 90.0% (2/20) | One fewer validation miss; aggregate RVO only. |
| Uveitis | DINOv2, C=0.03 | 70.0% (6/20) | 94.0% | 55.0% (9/20) | Three fewer validation misses, but still inadequate for reassurance. |

These are not independently confirmed gains: both C and operating threshold were chosen using the same small validation sample. The encoders remain frozen; this is not end-to-end fine-tuning. Selected private heads remain on the L4 VM. Full AUROC/AUPRC values and artifact boundaries are in [`../reports/zhejiang_one_vs_rest_regularization_sweep_aggregate_20261121.md`](../reports/zhejiang_one_vs_rest_regularization_sweep_aggregate_20261121.md).

## Evidence and rights boundaries

The disease and control counts, split details, model outputs, and bootstrap summaries are described at aggregate level. Raw images, row-level manifests, and per-eye prediction files remain on the L4 VM. Dataset owners' permission was reported by the project owner; source-specific approval records and original terms have not been independently archived in this workspace. Do not redistribute source data or row-level outputs based on this overview.

## Sources and working files

- Zhejiang UWF inventory and split audit: [`disease_scope_and_progress_20261121.md`](disease_scope_and_progress_20261121.md).
- Current aggregate probe: [`../reports/zhejiang_convnext_linear_probe_20261121_aggregate.md`](../reports/zhejiang_convnext_linear_probe_20261121_aggregate.md).
- This document is not a product claim, clinical protocol, or substitute for clinician review.

### End-to-end ConvNeXt-Tiny fine-tuning (validation-only)

A patient-balanced partial fine-tune warm-started from the saved epoch-11 checkpoint and selected epoch 30 by validation loss (0.17971; final epoch 32). The six independent disease-vs-all heads were thresholded on the same 137-eye validation cohort to target at least 90% specificity. This is a development estimate, with same-validation early stopping, threshold selection, and candidate selection; it is not an external test. The full table and checkpoint hash are in [`../reports/zhejiang_convnext_end_to_end_validation_aggregate_20261121.md`](../reports/zhejiang_convnext_end_to_end_validation_aggregate_20261121.md).

| Disease | Validation sensitivity | Misses / positives | Specificity | Development interpretation |
|---|---:|---:|---:|---|
| AMD | 93.75% | 1 / 16 | 94.21% | Candidate signal only; small positive set. |
| Diabetic retinopathy* | 90.48% | 2 / 21 | 95.69% | Two misses; selected frozen-feature candidate previously had one miss at a different operating point. |
| Pathological myopia | 85.00% | 3 / 20 | 90.60% | Not suitable for a reassuring negative. |
| Retinal detachment | 90.00% | 2 / 20 | 94.02% | Small validation set; cannot rule out disease. |
| Retinal vein occlusion | 100.00% | 0 / 20 | 90.60% | Zero misses in 20 validation positives is not a zero-risk guarantee. |
| Uveitis | 40.00% | 12 / 20 | 93.16% | Fails as a reassuring-negative task. |

The fine-tune lowered validation loss from 0.21235 at the saved epoch-11 state to 0.17971 at the selected checkpoint, but did **not** improve every disease’s operating-point sensitivity; uveitis was substantially worse than its frozen-feature development candidate. Do not replace the disease-specific frozen-feature candidate list with this single shared fine-tune. Model weights and per-eye outputs remain private on the VM. *DR remains exposed at source level from prior project analysis.

### Glaucoma-suspect track (HYGD)

The best preserved frozen-feature candidate is RETFound-MAE ViT-L/16 on the previously used HYGD group-verified validation partition: accuracy 96.8%, sensitivity 95.8%, specificity 100%, AUROC 0.991 and AUPRC 0.998 at a fixed 0.5 cutoff (125 images, 95 GON+; 43 validation patients). DINOv2, VisionFM and FLAIR are lower-AUROC alternatives in the same development set. A prior RETFound partial adaptation reached 96.0% accuracy and AUROC 0.978, below the frozen encoder's AUROC. The head predicts HYGD's GON+/GON− label, not a definitive glaucoma diagnosis. Validation patients have already been used in project development, so these are not independent results. The comparison and provenance are in [`../reports/amd_glaucoma_suspect_development_comparison_20261121.md`](../reports/amd_glaucoma_suspect_development_comparison_20261121.md). Further retuning this split would not add confirmatory evidence; an unused patient-level cohort is the next meaningful validation step.
