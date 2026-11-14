# Golden Retina Multi-Disease Fundus Research System

## Technical system note

**Snapshot date:** 14 November 2026
**Evidence snapshot:** 14 November 2026, 10:31 UTC. Twelve disease/finding specialists have completed one held-out image-split test pass; CRS DINOv2 training is in progress; four low-support targets remain queued.
**Status:** Research and development. Not a clinically validated diagnostic system.

## 1. Executive summary

Golden Retina is a software research project that evaluates whether color fundus photographs can support separate, disease-specific image classifiers. This note describes the current RFMiD specialist sweep, its data contract, training and selection procedure, measured results, and the limits of those results. It is intended for machine-learning, biomedical-imaging, ophthalmology, and clinical-informatics researchers.

The current experiment trains a separate binary image classifier for each selected RFMiD label. It compares an ImageNet-initialized CNN or a fixed-revision DINOv2-S/14 model for each label, selects a candidate by validation AUROC, and evaluates only that selected candidate once on the dataset's official test split. Outputs are dataset-defined image labels; they are not diagnoses, disease staging, or patient-level risk estimates.

The central evidence constraint is the split unit: RFMiD supplies 3,200 images divided into 1,920 training, 640 validation, and 640 test images, but patient identifiers are not available in the staged labels. The official split therefore supports image-level held-out evaluation, not demonstrated patient-disjoint validation. It also does not test Golden Retina's intended camera, acquisition procedure, or clinical workflow.

## 2. System boundary and intended input

### 2.1 Image contract

- **Modality:** one color fundus photograph per model invocation.
- **View:** full-field image. This sweep does not infer a macula/disc crop or use paired views; RFMiD ROI coordinates and verified same-eye pairs are not part of this run.
- **Labels:** one binary RFMiD column at a time. Labels can co-occur across separate specialists; model scores are not a mutually exclusive diagnosis set.
- **Preprocessing:** RGB conversion, fixed square resize, and ImageNet channel normalization. Training-only augmentation comprises horizontal flip, rotation up to 8 degrees, and small brightness/contrast/saturation jitter.
- **Out of scope:** OCT, angiography, corneal topography, slit-lamp findings, visual fields, laboratory data, and clinical history. A fundus model cannot be assumed to detect conditions requiring those modalities.

### 2.2 Separation from other Golden Retina components

This specialist sweep is distinct from:

1. the earlier multi-label RFMiD ConvNeXt-Tiny checkpoint used by the local research interface;
2. the separate DR case-review and score-policy workflow;
3. historical HYGD glaucoma/GON internal-validation evidence; and
4. the media-haze (MH) head, which is treated as an image visibility/quality signal, not a disease classifier.

Numbers from those components must not be combined into one product-performance estimate. The specialist results below are a new, per-label experiment.

## 3. Dataset and label provenance

The run uses the `ctmedtech/RFMID` dataset snapshot at revision `16ee0cf528e9cb6ec38a6d7b1f3e4580172bc1da`, configured in [`configs/rfmid_expert_architecture_sweep_20261114.json`](../configs/rfmid_expert_architecture_sweep_20261114.json). The RFMiD data descriptor reports 3,200 color fundus images, three fundus-camera models, and 46 conditions/labels with expert-adjudicated annotations; the paper describes a 60/20/20 image split. The descriptor lists the dataset license as CC BY 4.0. Use of a hosted mirror or any derived artifact should still follow the terms applicable to that copy and its model weights. See the [RFMiD data descriptor](https://doi.org/10.3390/data6020014) and the [dataset configuration](../configs/rfmid_expert_architecture_sweep_20261114.json).

The project rechecked the official split manifests and exact image-content duplicates. The audit found no exact duplicate-content groups among the staged 3,200 images. This does not rule out near-duplicates, repeated visits, or multiple images from one person. Patient identifiers are absent, so patient overlap across image partitions cannot be audited. The current results are consequently described as **image-split test results**, never as independent patient-level or external validation.

`Disease_Risk` is excluded from the specialist disease heads. The active targets and positive-image support (train / validation / test) are:

| RFMiD label | Interpretation used here | Positive images, train / validation / test | Tier |
|---|---|---:|---|
| DR | Diabetic-retinopathy label | 376 / 132 / 124 | Primary support |
| ARMD | Age-related macular degeneration label | 100 / 38 / 31 | Primary support |
| DN | Drusen finding | 138 / 46 / 46 | Primary support |
| MYA | Myopic fundus appearance label | 101 / 34 / 32 | Primary support |
| BRVO | Branch retinal vein occlusion label | 73 / 23 / 23 | Primary support |
| ODC | Optic-disc cupping finding | 282 / 72 / 91 | Primary support |
| ODP | Optic-disc pallor finding | 65 / 26 / 24 | Primary support |
| ODE | Optic-disc edema finding | 58 / 21 / 17 | Primary support |
| TSLN | Retinal tessellation label | 186 / 65 / 53 | Primary support |
| LS | Retinal laser-scar finding | 47 / 17 / 15 | Primary support |
| CSR | Central serous retinopathy label | 37 / 11 / 13 | Primary support |
| RS | Retinitis finding label | 43 / 14 / 14 | Primary support |
| CRS | Chorioretinitis finding label | 32 / 11 / 11 | Primary support |
| CRVO | Central retinal vein occlusion label | 28 / 8 / 9 | Exploratory, low support |
| ERM | Epiretinal membrane appearance label | 14 / 7 / 5 | Exploratory, low support |
| MS | Macular-scar finding (not multiple sclerosis) | 15 / 5 / 7 | Exploratory, low support |
| RT | Retinal traction finding; OCT is preferred for extent | 14 / 6 / 5 | Exploratory, low support |

The four low-support labels are tracked separately because their small positive counts make both model selection and test estimates unstable. They are not evidence of deployable disease classifiers.

## 4. Candidate models and training procedure

### 4.1 Per-label architecture comparison

Each target has two configured candidates. Selection is based on **validation AUROC only**:

| Target group | Candidate architectures |
|---|---|
| DR, ARMD, DN, TSLN, LS, CSR, RS, CRS, ERM | ConvNeXt-Tiny and DINOv2 ViT-S/14 |
| MYA, BRVO, CRVO, MS | EfficientNet-B0 and ConvNeXt-Tiny |
| ODC, ODP, ODE, RT | ResNet-50 and ConvNeXt-Tiny |

The CNNs use the corresponding TorchVision default ImageNet weights. DINOv2 uses `facebook/dinov2-small` at revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`; its pooled class token feeds a newly initialized one-logit linear head. All model parameters, including encoder parameters, are passed to AdamW and are trainable during fine-tuning. This is not a frozen-feature-only probe.

### 4.2 Optimization and checkpoint selection

- **Output:** one scalar logit and sigmoid score per image and target.
- **Loss:** binary cross-entropy with logits and a positive-class weight equal to the square root of the negative/positive training ratio, clipped to `[1, 10]`.
- **Optimizer:** AdamW, learning rate `2e-5`, weight decay `2e-4`.
- **Maximum epochs:** 25. Training is also bounded to 420 seconds per candidate in the current sweep; early stopping occurs after five consecutive epochs without validation-AUROC improvement.
- **Checkpoint rule:** retain the epoch with the best validation AUROC. The architecture winner is the candidate with the larger validation AUROC for that target.
- **Test policy:** evaluate only the validation-selected checkpoint once on the official image test split. Test performance is not used to choose architectures, epochs, or thresholds.
- **Reported operating threshold:** fixed score threshold `0.5`; no clinical calibration or test-set threshold search is claimed.
- **Software/hardware:** the run card records Torch `2.9.1+cu129`, TorchVision `0.24.1+cu129`, and the L4 execution environment. Candidate checkpoints and per-run summaries contain the detailed provenance and SHA-256 values.

## 5. Snapshot results

The following values are for the single selected specialist per completed target, on the RFMiD official **image-level** test split, at threshold 0.5. `Misses` is the count of positive test images classified below threshold. AUROC and average precision use continuous sigmoid scores. No patient-cluster confidence intervals are available because patient IDs are absent.

| Target | Validation-selected model | Test positive images | Accuracy | Sensitivity | Specificity | AUROC | AP | Misses / positive |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DR | ConvNeXt-Tiny | 124 | 92.81% | 86.29% | 94.38% | 0.9754 | 0.9203 | 17 / 124 |
| ARMD | ConvNeXt-Tiny | 31 | 94.84% | 74.19% | 95.89% | 0.9538 | 0.6746 | 8 / 31 |
| DN | ConvNeXt-Tiny | 46 | 92.97% | 2.17% | 100.00% | 0.8109 | 0.2832 | 45 / 46 |
| MYA | ConvNeXt-Tiny | 32 | 97.19% | 100.00% | 97.04% | 0.9955 | 0.8910 | 0 / 32 |
| BRVO | ConvNeXt-Tiny | 23 | 95.63% | 78.26% | 96.27% | 0.9115 | 0.5931 | 5 / 23 |
| ODC | ConvNeXt-Tiny | 91 | 89.22% | 40.66% | 97.27% | 0.9162 | 0.6830 | 54 / 91 |
| ODP | ResNet-50 | 24 | 96.25% | 0.00% | 100.00% | 0.7587 | 0.1283 | 24 / 24 |
| ODE | ConvNeXt-Tiny | 17 | 97.81% | 82.35% | 98.23% | 0.9523 | 0.7877 | 3 / 17 |
| TSLN | ConvNeXt-Tiny | 53 | 92.81% | 58.49% | 95.91% | 0.9510 | 0.6034 | 22 / 53 |
| LS | ConvNeXt-Tiny | 15 | 98.12% | 33.33% | 99.68% | 0.9025 | 0.3541 | 10 / 15 |
| CSR | ConvNeXt-Tiny | 13 | 97.66% | 92.31% | 97.77% | 0.9942 | 0.8226 | 1 / 13 |
| RS | ConvNeXt-Tiny | 14 | 97.50% | 50.00% | 98.56% | 0.9623 | 0.5258 | 7 / 14 |

### 5.1 Interpretation

Accuracy alone is particularly misleading for rare findings. DN achieved 92.97% accuracy while missing 45 of 46 positive images; ODP achieved 96.25% accuracy while missing all 24 positive images. These are failure results, not successful screening performance. MYA's 0/32 misses and CSR's 1/13 misses are small image counts and do not establish a low clinical miss risk. The metrics also depend on the RFMiD image prevalence and do not estimate real-world predictive value.

The model sweep is still in progress. At this snapshot, CRS is training; the four low-support targets remain to be analyzed under their exploratory tier; the validation-only DR DINOv2 recovery, consolidated uncertainty review, model registry, packaging, artifact pull, safe GitHub publication, and Golden Retina cloud shutdown are not yet complete. Results must be refreshed from the final run summary before using this note as a final experiment report.

## 6. Existing non-sweep evidence (kept separate)

- **Older multi-label RFMiD interface model:** the repository README records a distinct 36-head ConvNeXt-Tiny run with macro AUROC 0.894 and a `Disease_Risk` operating point. It is not the set of specialist metrics above and must not be combined with them.
- **HYGD GON:** a historical RETFound-MAE ViT-L/16 internal-validation result on 125 images (95 positive, 30 negative) is recorded separately. No test rows were loaded for this snapshot; it is not an external validation.
- **MH:** the older multi-label baseline's media-haze result is an image visibility/quality signal, not a disease classifier.
- **70.13 score-policy replay:** this is a score-policy replay, not raw-image model accuracy.

## 7. Reproducibility and artifact map

| Artifact | Purpose |
|---|---|
| `configs/rfmid_expert_architecture_sweep_20261114.json` | Frozen target list, model candidates, split-support gates, and selection rule |
| `src/train_rfmid_disease_expert.py` | One-target training and single-checkpoint evaluation |
| `src/run_rfmid_expert_sweep.py` | Sequential per-target candidate comparison and test execution |
| `results/expert_sweep/sweep_summary.json` | Target-level candidate selection and test status |
| `results/expert_sweep/candidates/<target>/<architecture>/` | Per-candidate validation history, summary, and console log |
| `results/expert_sweep/final_test/<target>/` | One-pass selected-candidate test summary and log |
| `results/disease_expansion_20261114/status.json` | Tracker snapshot, support tiers, issues, and progress |

Images, model checkpoints, and any row-level outputs are not part of the public source release by default. A release must be limited to code, provenance, aggregate results, and artifacts whose licenses permit redistribution. The official data descriptor and the hosted mirror may have different applicable terms; verify before redistribution.

## 8. Limitations and required next validation

1. **Patient clustering:** RFMiD patient IDs are unavailable in the staged split, so patient-disjoint testing and patient-level uncertainty are not established.
2. **Domain transfer:** the test partition is from the same dataset source and its image acquisition distribution. The intended Golden Retina camera and 28D-lens workflow have not been validated.
3. **Label semantics:** findings are dataset annotations, not a complete clinical diagnosis. Several labels may represent appearance or a sign rather than a definitive disease.
4. **Class imbalance and support:** several positive test counts are small. Rare-label accuracy can be high while sensitivity is zero or near zero.
5. **Calibration and operating point:** the 0.5 threshold is a reporting convention for this development run, not a clinically selected threshold. Calibration, decision-curve utility, and operating-point confidence bounds have not been established.
6. **No prospective evidence:** no prospective clinical workflow, clinician-reader study, patient outcome, or clinical-benefit evaluation is reported.
7. **No deployment-security evaluation:** development UI/API code is not a production clinical service; security, identity, uptime, privacy, and governance require separate engineering and review.

The next confirmatory experiment should use a frozen model and preprocessing contract, patient-disjoint sampling, independently adjudicated reference labels, a source/camera domain representative of intended use, prespecified operating thresholds, subgroup analysis, and patient-level uncertainty. The reporting plan should be checked against the [CLAIM 2024 imaging-AI checklist](https://doi.org/10.1148/ryai.240300). These steps improve evaluation quality; they do not guarantee clinical performance or publication acceptance.

## 9. Key references

1. S. Pachade et al., “Retinal Fundus Multi-Disease Image Dataset (RFMiD): A Dataset for Multi-Disease Detection Research,” *Data*, 6(2), 14, 2021. [https://doi.org/10.3390/data6020014](https://doi.org/10.3390/data6020014).
2. J. Mongan et al., “Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update,” *Radiology: Artificial Intelligence*, 2024. [https://doi.org/10.1148/ryai.240300](https://doi.org/10.1148/ryai.240300).
3. DINOv2 model card and weights: [`facebook/dinov2-small`](https://huggingface.co/facebook/dinov2-small); this run pins revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`.
