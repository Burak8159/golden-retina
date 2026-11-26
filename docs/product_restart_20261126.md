# Golden Retina ML restart — 26 November 2026

## End goal

Build a reproducible **fundus-image ML research product** using only the home RTX 3050 Ti for training. Mobile app work is deferred to a separate request. No Google Cloud GPU is part of this campaign. Earlier cloud/model work is retained as historical evidence and does not count as progress in the restart.

The finished ML package should provide disease-specific models with explicit inputs, source label definitions, preprocessing, model/version provenance, score semantics, model cards and a local inference interface. It is a research product, not autonomous clinical diagnosis, calibrated risk, a medical device, validated goggles, or a safe negative screen.

## Disease groups and decisions

| Group | Disease task | Current decision |
|---|---|---|
| Retinal neurovascular/metabolic | Diabetic retinopathy: grade 0–4; separately define referable DR as grade ≥2 only when the source supports that mapping | **First priority.** RTX has MMRDR CFP images/labels. The staged 8,893-row export has no patient or eye IDs and no test-prefixed rows. It can support development, but not patient-independent validation. |
| Macula | Age-related macular degeneration | Keep a separate AMD task. Existing RFMiD head is research-only. No fresh patient/source-linked AMD cohort is currently confirmed on the RTX. |
| Optic nerve | Glaucoma-suspect screening | Only proceed if source labels define glaucoma/suspect status and provide a suitable reference standard. Cupping, pallor and edema are image findings, not glaucoma diagnoses. No installable glaucoma model or fresh eligible cohort is currently confirmed. |
| Retinal vascular occlusion | Branch RVO and central RVO as separate tasks | Existing RFMiD heads have limited image-level support. Keep exploratory until a new patient/source-held-out cohort is available. |

Myopia and RFMiD finding labels are outside this restart’s primary scope. Cataract, keratoconus, corneal disease, OCT-only disease and an “any retinal disease” output are excluded from this fundus-only effort unless data establish a distinct, valid fundus task. Image quality and physical eye tracking are also out of scope. Synthetic eye localization is not a real-camera tracker.

## Success criteria

- Every disease is evaluated separately; label definitions match the source and intended task.
- Patient/eye groups stay together across train, validation and test. A source-held-out cohort is preferred for transfer.
- A locked test set is used once, after model and thresholds are frozen. Previously exposed partitions are development data, not fresh confirmation.
- Report per-disease AUROC, AUPRC, sensitivity, specificity, accuracy, miss counts, calibration and patient-cluster confidence intervals. Accuracy is secondary to sensitivity/misses at a prespecified operating point for referral screening.
- A disease with inadequate labels, source/patient groups or test support remains unsupported. Do not replace missing evidence with an image-level random split and call it external validation.
- Model artifacts have exact preprocessing and output contracts, hashes, provenance and reproducible run arguments. Raw scores are never called calibrated probabilities or clinical risk without calibration evidence.

## Ten phases

| Phase | Completion evidence |
|---|---|
| 1. Reconstruct ML workspace | Reconcile training/inference code and model artifacts; verify RTX; create a separate Golden Retina worker directory; inventory images, labels, grouping and storage. |
| 2. Freeze disease tasks | Record label meanings, modality, comparison population, source/rights basis, group identifiers, exclusions and output semantics for DR, AMD, glaucoma-suspect, BRVO and CRVO. |
| 3. Audit duplicates and freeze eligible splits | Check exact/perceptual duplicates and source/patient/eye overlap. Freeze manifests before candidate selection. If only image-level grouping is possible, mark the run development-only. |
| 4. Reproduce baselines | Record exact environment, command, data manifest/hash, logs, metrics and checkpoint hash. Failed reproduction is recorded as failed, not approximated. |
| 5. Compare model families | Compare ConvNeXt-Tiny baseline with resource-feasible, permitted foundation encoders (for example DINOv2 or RETFound when weights and terms are available). Select on validation only. |
| 6. Train DR | First train a grade-aware and referable-DR baseline on eligible MMRDR CFP development data on RTX; checkpoint/resume and keep row-level outputs private. The staged export cannot support patient-independent claims. |
| 7. Train other disease tasks | Train AMD, glaucoma-suspect, BRVO and CRVO separately only after each has adequate labeled grouped data. Do not equate image findings to a disease. |
| 8. Locked disease evaluation | Apply each frozen candidate once to an eligible patient/source-held-out cohort; report uncertainty, misses, calibration and errors. No eligible cohort means no external result. |
| 9. Package the ML product | Export selected models to ONNX where appropriate; verify source/export parity, preprocessing, versions, size, runtime, hashes and local inference interface. |
| 10. Evidence and closeout | Produce disease run cards and product instructions; pull/checksum worker artifacts; publish only allowed code and aggregates; verify no training process remains. Mobile integration is deferred. |

## Verified starting evidence and current gaps

- Live SSH check: NVIDIA RTX 3050 Ti Laptop GPU, 4,096 MiB, 9 MiB used and 0% utilization. No training run had started at that check.
- An isolated Golden Retina directory exists on the RTX; the legacy ActiveRetina directory was left untouched.
- MMRDR CFP inventory: 8,893 rows, grades 0–4, all filenames have the training prefix, 8,893 matching image files, no duplicate basenames, no patient/eye ID columns, and no candidate paired filename stems. This is not proof that patients are unique. Aggregate audit: reports/product_restart_20261126/phase1_mmrdr_cfp_inventory.json; SHA-256 was checked against the RTX copy.
- The local registry has 18 ONNX files (one synthetic eye-localization model and 17 RFMiD disease/finding specialists). Every file’s SHA-256 matches the active manifest. Both phone codebases contain an ONNX inference path; mobile work is deferred.
- Existing RFMiD measurements are image-split development evidence with prior project exposure and unverified patient independence. They are not new restart results.
- Current macOS developer selection is Command Line Tools; xcodebuild/simctl are unavailable through it. This is parked with mobile work.
- No cloud GPU has been launched in this restart. Retained cloud storage and billing have not been re-audited here.

## Operating rules

- RTX only; no Google Cloud compute.
- Keep patient images, row-level labels/predictions and private checkpoints on the worker unless their terms explicitly permit export.
- No candidate or threshold selection on the locked test.
- Keep disease labels and image findings separate.
- If the data do not support a patient/source-separated estimate, downgrade the claim and continue only with eligible tasks.
