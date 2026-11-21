# Golden Retina evaluation protocol

**Status:** prespecified project protocol; task-specific endpoints and label maps still require source-level verification.
**Use:** development and retrospective evaluation. This document does not establish clinical validation.

## 1. Freeze the question before training

Each experiment must identify one intended screening question, the supported image modality, target population/source, prediction unit (image, eye, or patient), target label and its authoritative definition, model/preprocessing version, split, primary metric, and threshold-selection rule. A broad label such as “any retinal disease” is not a valid endpoint unless a source supplies a compatible, consistently adjudicated label. Otherwise train and report disease-specific tasks separately.

Do not treat a raw sigmoid/softmax output as a calibrated disease risk. User interfaces may display a percentage only as a model score unless calibration and intended-use validation support a probabilistic interpretation.

## 2. Data and split controls

1. Keep all images, eyes, visits, and views from one patient in the same partition.
2. Check exact and near-duplicate image overlap before splitting. Preserve source identifiers and acquisition domains in the private manifest.
3. Record training/pretraining provenance where known. A nominally different test dataset is not independent if its images or patients overlap the model's training data.
4. Use training data for fitting, validation data for model selection and threshold choice, and a locked test set for one final evaluation. Do not revise the model, label mapping, or threshold after viewing locked-test results.
5. For a claimed cross-source transfer experiment, reserve the target source before selection. If the target source informed model choice or adaptation, describe the result as target-source development evidence.
6. Treat multi-view fusion as a paired analysis only when patient, eye, view identity, and acquisition correspondence are explicitly verified. Keep unpaired and missing views visible in the denominator.

## 3. Required task-specific metrics

| Task | Primary reporting | Additional reporting |
|---|---|---|
| Binary disease screening | AUROC and AUPRC with patient-cluster confidence intervals | Sensitivity, specificity, PPV, NPV, confusion counts, calibration, coverage, and a prespecified operating point |
| Ordinal grading | Quadratic-weighted kappa and ordinal error | Per-grade support, confusion matrix, one-vs-rest discrimination, and missing/uncertain labels |
| Multi-label disease/finding | Per-label AUROC/AUPRC and macro summary | Per-label support, threshold policy, calibration, and all-label regressions; do not hide rare-label failure in a macro average |
| Image quality / gradability | Per-class sensitivity/specificity and gradable coverage | Disease-task performance by quality stratum and abstention/reacquisition rate |
| Adaptive acquisition / paired views | Disease risk and sensitivity at each acquisition cost | Number of images/fields used, paired cohort size, missing-view behavior, and comparison with fixed-order baselines |

Report image-level, eye-level, and patient-level estimands separately when the source supports them. State the aggregation function (e.g., maximum or mean score) before evaluation; never switch it after seeing results.

## 4. Uncertainty and calibration

- Resample patients, retaining every sampled patient's eyes and images together. Use a fixed, recorded seed and enough bootstrap draws for stable intervals; report the interval method.
- For comparisons, use paired patient resampling on the same eligible patients and report the interval for the difference, not only separate model intervals.
- Report calibration only on data not used to fit the calibrator. At minimum provide Brier score and a reliability summary; avoid presenting ECE alone.
- For a safety-oriented sensitivity target, report the numerator, denominator, false-negative count, and confidence interval. Zero observed misses is not zero population risk.
- Mark estimates with inadequate patient/class support as exploratory or not estimable; do not manufacture a stable interval from a handful of cases.

## 5. Error and subgroup analysis

For each frozen test, retain private row-level predictions and audit false negatives, false positives, missing labels, image-quality failures, and model abstentions. Summarize by source/camera, disease severity, quality, and available demographic groups only when terms permit and support is adequate. Do not publish patient rows or identifying images. Track all prespecified tasks, including failures and regressions.

## 6. Reproducibility record

Each run card should record repository revision, source/version and terms reference, private manifest hash, patient split method/seed, label mapping, input preprocessing, model/checkpoint hash, software and accelerator versions, full training/evaluation arguments, threshold provenance, primary and secondary metrics, confidence-interval method, exclusions, runtime, and settled/estimated cloud spend. Keep raw manifests, row-level predictions, and restricted checkpoints in authorized storage; publish only reviewed, permitted artifacts and aggregate results.

## 7. Existing project tooling and remaining work

The checkout already contains patient-split builders, exact-duplicate audits, frozen-manifest verification, source-specific evaluation commands, patient-cluster bootstrap analyses, quality-stratified analyses, and locked-cohort preparation under `src/golden_retina/` and `tools/technical_dev/`. Their presence does not mean they have been rerun against the current VM data. Before a new phase is marked complete, reconcile the selected command with the current manifest and verify its input/output semantics on the authorized VM.

The immediate blocking work remains the Phase 2 cloud spend readout. Dataset staging, training, or new result claims must not be marked complete until that gate passes and the VM reports the required files and environment.
