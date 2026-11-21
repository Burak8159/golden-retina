# Pre-closeout evidence ledger

**Campaign:** Golden Retina multi-disease development, 2026-11-21
**Purpose:** make the evidence unit, selection exposure, and permitted interpretation explicit before final VM artifact collection.
**Overall boundary:** this is a retrospective model-development and software audit. No result below is independent clinical validation or evidence of clinical utility.

## Model and dataset evidence

| Track | Input and labels | Partition/exposure | Current evidence | Permitted interpretation |
|---|---|---|---|---|
| Zhejiang UWF six disease heads | Color ultra-wide-field fundus; six mutually exclusive source labels: AMD, DR, pathological myopia, retinal detachment, RVO, uveitis | Patient-grouped 60/20/20; 700 images / 602 patients; all current runs trained and selected on train/validation. 137 validation eyes; thresholds selected on those same validation eyes. DR source had earlier project exposure. | Partial ConvNeXt-Tiny fine-tune: AMD 15/16 detected, DR 19/21, PM 17/20, RD 18/20, RVO 20/20, uveitis 8/20 at validation-selected specificity target >=90%. | Internal, single-source development estimates only. Do not claim a confirmed threshold or zero miss risk. Uveitis and PM are poor negative screens at this operating point. The single-label setup does not test comorbidity. |
| Zhejiang frozen encoders | Same Zhejiang six tasks | Same validation cohort; encoder and regularization comparisons reused the validation set. | Earlier candidates sometimes had better disease-specific sensitivity than the shared fine-tune; no single encoder/fine-tune dominates. | Candidate-selection evidence only; repeated validation exposure makes estimates optimistic. |
| HYGD GON task | Fundus image; dataset GON+/GON− glaucoma-suspect label | 501 train images / 125 validation images; 43 validation patients. This validation population has prior project development exposure. | RETFound-MAE fixed 0.5: 96.8% accuracy, 95.8% sensitivity, 100% specificity, AUROC .991; DINOv2, VisionFM and FLAIR comparisons saved. | Research classification of the dataset's GON label only. Not definitive glaucoma diagnosis, not independent confirmation. |
| RFMiD ONNX heads | Fundus images; 17 condition-label and image-finding heads | Legacy image-level split; patient independence unverified. Registry stores positive image counts and test accuracy at 0.5, but no patient-grouped evidence. | Four post-hoc shortlist heads (DR, AMD, myopia, BRVO) plus exploratory disease/finding heads; models are downloadable and locally executed. | Exposed development artifacts only. Do not label raw model scores as patient risk or clinical probabilities. Findings such as drusen/cupping are not diagnoses. |
| DeepDRiD paired-view policy | First-view and paired macula/disc score predictions; calibration and validation patient folds | Saved cross-fit predictions. View-suffix mapping is not verified; historical test rows exist in files and are excluded from decisions. | Inferred mapping validation: 0/18 positive patients missed, 32.6% specificity, 83.3% second-view acquisition; exact one-sided 95% upper miss-risk bound 15.3%. Swapped mapping: 0/18, 26.1% specificity, 94.9% acquisition. | Low-efficiency research policy; not useful or risk-certified. Mapping sensitivity prevents promotion. |
| Functional vision, quality and lesion detection | Acuity/contrast/color/field/fixation; image gradability; retinal lesions | No calibrated device test or verified quality/lesion validation join in this campaign. | Explicit unavailable states; no new supported model. MMRDR lesion archive is not staged. | Do not emit measurements, gradability, or lesion risk from the current disease heads. |
| Eye localization | Synthetic rendered eye/face images | Synthetic-only training/evaluation; no real human, IR, goggle, or motor-loop validation. | Small ONNX segmentation model is packaged; outputs are an unvalidated synthetic-domain localization result. | Do not call a real eye tracker or use it to control motors. |

## Selection, uncertainty, and exposure

- Zhejiang validation data drove model family/comparator selection, threshold selection, and early stopping. Report its numbers as validation development results. The patient-grouped test partition was filtered from this campaign's image loading, but the DR task has earlier project exposure; no current run is a sealed project-level confirmatory test.
- Small positive denominators materially limit conclusions. For example, 20/20 RVO detections still have an 83.2% exact two-sided 95% binomial lower bound on sensitivity. A bootstrap interval degenerating at 100% does not remove that limitation.
- HYGD and RFMiD evidence has its own reuse/split limitations; do not combine their percentages with Zhejiang to make a pooled performance claim.
- CCG validation has only 18 positive patients per mapping branch. Zero observed misses does not certify a low miss rate; the one-sided 95% exact upper bound is 15.3%.
- No calibration cohort independent of model development, external-site confirmation, prospective workflow evidence, or device performance study is documented here.

## Reproducibility and distribution state

- Local aggregate reports and JSONs are indexed with file size and SHA-256 in [`precloseout_artifact_index_20261121.json`](precloseout_artifact_index_20261121.json). Recompute this index after final VM pull if any aggregate changes.
- Exact training checkpoints and row-level predictions remain in private VM storage unless individually approved for transfer. Raw images and row-level predictions are excluded from GitHub publication.
- The repository owner reports source-use permission. Preserve that owner-provided basis, attribution and source restrictions; this ledger is not independent legal verification of each dataset term.
- The mBRSET access decision and dataset grant are not established by this campaign record. Do not stage mBRSET until user reports its arrival and official access prerequisites are confirmed.
- Model registry weights are shared model artifacts; publish only after confirming their upstream license and attribution requirements. A public URL or a research-only project does not itself authorize redistribution.

## Closeout decisions

1. Preserve the per-disease reports and unfavorable results as written; do not replace them with a single pooled “accuracy.”
2. Keep uveitis and pathological myopia out of any reassuring-negative product workflow; keep RVO's zero observed misses in its small validation set explicitly uncertain.
3. Keep the paired-view policy at no-go until view mapping is verified and efficiency/risk targets are met on new patients.
4. Continue with the frozen evidence record; do not retune against these validation outcomes or use historical test rows to pick a candidate.
5. Pull and checksum remaining VM artifacts, then publish only permitted code/docs/aggregates and shut down Golden Retina cloud resources before the mobile workstream.
