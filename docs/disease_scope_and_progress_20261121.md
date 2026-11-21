# Staged cohort and disease scope audit

Updated 2026-11-21. All raw images and row-level manifests remain on the Golden Retina L4 VM. This report contains aggregate counts only.

## End-state disease screening tracks

The proposed fundus-photo screening targets are separate disease-specific tasks: diabetic retinopathy (DR), age-related macular degeneration (AMD), glaucoma-suspect screening, pathological myopia, retinal detachment (RD), retinal vein occlusion (RVO), and uveitis. These are research screening/referral outputs, not clinical diagnoses. CSR, retinitis/chorioretinitis, and epiretinal membrane remain conditional exploratory targets until adequate patient-linked cohorts exist. Auxiliary morphology and treatment findings remain separate from diseases.

For the Zhejiang UWF task builder, each disease is compared with the dataset's Healthy category only. Other diseases are excluded from that binary task; they are not treated as healthy. The source labels are mutually exclusive categories and cannot model real-world comorbidity.

## Staged disease cohorts

| Source | Aggregate inventory | Existing patient split / label evidence | Current use boundary |
|---|---:|---|---|
| Zhejiang UWF | 700 images; 100 each for AMD, DR, Healthy, pathological myopia, RD, RVO, uveitis; 602 participants and 689 eyes | All 700 image references match; zero missing IDs, duplicate metadata IDs, cross-label patients, or exact duplicate image groups. Frozen deterministic 60/20/20 patient split. | Earlier DR-versus-Healthy outputs expose the DR cohort. This campaign ran a disease-versus-Healthy ConvNeXt probe and a separate three-encoder disease-versus-all validation comparison. One source only; no independent or clinical validation. |
| DeepDRiD | 1,377 images; 346 patient groups | Prior manifest: group-verified 1,124 train / 253 validation; zero patient overlap; no unused test partition in this development manifest. | Development only; prior model selection already used these train/validation cohorts. |
| DRTiD | 2,666 images representing 1,333 paired-eye groups | Prior manifest: group-verified 2,166 train / 500 validation; zero patient overlap. | Development only. The official test has previously been inspected by the project and cannot be presented as untouched confirmation. |
| HYGD | 626 images; 244 patient groups | Prior GON+ / GON− labels: 459 / 167. Group-verified 501 train / 125 validation, zero patient overlap; no unused test. | Glaucoma-suspect research only. GON labels do not establish a clinical glaucoma diagnosis. |
| DDR HF specificity subset | 3,002 images in the staged archive | Prior manifest marks patient grouping unverified for all images and uses the source as training only. | Training-only unless verified participant linkage is recovered; never a patient-independent test. |

All 7,671 rows in the prior combined development manifest resolve to image files. Its exact SHA-256 comparison found zero duplicate-content groups and zero cross-source exact duplicates. This does not rule out near-duplicates or shared participants across datasets.

The project owner reports that dataset owners approved the research use. The audit workspace does not yet contain the source-specific approval messages or the exact original release/terms documents for every archive. This is recorded as owner-reported permission, not independently verified legal advice; do not redistribute raw data or row-level outputs.

## Zhejiang patient split and exposure boundary

The 700-row manifest is grouped by patient, all eyes from a patient stay in one partition, and train/validation/test have zero patient overlap. Per-diagnosis patient counts (train / validation / test) are: AMD 31/10/10; DR 32/11/10; Healthy 59/20/20; pathological myopia 59/20/20; RD 60/20/20; RVO 60/20/20; uveitis 60/20/20. The manifest SHA-256 is `a1ea32ef2ba70422e72bcff7caff81e99c130d74615771f5f4bb7acb69372659` and stays on the VM.

An older project analysis used Zhejiang DR and Healthy rows and generated DR scores for 200 images from 152 patients. Therefore, the new patient-grouped split is leakage-safe but the dataset is not a project-wide untouched cohort for DR. For other categories, current campaign results remain internal single-source development evidence: the initial disease-versus-Healthy ConvNeXt test probe and subsequent disease-versus-all validation-only comparisons. They are not independent external validation.

## Historical metrics (not current campaign results)

Prior RFMiD figures in the tracker are image-level development outputs with patient independence unavailable. They remain separate from all current work. Examples that explain scope decisions: drusen had 45/46 positives missed; optic-disc cupping 54/91 missed; generic myopic appearance is not pathological myopia; historical BRVO is not equivalent to aggregate RVO; and sparse ERM support had 5/5 misses. High accuracy under class imbalance is not evidence of useful disease sensitivity.

## Current campaign state

The cohort audit and evaluation infrastructure are complete. Separate disease-vs-Healthy manifests and the first frozen ConvNeXt-Tiny probes have been run. A DINOv2 ViT-B/14 frozen-feature comparison has also been run on training and validation only; its validation thresholding is optimistic and candidate-selection-only. A FLAIR comparison is currently underway. The initial ConvNeXt test partition has already been scored once and is reserved for descriptive reporting only; no follow-on model selection or threshold tuning may use it. See [`disease_by_disease_overview_20261121.md`](disease_by_disease_overview_20261121.md) for the per-disease decision and foundation-encoder comparison. No result is independent clinical or external validation; scores are not calibrated risks or diagnoses.
