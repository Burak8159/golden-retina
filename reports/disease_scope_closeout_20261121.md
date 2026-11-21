# Golden Retina disease-scope closeout — 2026-11-21

## Decision

This campaign closes the finite disease register currently tracked in the project. “Finished” means that each registered target has a recorded evidence disposition; it does not mean that every retinal disease has a validated classifier. A universal “all retinal disease” claim is not supported by the available datasets, labels, imaging modalities, or independent patient-level testing.

The evaluated input scope is color fundus photography / ultra-widefield fundus where the source permits it. OCT-only structural diagnoses, OCTA/angiography findings, corneal/topography disorders, and pediatric ROP are outside this image pipeline unless a compatible modality and an authorized, adequately labeled cohort are separately supplied.

## Disease-by-disease disposition

| Registered target | Evidence available in this campaign | Disposition |
|---|---|---|
| Diabetic retinopathy | Zhejiang internal patient-grouped test: 136 eyes / 120 patients; 94.7% sensitivity, 88.9% specificity, 1/19 positives missed. RFMiD also has exposed image-split evidence. | Research development only. Prior project exposure and single-source test prevent independent-validation claims; missed positives preclude safe-negative use. |
| Age-related macular degeneration | Zhejiang internal test: 94.1% sensitivity and specificity; 1/17 positives missed. RFMiD source label is a shared-model exploratory result. | Research development only; not an independently validated AMD diagnostic. |
| Glaucoma-suspect screening | RIM-ONE DL official hospital split: source-only random-init ResNet-18, 62.64% accuracy, 87.50% sensitivity, 50.85% specificity, AUROC 0.774; 7/56 positive images missed. Earlier protocol-deviation model opened the test first. | No-go for autonomous screening. Exposed exploratory benchmark, no patient IDs, low specificity and missed cases. |
| Pathological myopia | Zhejiang internal test: 85.0% sensitivity; 3/20 positives missed. | Research-only; not a safe negative screen. |
| Retinal detachment | Zhejiang internal test: 95.0% sensitivity; 1/20 positives missed. | Research-only; missed positives and no independent external validation. |
| Retinal vein occlusion | Zhejiang internal test: 95.0% sensitivity; 1/20 positives missed. RFMiD BRVO/CRVO heads are separate shared-model exploratory labels. | Research-only; source labels and disease subtypes must not be collapsed into a validated general RVO claim. |
| Uveitis | Zhejiang internal test: 35.0% sensitivity; 13/20 positives missed. | Explicit no-go for screening; current candidate misses most positive eyes. |
| Central serous retinopathy / central serous chorioretinopathy | RFMiD shared-model exploratory label: 84.6% sensitivity, 96.0% specificity, 2/13 positives missed, exposed image split without patient IDs. Mendeley V5 offers a potentially useful separate cohort, but download requires an authenticated public-dataset API path; no archive was staged or audited in this VM-only workflow. | No-go for a new specialist run in this campaign. Do not claim the candidate cohort was used. The RFMiD result remains exploratory, not independent validation. |
| Retinitis / chorioretinitis | RFMiD RS: 85.7% sensitivity, 2/14 missed; CRS: 90.9%, 1/11 missed; exposed image split. | Exploratory labels only; not standalone calibrated disease models. |
| Epiretinal membrane | RFMiD: 0/5 positive sensitivity; 5/5 missed. A separate public sample found during source review had positives only and could not supply negatives. | No-go. Current evidence is unusable for screening. |

## Other RFMiD source labels

The shared ConvNeXt-Tiny model’s 24 train-supported outputs include disease labels and image findings; they are not 24 specialist models. RFMiD’s image-level partition was already exposed in project work, and patient IDs are unavailable. Rare-label accuracy is especially misleading when almost all examples are negative. Per-label supports, thresholds, and misses are in `reports/remote_aggregates/rfmid_expanded10_v2_test_metrics.csv`; aggregate limitations are recorded in `reports/remote_aggregates/rfmid_expanded10_v2_test_report.md`.

## Claims and deployment boundary

- No model in this report is clinically validated, autonomous, or suitable for diagnosis, triage, reassurance, or treatment decisions.
- Scores are model outputs on source-specific research labels, not calibrated probabilities of disease.
- “100% specificity” on a tiny label sample does not establish a safe negative result.
- Every target without an adequate compatible cohort is marked outside scope / no-go, not silently treated as normal.
- The next confirmatory step requires a frozen candidate, prespecified threshold, and untouched patient-level cohort from an independent source, with reference-standard grading and uncertainty intervals.

## Data and source handling

RIM-ONE DL remains on the VM because its source terms prohibit redistribution. The Mendeley V5 CSC candidate was inspected at the source page only; it was not downloaded. No row-level predictions, patient manifests, private images, checkpoints, or credentials are included in the public repository.
