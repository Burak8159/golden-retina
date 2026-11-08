# Block 22–23: frozen Zhejiang transfer and audit

**Status:** One-shot exploratory source-transfer inference completed with the frozen Block 21 candidate. This is not a referable-DR external validation or clinical validation.

## Locked evaluation

- Frozen ConvNeXt-Tiny mixed-view model: checkpoint SHA-256 `61f2e891a8d71e186876fa2dec437afa73b0970f33e92412850af7c24d19c926`.
- The checkpoint, preprocessing, and validation-derived operating thresholds were frozen before inference; no updates followed the pass.
- Figshare archive size was 9,018,811,972 bytes and MD5 was verified as `ddee42dc99792545654a65413c0fdb6f`; ZIP CRC validation completed during extraction.
- External manifest: 700 images, 602 patients, 689 eyes. There are 53 DR patients and 99 healthy patients, despite 100 images in each category. The diagnosis labels identify a source-defined disease category, not referable-DR grade.
- The dataset paper describes images as 3900×3072 Optos 200Tx UWF and quality labels as 0=poor / 1=good. Its quality annotation was reviewed by ophthalmologists; see the [dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11579006/).

## DR versus healthy ranking sensitivity

The frozen referable-DR risk score produced different estimates under different patient aggregation rules because positive and healthy patients have different numbers of images:

| Patient scoring rule | AUROC | Patient-bootstrap 95% interval | AUPRC |
|---|---:|---:|---:|
| Mean of available image scores per patient | 0.705 | 0.609–0.795 | 0.654 |
| One deterministic eye per patient | 0.793 | 0.705–0.871 | 0.767 |
| Maximum image score per patient | 0.881 | 0.818–0.937 | 0.852 |

The max rule is the frozen patient-level OR-style aggregation, but here it can benefit the DR group because 47 of 53 DR patients contribute both eyes while 98 of 99 healthy patients contribute one image. Therefore, the max-score AUROC is not a clean estimate of a single-image disease classifier. The mean and one-eye estimates are sensitivity analyses, not replacement claim targets. All results remain exploratory because source labels are DR diagnosis versus healthy, rather than referable-grade labels.

## Other source-defined categories

The same frozen score, trained for referable DR, yielded exploratory patient-mean AUROCs versus healthy of **0.334 AMD**, **0.683 pathologic myopia**, **0.917 retinal detachment**, **0.908 retinal vein occlusion**, and **0.695 uveitis**. These are not disease-specific model results or supported disease heads. The high RD/RVO rankings may reflect disease appearance, cohort composition, acquisition/domain cues, or combinations thereof; this study cannot isolate those mechanisms.

## Quality analysis

The Zhejiang paper defines overall quality 0 as poor and 1 as good. Against poor versus good overall quality, the model’s quality/gradability score had image-level AUROC 0.688 (AUPRC 0.378). Applying the frozen threshold selected on DDR gradability (0.2110) gave 28.2% sensitivity for poor images and 91.9% specificity for good images. This target differs from DDR grade-5 ungradability, so the result is exploratory. The risk-score AUROC for DR versus healthy was 0.755 among good-quality images, versus 0.777 using all image rows; RD and RVO remained about 0.904 and 0.906 in the good-quality subset. This does not prove the score is free of acquisition shortcuts; it only shows that the simple overall-quality label does not explain all observed ranking in this cohort.

## Interpretation boundary

- No threshold-based DR-vs-healthy accuracy, sensitivity, or specificity is reported: the frozen threshold is for referable-DR triage, and Zhejiang does not supply referable-grade labels.
- Calibration and predictive values are not interpreted because labels/targets differ and this is an enriched case-control-like cohort.
- No weight, threshold, or preprocessing change followed the one-shot output.
- Patient-level manifests, row-level predictions, and source images are excluded from the public repository. The frozen checkpoint was copied to a Git-ignored private local path; the VM copy is removed during cloud closeout.

## Artifacts

- `zhejiang_external_transfer_public_run_card.json`
- `zhejiang_disease_transfer_aggregate.csv`
- `zhejiang_transfer_sampling_quality.svg`
