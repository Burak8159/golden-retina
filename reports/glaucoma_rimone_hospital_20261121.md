# Glaucoma recovery: RIM-ONE DL hospital split

**Status:** completed as exploratory development evidence. This is not an autonomous diagnosis, clinical validation, or a deployable glaucoma risk model.

## Data and source protocol

We used only the official RIM-ONE DL hospital-partition images for the source-only run: HUC training hospitals and HUMS/HCSC held-out hospitals, as specified by the dataset authors. The official repository describes 313 normal and 172 glaucoma images across three Spanish hospitals, instructs users to preserve its partitions, and says not to add images from other databases to training or tuning. It permits research/education use but prohibits copying and redistribution. The dataset paper says two experts assessed the images. ([official repository and data-use instructions](https://github.com/miag-ull/rim-one-dl), [dataset paper](https://doi.org/10.5566/ias.2346))

The VM archive SHA-256 is `85aed5f95c794f52d11b6ed953032108a66576ffcff4b587365df467491b603d`. The official hospital partition contained 311 training-hospital images (195 normal, 116 glaucoma) and 174 held-out-hospital images (118 normal, 56 glaucoma). All 485 decoded; exact cross-partition duplicates: 0. One dHash candidate was inspected (distance 4, resized pixel MAE 27.203/255, correlation 0.9578); it was not an exact duplicate and was retained. Patient identifiers are absent, so patient-level separation is unverified.

## Models and results

The internal validation subset was a stratified image-level 80/20 split of the training-hospital images (seed 8159). The candidate threshold was selected on this validation subset to target at least 90% sensitivity while maximizing specificity. At the time of this run, the official hospital test was already exposed by the first exploratory model; therefore neither model result below is an untouched confirmatory result.

| Run | Held-out hospital accuracy | Sensitivity | Specificity | AUROC | AP | False negatives |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-18, random initialization, trained only on RIM-ONE training images | 62.64% | 87.50% | 50.85% | 0.774 | 0.617 | 7/56 |
| ConvNeXt-Tiny, ImageNet initialization, protocol-deviation exploration | 78.16% | 83.93% | 75.42% | 0.899 | 0.788 | 9/56 |

The scratch model reached validation AUROC 0.929 and validation sensitivity 91.3%; that operating point did not transfer to the held-out hospitals. The pretrained run is recorded for audit only: ImageNet initialization means it does not satisfy a strict reading of the repository's source-only training/tuning instruction, so it must not be presented as the comparable source-only result. Its image-level AUROC bootstrap 95% interval was 0.849–0.942; this does not account for patient clustering.

## Decision

**No-go for an autonomous or clinical glaucoma classifier.** Neither exploratory run met the 90% sensitivity target on the hospital partition; the source-only model missed 7/56 glaucoma images. This is evidence of a validation-to-hospital transfer gap, not a threshold to tune on the test. Do not adapt the threshold using these hospital labels or ship either checkpoint in the consumer app. A fresh, properly governed external cohort with patient identifiers and a prespecified locked model is needed for confirmatory work.

The row-level manifest, predictions, checkpoint, and dataset remain on the VM under the source's non-redistribution terms. The public report contains aggregate metrics only. A temporary Cloud NAT used for download was deleted after transfer.

## Reproduction contract

- Trainer: `src/golden_retina/train_rimone_glaucoma_scratch.py`.
- Run configuration: `configs/rimone_glaucoma_hospital_20261121.json`.
- Data audit: `reports/remote_aggregates/rimone_glaucoma_20261121.json`.
- The private split manifest, per-image predictions, and checkpoint are intentionally omitted from GitHub. The official dataset download and its own license terms govern access.
