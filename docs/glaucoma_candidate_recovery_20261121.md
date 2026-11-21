# Glaucoma research restart: dataset and evaluation plan

Updated 2026-11-21. This is a research-only, image-based glaucoma classification benchmark; it is not a clinical diagnosis or deployment claim.

## Decision

Use the **RIM-ONE DL hospital-partition variant** for a standalone glaucoma-versus-normal fundus benchmark. The source repository describes 485 photographs (313 normal, 172 glaucoma), with images acquired at three hospitals and an official hospital-based training/test variant. The held-out hospitals are the main transfer test. We will not mix HYGD, RFMiD, PAPILA, or other retinal images into this source-specific benchmark or tune on its test labels.

The official repository states that the data may be used free of charge for research and education, while copying/redistribution is prohibited; it also asks researchers to use the published partitions and not add images from other databases when reporting this benchmark. Therefore raw data, image-level predictions, and checkpoints remain private to the VM/workspace. Publish only code and non-identifying aggregates if compatible with the source terms.

## Why this is the first route

- The task label is explicitly glaucoma versus normal, unlike RFMiD optic-disc cupping, which is a finding and not a glaucoma diagnosis.
- It provides a hospital-separated split, so we can test acquisition-site transfer without repurposing the already exposed HYGD internal validation.
- It is directly downloadable from the source repository and requires no new credentialing step.

PAPILA is a useful alternative because it contains both eyes and patient-level clinical labels for 244 patients, but the data-release license is not consistently represented across available catalogs. It is not selected for this run until the release record’s exact terms are verified. The paper itself is CC BY 4.0, but that does not automatically settle rights in the separately released archive.

## Execution gates

1. The archive was downloaded directly to the existing Golden Retina L4 VM. It is 222,561,134 bytes, SHA-256 `85aed5f95c794f52d11b6ed953032108a66576ffcff4b587365df467491b603d`. Only the hospital partition was extracted.
2. Audit verified 485 readable images: training hospitals 195 normal/116 glaucoma; held-out hospitals 118 normal/56 glaucoma. No exact duplicate-content groups were found across partitions. One perceptual-hash candidate had Hamming distance 4, pixel MAE 27.203/255 and correlation 0.9578 after resizing; it is not an exact duplicate and was retained. Patient IDs are absent, so patient separation is unverified.
3. A deterministic stratified 80/20 image-level train/validation split was created only inside the official training hospitals (seed 8159). The official test was initially sealed, then opened by the first exploratory run.
4. The first exploratory ConvNeXt-Tiny run used ImageNet initialization. Because the repository instructs users to train/tune only with RIM-ONE images, this run is recorded as a protocol-deviation exploration and is not the source-only result. A second ResNet-18 run used random initialization and only the RIM-ONE hospital training images.
5. The operating point was selected on the training-hospital validation split to target at least 90% sensitivity while maximizing specificity. On the hospital partition, the source-only run achieved 87.5% sensitivity and missed 7/56 glaucoma images. Since the test had already been opened in the first run, neither result is an untouched confirmatory evaluation.
6. Keep this benchmark separate from HYGD and RFMiD. Neither result supports autonomous diagnosis. Hospital-held-out performance does not establish patient-level independence, broad glaucoma performance, prospective clinical validity, or safety.

## Audit and run artifacts

- VM-only archive: `/data/golden-retina/workspace/external/RIM-ONE-DL-20261121/RIM-ONE_DL_images.zip`.
- VM-only row manifest, private split manifests, per-image scores and checkpoints; none are redistributed. The aggregate split audit and metrics are in `reports/remote_aggregates/rimone_glaucoma_20261121.json`.
- Source-only trainer: `src/golden_retina/train_rimone_glaucoma_scratch.py`; aggregate report: `reports/glaucoma_rimone_hospital_20261121.md`.
- A temporary Cloud NAT retrieved the official archive and a pretrained-weight candidate. The NAT and router were deleted after transfer; no public inbound address was assigned to the VM.

## Source references

- RIM-ONE DL official repository and terms: https://github.com/miag-ull/rim-one-dl
- Dataset paper: Fumero Batista et al., “RIM-ONE DL: A Unified Retinal Image Database for Assessing Glaucoma Using Deep Learning,” Image Analysis & Stereology, 2020. DOI: https://doi.org/10.5566/ias.2346
- PAPILA description and patient-level label structure: https://doi.org/10.1038/s41597-022-01388-1
