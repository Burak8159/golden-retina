# RETFound-Green exploratory transfer to Zhejiang UWF

**Status:** exploratory retrospective source-transfer analysis; not independent validation.

## Frozen evaluation

- Model: RETFound-Green adapted candidate, seed 20261111, best epoch 8; checkpoint SHA-256 `de221fc19b6e76fede71593d92363bac8d814ea3ef287b2cf586d9f97602e176`.
- Dataset: previously exposed Zhejiang single-center Optos UWF cohort. The original archive checksum matched the recorded Figshare dataset file MD5 `ddee42dc99792545654a65413c0fdb6f`.
- Evaluated subset: 200 images from 152 patients: 53 source-labeled DR and 99 healthy patients.
- Endpoint: source-defined DR diagnosis versus healthy. This is not the model's referable-DR (grade >=2) endpoint.
- Aggregation: maximum referable-head score over the available images for each patient.
- Threshold: none; no binary classification or referral operating point was evaluated.

## Results

- Patient AUROC: **0.9897** (stratified patient-bootstrap 95% CI **0.9773–0.9981**).
- Patient AUPRC: **0.9826** (stratified patient-bootstrap 95% CI **0.9629–0.9967**).
- Bootstrap replicates: 5,000; random seed 20261113.

The point estimates indicate strong ranking on this selected, previously exposed cohort. They do not demonstrate calibrated probabilities, referable-DR sensitivity/specificity, an autonomous operating point, generalization to new institutions, or clinical utility. The maximum-over-images aggregation also means the estimand is patient-level ranking with available image count, not per-image diagnosis.

## Artifacts and privacy

Aggregate result: `results/technical_dev_20261108_rerun/zhejiang_retfound_green_source_invariance_summary.json` (SHA-256 `0c62371fa8c41a6c2753e149a2599e8a41a8b9a10a99256331495f742a97e522`). Row-level predictions remain private on the research VM and are not included in this repository. RETFound-Green terms permit this internal non-commercial research use as scoped in the project record; no redistribution or deployment rights are inferred.
