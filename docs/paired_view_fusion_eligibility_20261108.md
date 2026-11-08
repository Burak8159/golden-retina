# Paired-view fusion eligibility audit (2026-11-08)

## Decision

Do not run paired-view fusion in this development sequence. The frozen manifest does not contain laterality or an eye identifier, so its repeated patient-level images cannot be verified as same-eye pairs. Treating them as paired views could fuse opposite eyes or different fields and would not test the stated same-eye complementary-view hypothesis.

## Evidence inspected

The frozen train/validation manifest is `results/technical_dev_20261108_rerun/train_valid_manifest.csv` (SHA-256 `d92debe39fab68f66d553c0215124f0da4ffca6a474760ebd7abfa354ea9468c`). Its columns are `image_path`, `source`, `split`, `patient_id`, `patient_group_verified`, `image_id`, `dr_grade`, `ungradable`, `lesions`, `gon`, and `image_sha256`; no laterality/eye/view mapping field is present.

There are multiple images per verified patient in DeepDRiD and DRTiD, but the manifest alone establishes patient grouping, not same-eye identity or view complementarity. Therefore image count per patient is insufficient evidence for fusion eligibility.

## Planned substitute

Use the stage-30 slot for a second-seed replication of the leading candidate, with identical source manifest, preprocessing, architecture, loss, and validation protocol. Compare patient-grouped performance and uncertainty against the first seed. If no reproducible gain appears, report the null result and do not tune further against the same validation set.

This is an internal development decision; no held-out test data are used.
