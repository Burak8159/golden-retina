# Golden Retina latest model state (2026-11-10)

This note identifies the latest locally available disease-classification and eye-finding artifacts before the current GitHub update. These are research/development results. They do not establish clinical performance.

## Retinal disease classification

The latest available adaptation is **RETFound-MAE ViT-L/16**, with the last transformer block, final normalization layer, and four task heads trainable; the rest of the backbone was frozen. Training used a combined DR-grade, referable-DR, masked glaucoma, and masked image-quality objective. The image-quality task has no validation labels in this split, so its performance is not estimable.

The checkpoint was selected at epoch 11 on validation AUROC. On the internal train/validation-only manifest (6,793 train rows; 878 validation rows; zero patient overlap; no test rows loaded), the recorded epoch-11 validation results are:

| Task/stratum | Validation result |
|---|---:|
| Referable DR, threshold 0.5 | accuracy 79.02%; sensitivity 71.86%; specificity 84.73%; AUROC 0.8721; AUPRC 0.8679 (n=753, 334 positive) |
| Five-grade DR | accuracy 57.37%; balanced accuracy 42.89%; macro-F1 0.4259 (n=753) |
| HYGD glaucoma | accuracy 96.00%; sensitivity 94.74%; specificity 100%; AUROC 0.9779; AUPRC 0.9943 (n=125, 95 positive) |
| DDR image quality | Not estimable: zero validation labels |
| DeepDRiD referable DR subgroup | accuracy 82.21%; AUROC 0.8914; sensitivity 79.55%; specificity 83.64% (n=253) |
| DRTiD paired referable DR subgroup | accuracy 77.40%; AUROC 0.8621; sensitivity 69.11%; specificity 85.43% (n=500) |

The 96% glaucoma figure is based on only 125 validation images and is not an external estimate. The combined validation cohort is not an independent test set. No test rows were loaded, no external clinical validation was performed, and this result does not support autonomous diagnosis or deployment claims. The result summary is `results/technical_dev_20261108_rerun/retfound_mae_adaptation/summary.json`; manifest SHA-256 `d92debe39fab68f66d553c0215124f0da4ffca6a474760ebd7abfa354ea9468c`.

The RETFound-MAE checkpoint was not available in the local release tree, and the RETFound-Green candidate weights are not redistributed here. The RETFound-Green research comparator from the prior frozen run had seed-1 internal validation accuracy 89.81%, sensitivity 90.00%, specificity 89.63%, AUROC 0.9491, and AUPRC 0.9546 at threshold 0.5. Seed 2 reduced specificity to 72.56%, so the operating point was not reproducible. These are exposed internal validation results, not independent external validation. See `docs/stage36_closeout_20261108.md` and `results/technical_dev_20261108_rerun/retfound_green_research_model_freeze.json`.

## Eye finding and tracking

The latest frozen, locally available eye model is a compact shared-eye `FaceEyeUNet` selected at epoch 24. Its small checkpoint is now included as `models/eye_finding/charmorph_shared_eye_v2b.pt` (SHA-256 `57ef89060125078d9aea7f3f8a21b380a269b3f34b20ef89d7ed32872262c2ab`). The run card is `results/eye_finding_ai/charmorph_training_v2/frozen_candidate_run_card.json`.

On 24 fresh renderer-seed frames from the same synthetic base identity, macro Dice was 0.8542 (iris 0.8086; pupil 0.8997), with pupil presence in 24/24 frames on each image side. Mean pupil-center error was 0.434 and 0.539 pixels at 384-pixel model input. On a 32-frame synthetic translated-eye sequence, detections were present in 32/32 frames per side; alpha-beta filtering yielded mean center errors of 1.195 and 1.036 pixels. The sequence used synthetic translation, not physical eyeball rotation or camera motion.

This is a synthetic engineering prototype evaluated from one base identity. It has not been evaluated on people, real eye-camera footage, or deployment hardware. The metrics must not be described as real-world tracking accuracy or coverage. The separate full-face renderer and tracking source is under `eye_finding_ai/realistic_3d_v3/`; earlier model experiments are retained as code but are not the current frozen candidate.

## Release boundary

This GitHub update includes the eye-finding source and its small synthetic-trained checkpoint, plus the latest disease-model source changes and aggregate run summary. It excludes raw datasets, row-level predictions/manifests, rendered training bundles, and disease-model checkpoints that are missing or not cleared for redistribution. Excluded artifacts remain local and are not required to interpret the reported aggregate metrics above.
