# Block 20: paired-view fusion versus macula-only control

**Status:** Internal development comparison complete. Neither arm is clinically validated.

## Design

The paired-view and macula-only arms used the same ConvNeXt-Tiny initialization, patient-level train/validation partitions, three-source manifest (DDR, DeepDRiD, DRTiD), seed (20261107), image size (320), optimizer settings, and 24-epoch/time budget. The paired arm consumed a second view when verified; the control consumed only the macula view. Final test inference was disabled in both arms. Checkpoints and row-level predictions remain private.

## Operating-point results on exposed internal validation

| Arm | Accuracy | Sensitivity | Specificity among gradable nonreferable | Missed referable / 436 | False referral / 369 | Ungradable routed to referral | Case-action coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Paired-view | 78.04% | 89.68% | 92.41% | 45 | 28 | 133/133 | 100% |
| Macula-only | 75.91% | 87.61% | 89.43% | 54 | 39 | 133/133 | 100% |

Thresholds and epochs were selected on this validation set. These are development estimates, not an independent test or evidence of zero-miss performance.

## Patient-clustered paired AUROC comparison

The prespecified primary endpoint is equal-weight macro patient-level AUROC over DRTiD and DeepDRiD. Two thousand paired patient bootstrap replicates, resampled within source, gave:

- Paired-view macro AUROC: **0.9585** (descriptive 95% percentile interval 0.9355–0.9761)
- Macula-only macro AUROC: **0.9345** (0.9023–0.9593)
- Paired-minus-control difference: **+0.0240** (0.0082–0.0419)
- DRTiD: AUROC 0.9376 vs 0.8958; AUPRC 0.9426 vs 0.9142 (paired vs control)
- DeepDRiD: AUROC 0.9794 vs 0.9733; AUPRC 0.9752 vs 0.9734

The bootstrap interval is descriptive and post-selection: validation data informed both checkpoint and operating-point selection. Although DDR is omitted from the primary comparison because its patient identities are not established and it contains single-view cases, checkpoint selection used validation results across the full source set, including DDR. This does not support confirmatory inference or clinical/external-validity claims.

## Decision

Advance the paired-view checkpoint to the already reserved one-shot source-level test because it is better on the prespecified internal macro-AUROC comparison. Freeze the artifact and its selected validation policy before any held-out inference. Do not tune after the held-out evaluation. The candidate’s validation policy still missed 45 of 436 referable cases, so it is not an autonomous zero-miss system.

## Reproducibility artifacts

- Bootstrap aggregate: `paired_view_vs_macula_validation_bootstrap.json`
- Paired checkpoint: `block20_paired/block20_paired.pt` (SHA-256 `61f2e891a8d71e186876fa2dec437afa73b0970f33e92412850af7c24d19c926`)
- Macula-only checkpoint: `block20_macula_only/block20_macula_only.pt` (SHA-256 `4a45cf1edbf608bd60a619a267896d0fe0b757e7d982ead776a1e016133b176e`)
- Paired run card: `block20_paired/run_card.json`
- Row-level predictions and weights are local/private and are not for public release.
