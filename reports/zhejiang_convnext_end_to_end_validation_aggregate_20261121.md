# Zhejiang ConvNeXt-Tiny partial fine-tuning: validation aggregate

**Run date:** 2026-11-21
**Status:** completed development run; aggregate only is checked into the local workspace.
**Encoder:** ImageNet-1K ConvNeXt-Tiny; feature blocks 6–7 and the classifier were trainable.
**Task:** six independent one-vs-rest sigmoid heads (AMD, diabetic retinopathy, pathological myopia, retinal detachment, retinal vein occlusion, uveitis); the source labels are mutually exclusive, so comorbidity was not assessed.
**Split:** Zhejiang UWF patient-grouped split; 137 validation eyes. Training and validation only were loaded; the test rows were filtered before image paths were resolved. The checkpoint was warm-started from the saved epoch-11 state; optimizer and AMP scaler state were reset. Best validation loss was at epoch 30 (0.17971); final training epoch was 32 after the patience rule.

## Validation results

Thresholds were selected on this same validation set to maximize sensitivity subject to specificity ≥90%. Consequently, these operating-point estimates are optimistic development results. Accuracy is descriptive and dominated by negatives. Values below are not independent, external, or clinical performance.

| Disease task | Positive / all validation eyes | Sensitivity (patient-bootstrap 95% CI) | Misses | Specificity | Accuracy | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| AMD | 16 / 137 | 93.75% (81.25–100%) | 1 | 94.21% | 94.16% | 0.951 | 0.766 |
| Diabetic retinopathy | 21 / 137 | 90.48% (71.43–100%) | 2 | 95.69% | 94.89% | 0.966 | 0.833 |
| Pathological myopia | 20 / 137 | 85.00% (70.00–100%) | 3 | 90.60% | 89.78% | 0.949 | 0.862 |
| Retinal detachment | 20 / 137 | 90.00% (75.00–100%) | 2 | 94.02% | 93.43% | 0.963 | 0.908 |
| Retinal vein occlusion | 20 / 137 | 100.00% (100–100% bootstrap; 83.2% exact lower bound) | 0 | 90.60% | 91.97% | 0.991 | 0.960 |
| Uveitis | 20 / 137 | 40.00% (20.00–60.00%) | 12 | 93.16% | 85.40% | 0.838 | 0.498 |

Patient-cluster intervals used 2,000 positive/negative patient-stratified bootstrap replicates, with both eyes retained together; they are conditional on the full-validation threshold. For RVO the bootstrap interval is degenerate because none of 20 observed positives were missed. The exact two-sided 95% binomial lower confidence bound for 20/20 successes is 83.2% (20 positive eyes correspond to 20 distinct patients here), illustrating why zero observed misses is not a zero-risk guarantee.

## Interpretation and limits

- This run lowers validation loss relative to the saved epoch-11 state (0.21235 to 0.17971 at the selected checkpoint), but it does not uniformly improve operating-point sensitivity over the already selected frozen-feature heads. For example, the frozen-feature C sweep had one DR validation miss versus two here, and six uveitis misses versus twelve here.
- The cohort is small and comes from one source. The same validation set controls training early stopping, threshold selection, and prior candidate selection. The reported sensitivity and specificity therefore do not establish out-of-source performance.
- Uveitis is unusable as a reassuring negative screen at this operating point. Pathological myopia also misses 3/20 positives. Even zero validation misses for RVO is not proof of zero risk with only 20 positives.
- The Zhejiang DR task has project-level prior exposure and is not a fresh confirmatory cohort. The full project test partition remains out of bounds for this run.
- Checkpoint SHA-256: `af95b8bfb9de42b41a20357f44f7a5545f610a9f22bfbd0431fe10bb64d7caef`. Model weights and row-level outputs remain private on the VM; the public/local report contains only aggregate metrics.
