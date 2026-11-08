# Stage 33 — Patient-level uncertainty and error report

**Scope:** synthesis of fixed development runs and one previously exposed Zhejiang exploratory transfer. No model/threshold selection was performed on Zhejiang.

## Main DR development comparison

Patient/source-grouped validation comprises 314 groups (150 positive) at a fixed score threshold of 0.5. The RETFound-Green source-invariance candidate had accuracy 89.81%, sensitivity 90.00%, specificity 89.63%, AUROC 0.9491, and AUPRC 0.9546. Against the matched non-quality control, paired bootstrap deltas were +8.92 percentage points accuracy (95% CI +5.10 to +13.06), −4.67 pp sensitivity (−8.23 to −1.48), +21.34 pp specificity (+15.29 to +28.00), +0.04 pp AUROC (−0.61 to +0.70), and +0.09 pp AUPRC (−0.55 to +0.73).

The apparent fixed-threshold gain is not stable across training seeds. Seed 2 yielded 82.17% accuracy, 92.67% sensitivity, 72.56% specificity, AUROC 0.9468, and AUPRC 0.9543. Relative to seed 1, specificity fell 17.07 pp (paired 95% CI −22.88 to −11.46), while AUROC and AUPRC changes were not distinguishable from zero. Therefore, the seed-1 89.63% specificity must not be presented as a robust operating point.

The auxiliary quality objective did not show a reliable gain: paired intervals included zero for accuracy, sensitivity, specificity, AUROC, and AUPRC. Quality-head performance could not be evaluated because the validation split had no quality labels.

## Exploratory Zhejiang transfer

On the previously exposed Zhejiang subset, source-defined DR diagnosis versus healthy control, the frozen candidate ranked 53 DR and 99 healthy patients (152 total; 200 images) with patient AUROC 0.9897 (stratified bootstrap 95% CI 0.9773–0.9981) and AUPRC 0.9826 (0.9629–0.9967). No binary threshold was applied. This endpoint is not referable-DR grade >=2 and cannot validate a referral operating point. The cohort is neither untouched nor independent.

## Interpretation and failure boundaries

- The source-adversarial objective changed the operating-point tradeoff but did not improve ranking metrics over the matched control.
- A high result from one seed did not reproduce; report seed variability and do not lock the favorable threshold result as general performance.
- The Zhejiang ranking result is exploratory and endpoint-mismatched; it is not clinical validation.
- The supported disease tasks remain DR and HYGD glaucoma only. No evidence supports AMD, RVO, or broad “any retinal disease” claims.
- No independent cohort, clinical workflow study, calibration certification, or autonomous diagnosis certification is established by these analyses.

## Source artifacts

- `results/technical_dev_20261108_rerun/source_invariance_vs_control_patient_paired.json`
- `results/technical_dev_20261108_rerun/retfound_green_source_invariance_02/source_invariance_seed_stability_patient.json`
- `results/technical_dev_20261108_rerun/retfound_green_quality_ablation_patient_summary.json`
- `results/technical_dev_20261108_rerun/zhejiang_retfound_green_source_invariance_summary.json`
