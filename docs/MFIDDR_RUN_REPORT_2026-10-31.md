# MFIDDR L4 training and held-out evaluation

**Run date:** 2026-10-31  
**Status:** completed retrospective, patient-level benchmark evaluation  
**Primary objective:** exceed 90% patient-level referable-DR accuracy on the official MFIDDR test split; report sensitivity and misses separately.

## Dataset and split

MFIDDR provides four fundus fields per eye and DR grades 0–4. A patient is positive when either eye has a grade of at least 2. Splits were patient-disjoint: 2,770 training patients, 488 validation patients (95 positive), and 1,083 official-test patients (203 positive). The test set had zero patient overlap with train/validation. The test archive SHA-256 was `e7da12475f4e56902c5d3fd02dd55c423611e060e69c87220089d7e21320973a`.

The model is a fully fine-tuned pretrained ConvNeXt-Tiny (`convnext_tiny.fb_in22k_ft_in1k`) at 512-pixel input size. The paired model aggregates up to four fields per eye and then the two eyes at patient level. `F1`–`F4` are treated as dataset field codes; their clinical anatomy mapping was not independently verified. MFIDDR supplies no grade-5/ungradable cases, so this evaluation does not establish performance on ungradable images.

## Development runs

Thresholds and checkpoints were selected on validation patients only. No official-test result was used to tune model weights or thresholds.

| Candidate | Validation operating point | Result |
|---|---|---|
| F1-only ConvNeXt-Tiny, 8 epochs | Maximum accuracy; epoch 6 | 96.31% accuracy, 87.37% sensitivity, 98.47% specificity; 12/95 positives missed |
| Four-view ConvNeXt-Tiny, 12 epochs | Maximum accuracy; epoch 11 | 97.13% accuracy, 90.53% sensitivity, 98.73% specificity; 9/95 missed |
| Hard-boundary warm-start | Pivoted after 6 epochs when it did not improve the selected operating point | Best remained its epoch-0 starting point: 91.60% accuracy, 96.84% sensitivity; 3/95 missed |
| Natural-distribution ordinal fine-tune, 10 completed epochs | Maximum sensitivity subject to at least 90% validation accuracy; best epoch 1 | 95.08% accuracy, 97.89% sensitivity, 94.40% specificity; 2/95 missed |

The natural-distribution run also had a separate validation threshold with zero misses: 80.74% accuracy and 76.08% specificity. This shows the validation trade-off rather than a zero-miss, high-accuracy result.

## Official-test results for the frozen natural-distribution checkpoint

The evaluation used the natural-distribution checkpoint selected by the validation sensitivity-at-90%-accuracy objective. Thresholds below were selected on validation patients and then applied unchanged to the official test patients. Each patient receives a binary action, so action coverage is 100% in this retrospective evaluation.

| Views | Validation-selected policy | Accuracy (95% patient-bootstrap CI) | Sensitivity (95% CI) | Specificity (95% CI) | Missed positives | AUROC / AP |
|---:|---|---:|---:|---:|---:|---:|
| 1 (F1) | Maximum validation accuracy | 95.29% (94.00–96.49) | 89.16% (84.73–93.60) | 96.70% (95.45–97.84) | 22/203 | 0.9841 / 0.9469 |
| 1 (F1) | Maximum sensitivity with validation accuracy floor of 90% | 89.20% (87.35–90.95) | 96.55% (93.60–99.01) | 87.50% (85.23–89.66) | 7/203 | 0.9841 / 0.9469 |
| 1 (F1) | Validation zero-miss-specificity policy | 72.95% (70.45–75.44) | 99.51% (98.52–100) | 66.82% (63.86–69.77) | 1/203 | 0.9841 / 0.9469 |
| 4 (F1–F4) | Maximum validation accuracy | **96.31% (95.11–97.32)** | 86.21% (81.28–90.64) | 98.64% (97.84–99.32) | 28/203 | 0.9931 / 0.9716 |
| 4 (F1–F4) | Maximum sensitivity with validation accuracy floor of 90% | **94.74% (93.44–96.03)** | **98.03% (96.06–99.51)** | **93.98% (92.39–95.45)** | **4/203** | 0.9931 / 0.9716 |
| 4 (F1–F4) | Validation zero-miss-specificity policy | 77.38% (74.98–79.78) | 100% (100–100) | 72.16% (69.20–75.11) | 0/203 | 0.9931 / 0.9716 |

The four-view model exceeds 90% held-out accuracy. At the accuracy-maximizing threshold it misses 28 positive patients; at the high-sensitivity threshold it retains 94.74% accuracy and misses 4; the zero-miss threshold lowers accuracy to 77.38%. Thus this run **does not** demonstrate both 90%+ accuracy and zero misses at once. With zero misses among 203 positive test patients, the one-sided exact 95% upper missed-risk bound is 1.46%, not zero risk.

Patient-bootstrap intervals use 2,000 stratified resamples conditional on the frozen checkpoint and validation-selected threshold. The threshold-selection and model-selection process itself is not included in these intervals. Sensitivity and specificity intervals were recomputed from the saved patient-level actions because an initial evaluation-helper version omitted those interval keys; that first attempt stopped before writing a test summary. The corrected evaluation did not change the checkpoint or validation-selected thresholds.

## Interpretation and limits

The result supports a retrospective MFIDDR finding: four-view inference has higher patient-level AUROC than the F1-only budget, and the four-view high-sensitivity operating point exceeds 90% accuracy on this official split. Accuracy alone hides missed referable patients; the operating-point table is the relevant result. These data do not establish prospective clinical benefit, transport to another camera/site/population, calibration stability, or autonomous clinical safety. The zero-miss point is a finite-sample observation, not a guarantee.

The official test was opened only after training and validation selection were complete. A first evaluator invocation stopped on a missing summary field before producing metrics; no test-derived value was used to alter the model or thresholds. The corrected invocation and fixed-decision bootstrap are recorded in the local run outputs.

## Reproducibility and release boundary

The source code, configuration, aggregate report, and bootstrap summary are suitable for the project repository. Dataset images, manifests with patient identifiers, row-level predictions, logs containing per-case data, and model checkpoints are kept local and are not included in the public commit. Dataset redistribution permissions for those excluded artifacts are not implied by this report.
