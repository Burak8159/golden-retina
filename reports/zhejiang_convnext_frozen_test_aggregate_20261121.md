# Zhejiang ConvNeXt frozen patient-grouped test results

**Run date:** 2026-11-21  
**Model:** ImageNet ConvNeXt-Tiny, partially fine-tuned shared backbone with six one-vs-rest sigmoid outputs (AMD, DR, pathological myopia, retinal detachment, RVO, uveitis).  
**Partition:** Frozen Zhejiang patient-grouped test split; 136 eyes from 120 patients.  
**Thresholds:** Each disease threshold was copied from the prior validation-only report, selected to maximize sensitivity subject to validation specificity ≥90%. No threshold was selected or adjusted on this test partition.  
**Uncertainty:** 2,000 stratified patient-cluster bootstrap replicates; intervals condition on the selected thresholds and do not account for earlier model-selection exposure.

| Disease | Positive eyes | Accuracy | Sensitivity | Specificity | Missed positives | AUROC | Sensitivity 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| AMD | 17 | 94.1% | 94.1% | 94.1% | 1/17 | 0.980 | 78.6–100.0% |
| DR | 19 | 89.7% | 94.7% | 88.9% | 1/19 | 0.972 | 82.4–100.0% |
| PM | 20 | 92.6% | 85.0% | 94.0% | 3/20 | 0.966 | 70.0–100.0% |
| RD | 20 | 89.7% | 95.0% | 88.8% | 1/20 | 0.972 | 85.0–100.0% |
| RVO | 20 | 93.4% | 95.0% | 93.1% | 1/20 | 0.983 | 85.0–100.0% |
| Uveitis | 20 | 82.4% | 35.0% | 90.5% | 13/20 | 0.864 | 15.0–55.1% |

## Interpretation

This is a one-shot, patient-grouped **internal test-partition** evaluation within one Zhejiang UWF source. It is not independent external validation, prospective evaluation, or clinical validation. The source uses mutually exclusive diagnosis labels; these results do not establish performance when diseases co-occur. Prior project exposure to this source, especially DR, limits confirmatory interpretation.

The results do not meet a zero-miss criterion: there were 1 AMD, 1 DR, 3 pathological-myopia, 1 retinal-detachment, 1 RVO, and 13 uveitis misses. Uveitis sensitivity was 35.0% (13/20 missed); pathological-myopia sensitivity was 85.0% (3/20 missed). Test specificity fell below 90% for DR and retinal detachment despite validation thresholds targeting at least 90%. The small positive counts yield wide patient-bootstrap intervals. These heads must not be presented as safe negative screens or medical risk probabilities.

## Reproducibility boundary

The evaluator, checkpoint SHA-256, validation-summary SHA-256, and frozen-manifest SHA-256 are recorded in the aggregate JSON. Raw images, patient identifiers, row-level scores, split manifest, and model checkpoint remain private.
