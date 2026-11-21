# Zhejiang UWF disease probe: aggregate report

**Date:** 2026-11-21
**Model:** Frozen torchvision ConvNeXt-Tiny ImageNet1K_V1 features with six independent L2 logistic heads (C=1). Heads were trained only on the patient-grouped train partition, with equal patient and class contribution.
**Threshold:** Selected on the validation partition to maximize sensitivity subject to specificity >=90%.
**Analysis unit:** Eye; multiple images for one eye were max-pooled. Uncertainty uses 2,000 class-stratified patient-cluster bootstrap replicates.
**Boundary:** Internal single-source UWF development estimate only. This is not independent external validation or clinical validation. The Zhejiang DR cohort was previously used for a DR-versus-Healthy project analysis; its current DR test results are exposed. Thresholds were not selected on test.

| Disease task vs Healthy | Test eyes (positive / control) | Accuracy | Sensitivity | Misses / positives | Specificity | AUROC | Patient-cluster 95% CI for sensitivity | Patient-cluster 95% CI for AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AMD | 37 (17 / 20) | 94.6% | 88.2% | 2 / 17 | 100.0% | 0.997 | 70.6–100% | 0.982–1.000 |
| DR* | 39 (19 / 20) | 94.9% | 100.0% | 0 / 19 | 90.0% | 0.995 | 100–100% | 0.975–1.000 |
| Pathological myopia | 40 (20 / 20) | 97.5% | 95.0% | 1 / 20 | 100.0% | 1.000 | 85–100% | 1.000–1.000 |
| Retinal detachment | 40 (20 / 20) | 80.0% | 65.0% | 7 / 20 | 95.0% | 0.943 | 45–85% | 0.865–0.993 |
| Aggregate RVO | 40 (20 / 20) | 90.0% | 95.0% | 1 / 20 | 85.0% | 0.978 | 85–100% | 0.925–1.000 |
| Uveitis | 40 (20 / 20) | 52.5% | 25.0% | 15 / 20 | 80.0% | 0.630 | 10–45% | 0.455–0.785 |

\*Prior project model predictions already existed for Zhejiang DR and Healthy patients; do not call the current DR estimate independent or untouched. “RVO” is an aggregate source label and does not distinguish BRVO from CRVO. “Pathological myopia” is the dataset's PM category, not generic myopic fundus appearance.

## Reading these results

- The high AMD and PM AUROCs are based on 17 and 20 disease eyes, respectively; they should not be read as reliable population performance. Patient-cluster bootstrap intervals describe uncertainty conditional on this small held-out sample and may be degenerate when observed ranks are perfectly separated.
- RD and uveitis are clear failures at the prespecified validation-derived operating points. Their test outcomes are not used to revise thresholds or select models. Any follow-on encoder comparison must use training/validation data only.
- DR zero misses among 19 eyes does not establish zero population miss risk. Its test cohort is previously exposed.
- Model scores are uncalibrated and are not disease-risk percentages or diagnoses. The source has one UWF acquisition domain and mutually exclusive categories; real-world comorbidity and cross-camera transfer are not represented.
- No glaucoma result was produced in this run. HYGD's previous group-verified train/validation data were already used for development and it has no unused patient test set.

This report publishes aggregate results only. Row-level manifests, per-eye scores, model heads, images, and bootstrap replicates remain in the authorized private archive and are not included in the public repository.
