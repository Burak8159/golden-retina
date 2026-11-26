# Golden Retina disease development closeout — 2026-11-27

This report consolidates the current disease-specific development evidence. It is not a clinical validation report, and none of these results establish clinical safety or readiness for autonomous diagnosis.

## Current disease dispositions

| Task | Best available evidence | Disposition |
|---|---|---|
| Diabetic retinopathy (DR) | DeepDRiD patient-grouped internal development: matched ResNet-50, ConvNeXt-Tiny and DINOv2 comparisons on fit/calibration/validation only. At the 100%-calibration-sensitivity operating point, paired-view DINOv2 validation AUROC 0.912, sensitivity 30/32 (2 misses), specificity 65.2%. ConvNeXt-Tiny reached 32/32 observed sensitivity but 30.4% specificity. FIVES image-level test: ResNet-50 sensitivity 41/50, specificity 31/50, accuracy 72%; 9 misses. | Development evidence only. FIVES is NO-GO at the evaluated operating point; DeepDRiD is a small, previously used internal cohort, not independent validation. |
| Age-related macular degeneration (AMD) | FIVES image-level test: ResNet-50 AUROC 0.937, sensitivity 49/49, specificity 15/50, accuracy 64.6%, 35 false referrals. | NO-GO at this operating point; no patient-independent or external validation. |
| Cataract severity | CSDI repeated image-SHA-grouped exploratory CV (187 images): ConvNeXt-Tiny MAE 0.678/10, RMSE 0.882, within-one-point 77.8%; DINOv2 MAE 0.681; ResNet-50 MAE 1.027; quality-only MAE 1.821. | Exploratory severity regression only; no patient IDs, not a validated cataract diagnosis. |
| Retinal vein occlusion (RVO) | CLARUS700 single-center UWF. Five archives matched publisher byte counts and MD5; extracted ZIP CRC/path checks passed. Train/validation/calibration: 585/148/146 images and 374/99/93 patient groups; zero patient or exact-image overlap. One absent train image and two duplicate validation rows excluded. ConvNeXt-Tiny epoch 5 selected by validation AP 0.929 (AUROC 0.987). On validation at the 95%-sensitivity training-calibration threshold: sensitivity 12/12, specificity 58.8%, accuracy 62.2%, 56 false referrals; patient-cluster bootstrap 95% CI for specificity 50.0–68.3%. | Development candidate only. Validation selected epochs/candidate and is selection-exposed; official test manifest/labels were not read. One site/device and only 12 validation positives. |
| Glaucoma | RIM-ONE DL hospital training partition only (311 images, 116 positive). No patient IDs; 311 unique exact-image hashes; fit/calibration/validation 187/62/62 with zero exact-hash overlap. Three random-init architectures compared. MobileNetV3-Small selected at epoch 6 by validation AP 0.732. At the calibration threshold targeting 95% sensitivity, validation sensitivity 22/23 (1 miss), specificity 6/39 (15.4%), accuracy 45.2%, 33 false positives, AUROC 0.778, AP 0.732. | Weak source-only development result, not deployable. Official test was not accessed in this run; prior test exposure elsewhere in project remains a limitation. |

## API and artifact boundary

The versioned research API is a local RTX service. Its registry pins each checkpoint by SHA-256. It returns raw sigmoid scores, explicitly not calibrated probabilities or diagnoses. RVO and glaucoma checkpoints, dataset files, per-image rows, and predictions remain on the RTX. Only aggregate JSON reports and source/configuration artifacts are intended for the repository.

## Overall conclusion

Current data support separate experimental disease tasks, not one universal retinal-disease classifier. DR and AMD have unfavorable or constrained operating-point evidence; cataract has only exploratory severity regression; RVO has promising rank separation but poor specificity and a small, selection-exposed validation set; glaucoma remains weak. None should be used for patient care or described as clinically validated. Further credible progress requires a new, untouched patient cohort and prespecified evaluation.
