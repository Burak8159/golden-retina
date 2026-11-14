# Golden Retina

## Product and research-partner brief

**For:** eye-care professionals, community health organizations, research groups, and potential future users
**Snapshot date:** 14 November 2026
**Current status:** Research prototype; not cleared or validated for clinical use.

## What Golden Retina is

Golden Retina is a Golden Horn 8159 project exploring software that analyzes color photographs of the back of the eye. The long-term goal is to study whether computer-assisted image review could make retinal screening more accessible, including in communities with limited access to eye-care professionals.

The repository contains an earlier local research interface and API, plus model-development tools. A separate training effort is evaluating one model per retinal finding in a public research dataset. The active specialist candidates and the current interface checkpoint are separate; the new per-disease results have not yet been integrated into that interface. This work is still being developed and is not an operational screening service.

## What it can do today

The earlier interface returns separate image-level scores for the labels in its RFMiD multi-label checkpoint. The active specialist program is separately evaluating 17 candidate labels or findings: DR, ARMD, drusen (DN), myopic fundus appearance (MYA), branch retinal vein occlusion (BRVO), optic-disc cupping (ODC), pallor (ODP), edema (ODE), tessellation (TSLN), laser-scar finding (LS), central serous retinopathy (CSR), retinitis finding (RS), chorioretinitis finding (CRS), central retinal vein occlusion (CRVO), epiretinal membrane appearance (ERM), macular scar (MS), and retinal traction (RT). The last four have very few positive examples and are exploratory. All 17 targets now have one RFMiD image-split test result; this does not establish patient-level or clinical validity. Media haze is tracked separately as a visibility/quality signal, not a disease. These labels describe the research dataset's annotation targets; they should not be read as a validated list of clinical diagnoses the product can make.

The software does **not** examine every eye condition. It accepts fundus photographs only; it cannot replace OCT, visual-field testing, corneal imaging, slit-lamp examination, or a clinician's history and examination. A score for a dataset label should not be interpreted as a complete diagnosis.

Separate [iOS](../ios/GoldenRetina/README.md) and [Android](../android/README.md) concept apps show a simulated device-discovery, connection, right-eye/left-eye scan, and sample-results journey. The goggles path is locked because the hardware is not available. The demo does not use Bluetooth or a camera, does not call a screening model, and does not produce patient-specific results.

## Current evidence—why this is still research

The current specialist results are from a held-out **image split** within RFMiD. They are not results from a new clinic, a patient-disjoint cohort, or the Golden Retina camera. At the fixed development threshold of 0.5, some test-set results illustrate why high accuracy is not enough:

| Research label | Test accuracy | Test sensitivity | Positive images missed |
|---|---:|---:|---:|
| Diabetic retinopathy | 92.81% | 86.29% | 17 / 124 |
| Age-related macular degeneration label | 94.84% | 74.19% | 8 / 31 |
| Drusen finding | 92.97% | 2.17% | 45 / 46 |
| Optic-disc pallor finding | 96.25% | 0.00% | 24 / 24 |
| Central serous retinopathy label | 97.66% | 92.31% | 1 / 13 |
| Epiretinal membrane appearance label | 99.22% | 0.00% | 5 / 5 |
| Macular scar finding (not multiple sclerosis) | 98.44% | 28.57% | 5 / 7 |
| Retinal traction finding | 99.38% | 20.00% | 4 / 5 |

These are dataset-specific image-level measurements. Several positive counts are small, patient-level separation cannot be verified from the staged metadata, and the test set has already been used once for the reported evaluation. The complete per-disease results and technical methods are in the [technical system note](golden_retina_technical_system_note_20261114.md). The separate DR DINOv2 recovery uses training and validation data only; the earlier DR test result remains the reported result.

**A high accuracy value does not mean that a model safely rules out disease.** For example, the drusen model's accuracy was 92.97% while it missed 45 of 46 positive test images. The current outputs must not be used to reassure a patient, delay care, initiate treatment, or make referral decisions.

## What a future clinical workflow might look like

If future independent studies support it, a carefully designed product could help organize retinal images for professional review. A possible future workflow would be:

1. A trained operator obtains a fundus image using a validated camera and protocol.
2. Software checks whether the image is usable and produces clearly labeled research scores.
3. A qualified eye-care professional reviews the image and all relevant clinical information.
4. Any concerning, uncertain, or unusable case follows a defined referral or repeat-imaging pathway.

This is a **future concept**, not a currently validated or deployable clinical pathway. The current model sweep has not established safe image-quality handling, clinical thresholds, or patient outcomes.

## Who can engage now

Potential partners can discuss research questions, label interpretation, target conditions, camera compatibility, and the design of future independent evaluations. Conversations are exploratory; they do not imply that a pilot is ready to begin or that the system is suitable for patient care.

Before any clinical evaluation, the project would need a frozen model and intended-use statement; patient-level independent validation with qualified reference grading; camera- and site-relevant testing; prespecified sensitivity, specificity, calibration, and failure criteria; appropriate ethics and consent determinations; privacy and security controls; and a defined clinician-led workflow. Regulatory and institutional requirements would also need to be assessed by the responsible organizations.

## Data and privacy for this prototype

The local research station is designed for local development and is not a production service. It does not provide production authentication, internet-facing security, uptime guarantees, or clinical-record management. Do not upload identifiable patient images or use the prototype for clinical decisions. Use only data for which the user and project have appropriate access rights, and check each dataset's terms before sharing images, derived records, model weights, or results.

## Project links

- [Golden Retina source repository](https://github.com/Burak8159/golden-retina)
- [Technical system note and per-disease results](golden_retina_technical_system_note_20261114.md)
- [Local RFMiD research-interface API description](rfmid_multidisease_ui_api.md)
- [Research manuscript and release materials](../paper/README.md)

Golden Retina's goal is to investigate whether retinal-image software can support more accessible, clinician-centered care. The present evidence is an early technical development step; it is not proof of clinical readiness or patient benefit.
