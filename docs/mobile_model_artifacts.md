# Mobile model artifacts

**Bundle:** `20261114-r3`

**Format:** ONNX

**Location:** [`../mobile_models/onnx`](../mobile_models/onnx)

The iOS and Android research apps bundle one synthetic eye-localization model
and 17 disease/finding specialists in the research artifact registry. The apps expose four condition-specific classifiers (diabetic retinopathy, age-related macular degeneration, myopia, and branch retinal vein occlusion); the remaining specialists stay out of the consumer-facing classifier because they failed the documented support/sensitivity screen. The apps execute the four selected models locally; they
do not upload selected images. `manifest.json` records each model's checksum,
input/output contract, source checkpoint hash, and version.

## Attribution and source

The disease/finding specialists were trained from the RFMiD image dataset. The
dataset descriptor by Pachade et al. identifies the dataset as licensed under
Creative Commons Attribution 4.0 International (CC BY 4.0):

> Pachade, S. et al. “Retinal Fundus Multi-Disease Image Dataset (RFMiD): A
> Dataset for Multi-Disease Detection Research.” *Data* 6(2), 14 (2021).
> https://doi.org/10.3390/data6020014

CC BY 4.0 requires attribution and indication of changes when redistributing
adapted material. The models are trained artifacts, not copies of the dataset;
the dataset images and row-level labels are not included in this repository.
The repository's top-level `LICENSE` continues to govern Golden Retina code
and artifacts except where the cited dataset license applies. No extra model
license or clinical-use permission is granted by this attribution note.

The eye-localization model was trained using synthetic renders only. It has not
been validated on people, infrared camera captures, or goggles.

## User-facing condition selection

The app exposes only DR, ARMD, MYA, and BRVO: each had at least 20 positive RFMiD test images and sensitivity of at least 0.70 at the recorded 0.5 threshold in the existing image-split evaluation. This is a post hoc development-display filter applied after reviewing the recorded image-split test summary, not a prespecified model-selection rule or clinical acceptance criterion. Because the final test results informed which conditions are exposed in the app, these values are exploratory and must not be treated as unbiased confirmatory estimates. In that evaluation, sensitivity was 86.3% (DR; 124 positive test images), 74.2% (ARMD; 31), 100.0% (MYA; 32), and 78.3% (BRVO; 23). The test partition is image-level; patient independence is unverified. Other labels, including drusen, optic-disc findings, and rare conditions, are omitted from the app classifier because their current test results/support did not meet this screen. The artifact registry retains those research models for auditability, but the mobile apps no longer offer them for classification.

## Evidence limits

The disease/finding models were evaluated on RFMiD's image-level split. Patient
independence, external-site performance, calibration for clinical use, and
performance on Golden Retina hardware have not been established. Several rare
labels have very small test support. The model outputs are uncalibrated per-image research scores, not probabilities of disease, diagnoses, or clinical risk estimates. No confidence or risk percentage is shown in the guided demo. See the [technical system
note](golden_retina_technical_system_note_20261114.md) and the
[per-disease metrics ledger](../results/expert_sweep/per_disease_metrics.csv).

The bundle is not a medical device and must not be used to make care decisions.
