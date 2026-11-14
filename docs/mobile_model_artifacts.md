# Mobile model artifacts

**Bundle:** `20261114-r3`

**Format:** ONNX

**Location:** [`../mobile_models/onnx`](../mobile_models/onnx)

The iOS and Android research apps bundle one synthetic eye-localization model
and 17 disease/finding specialists. The apps execute these models locally; they
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

## Evidence limits

The disease/finding models were evaluated on RFMiD's image-level split. Patient
independence, external-site performance, calibration for clinical use, and
performance on Golden Retina hardware have not been established. Several rare
labels have very small test support. The model outputs are uncalibrated research
scores, not diagnoses or clinical risk estimates. See the [technical system
note](golden_retina_technical_system_note_20261114.md) and the
[per-disease metrics ledger](../results/expert_sweep/per_disease_metrics.csv).

The bundle is not a medical device and must not be used to make care decisions.
