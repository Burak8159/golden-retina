# Mobile model artifacts

**Bundle:** `20261118-r5` (model weights unchanged; app catalog expanded)

**Format:** ONNX

**Location:** [`../mobile_models/onnx`](../mobile_models/onnx)

The iOS and Android research apps use one synthetic eye-localization model and
17 RFMiD disease/finding specialists available individually through the model
registry. The on-device model lab organizes these outputs into eight expandable
groups. The four post-hoc shortlist heads (diabetic retinopathy, AMD, myopia,
and BRVO) are distinguished from 13 exploratory heads. The exploratory set
includes disease labels and observable findings; it is not a list of 13
validated diseases. Users install only the heads they choose, and inference
runs locally without uploading selected images. `manifest.json` records model
checksums, input/output contracts, checkpoint hashes, shortlist/exploratory
roles, output types, and the shared group taxonomy.

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

## App output groups and evidence status

The app presents the full catalog in these groups:

1. **Diabetic retinopathy:** DR.
2. **Retinal vein occlusions:** BRVO, CRVO.
3. **Macula and central retina:** AMD/ARMD, drusen, central serous retinopathy, epiretinal-membrane appearance, macular scar.
4. **Optic-disc findings:** cupping, pallor, edema. These are image findings, not glaucoma diagnoses.
5. **Myopia and fundus patterns:** myopia, tessellated-fundus appearance.
6. **Chorioretinal inflammation:** chorioretinitis label.
7. **Scarring and treatment findings:** laser scar, retinal scar.
8. **Retinal structural findings:** retinal traction.

The four core shortlist heads—DR, AMD/ARMD, myopia, and BRVO—each had at least 20 positive RFMiD test images and sensitivity of at least 0.70 at the recorded 0.5 threshold in the existing image-split evaluation. This is a post-hoc development-display filter informed by that test summary, not a prespecified selection rule; these values are exploratory and not unbiased confirmatory estimates. Recorded sensitivity was 86.3% (DR; 124 positive test images), 74.2% (ARMD; 31), 100.0% (MYA; 32), and 78.3% (BRVO; 23). The test partition is image-level; patient independence is unverified.

The other 13 heads are exposed as **exploratory** so results can be inspected, but some have very low positive support and poor sensitivity. For example, the RFMiD ODP output missed all 24 positive test images, the ERM head missed all five, and several other rare-label heads were evaluated on fewer than 20 positives. Their app visibility is not an endorsement for screening or a claim of reliable detection. Glaucoma is not included: the older HYGD internal-validation result has no checkpoint in this checkout or mobile bundle, and RFMiD optic-disc cupping is a finding rather than a glaucoma diagnosis.

## Evidence limits

The disease/finding models were evaluated on RFMiD's image-level split. Patient
independence, external-site performance, calibration for clinical use, and
performance on Golden Retina hardware have not been established. Several rare
labels have very small test support. The model lab shows the uncalibrated per-image score and whether it crosses the 0.50 research cutoff. These are not calibrated disease probabilities, diagnoses, or clinical risk estimates. The guided demo uses fixed sample values solely to preview the eight-group layout; they are not produced by models or images. See the [technical system
note](golden_retina_technical_system_note_20261114.md) and the
[per-disease metrics ledger](../results/expert_sweep/per_disease_metrics.csv).

The bundle is not a medical device and must not be used to make care decisions.

### Current disease-specific measurements

The existing RFMiD image-split report at threshold 0.50 is shown below. These results are exposed, image-level development metrics; they are not estimates of patient-level or clinical performance.

| App output label | RFMiD test positives | Accuracy | Sensitivity | Specificity |
|---|---:|---:|---:|---:|
| Diabetic retinopathy | 124 | 92.81% | 86.29% | 94.38% |
| Age-related macular degeneration | 31 | 94.84% | 74.19% | 95.89% |
| Myopia | 32 | 97.19% | 100.00% | 97.04% |
| Branch retinal vein occlusion | 23 | 95.63% | 78.26% | 96.27% |

### Glaucoma coverage

Glaucoma is not an installed or exposed mobile classifier. The repository contains a historical HYGD internal-validation summary, but the trained RETFound-MAE checkpoint is not present in this checkout and is not an ONNX artifact in the mobile registry. Therefore the apps cannot run that model. The RFMiD optic-disc-cupping head is intentionally excluded: optic-disc cupping is an observed finding and is not specific to glaucoma. The historical HYGD summary used 125 validation images (95 positive), so it must not be presented as external or confirmatory evidence.
