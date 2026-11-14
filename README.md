# Golden Horn 8159: Accessible Retinal Health Impact Project

**Golden Retina** is the software project of Golden Horn 8159's effort to make
retinal-image analysis more accessible. The long-term aim is to support earlier
recognition of retinal disease in communities with limited access to eye-care
services, while keeping clinicians and patients at the center of care.

This repository contains working image-analysis prototypes, a local scan
interface, and the engineering and evaluation tools used to improve them. It is
an impact-driven development project, not only a paper repository.

## The Need

Many retinal conditions can threaten sight, and access to timely specialist
assessment is uneven. Fundus photography can make retinal images easier to
capture and review, but an AI result is only useful when its limits are
understood and the surrounding care pathway is reliable. Golden Retina explores
software that can help organize and interpret retinal images as one part of a
future, clinician-led screening workflow.

## What Works Today

- A local browser interface accepts a fundus image and returns separate
  research-model outputs for the conditions represented in the RFMiD label set.
- A JSON API exposes the same inference workflow for software integration.
- A separate diabetic-retinopathy workflow supports paired-image case review,
  image-quality handling, and an auditable case report.
- The software records model and input provenance where supported by each
  workflow, and the research station processes uploaded images in memory rather
  than saving them.

The multi-disease RFMiD prototype uses a fine-tuned ConvNeXt-Tiny model. Its
held-out RFMiD image-split results included a macro AUROC of 0.894 across 36
label heads with both classes represented in that split. For the dataset's
overall `Disease_Risk` label, accuracy was 92.7%, sensitivity 92.9%, and
specificity 91.8% at the development threshold. These are retrospective
dataset-split measurements, not evidence of performance in a clinic or on new
populations. The split's patient-level independence has not been verified.

## Try the Local Research Interface

The RFMiD checkpoint is a local model artifact and is not distributed in this
repository. After obtaining the project environment and placing the checkpoint
at `models/rfmid_convnext_tiny_multilabel_20261021.pt`, start the station:

```bash
PYTHONPATH=src .venv-golden-retina/bin/python -m golden_retina.rfmid_multidisease_station
```

Open [http://127.0.0.1:8767](http://127.0.0.1:8767). The page is bound to the
local computer only. It accepts JPEG, PNG, and WebP images up to 16 MB. The
JSON endpoint is `POST /api/predict`; see
[`docs/rfmid_multidisease_ui_api.md`](docs/rfmid_multidisease_ui_api.md) for the
request and response details.

For the separate DR case workflow and browser station, see
[`docs/api_contract.md`](docs/api_contract.md) and
[`docs/hardware_enclosure_clinical_station.md`](docs/hardware_enclosure_clinical_station.md).

## Responsible Use

**Research prototype only.** The model outputs are not diagnoses and must not
be used to decide that a person is healthy, defer care, or initiate treatment.
The models have documented errors and have not been clinically validated. A
positive, negative, or ungradable result requires appropriate professional
review; out-of-scope diseases and image types may not be detected.

The current RFMiD model is evaluated retrospectively on an image-level split.
Patient-level separation, external-site performance, calibration for clinical
use, prospective workflow benefit, and safety have not been established. The
separate DR score-policy replay figures in this repository are development
references, not raw-image model accuracy or independent clinical validation.

Audience-specific project briefs are available: the [technical system note](docs/golden_retina_technical_system_note_20261114.md)
documents the current per-disease specialist sweep and its evidence boundaries;
the [customer and research-partner brief](docs/golden_retina_customer_partner_brief_20261114.md)
describes the prototype's present capabilities and non-clinical status.

## Impact Path

1. Build accessible, understandable software for retinal-image review.
2. Improve disease-specific models using properly labeled and permissioned
   data, while reporting each condition separately.
3. Validate performance on patient-disjoint and independent clinical cohorts
   with qualified reference grading.
4. Work with eye-care professionals and community partners to assess whether
   the technology can support a safe, useful screening pathway.

The goal is meaningful access and better-informed care, not replacing eye-care
professionals or promising perfect automated diagnosis.

## Development

Install dependencies in a project environment, then run the test suite:

```bash
PYTHONPATH=src .venv-golden-retina/bin/python -m pytest -q
```

Key areas:

- `src/golden_retina/`: image models, inference services, and workflow software
- `docs/`: product, data, validation, and API documentation
- `configs/`: model and evaluation configurations
- `tests/`: software and evidence-integrity tests
- `paper/`: research manuscripts and reproducibility material

Dataset images, model checkpoints, and row-level prediction exports are not
included by default. Check each dataset's terms before acquiring, using, or
redistributing data or derived artifacts.

## Contact and Collaboration

Golden Horn 8159 welcomes conversations with eye-care professionals,
researchers, community health organizations, and technical collaborators who
can help evaluate responsible ways to improve access to retinal care.

## Separate eye-finding research prototype

The repository also contains a separate synthetic eye-finding and tracking experiment; it does not diagnose retinal disease. Its current shared-eye segmentation candidate achieved 0.854 fresh-seed synthetic macro Dice and 24/24 pupil presence per image side on 24 held-out renderer seeds. It beat a train-only constant-coordinate baseline on a 32-frame synthetic translated-eye sequence. The small frozen checkpoint is included at `models/eye_finding/charmorph_shared_eye_v2b.pt`; its SHA-256 is recorded in the model run card. Evaluation used one synthetic base identity and has no human-camera validation. The research-only single-frame CLI and loopback API are documented in [`eye_finding_ai/realistic_3d_v3/README.md`](eye_finding_ai/realistic_3d_v3/README.md).

The latest retinal-model development results and their test-set limitations are summarized in [`docs/latest_model_state_20261110.md`](docs/latest_model_state_20261110.md).
