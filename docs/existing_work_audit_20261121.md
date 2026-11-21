# Golden Retina existing-work reconstruction

> **Historical baseline, not the current live status.** This audit records the initial VM-recovery checkpoint before the 2026-11-21 RFMiD L4 work. Later sections in this file record additional discoveries and budget revisions, but the current campaign state is phase 34/39: one guarded L4 instance is active, the original disease campaign is closed, and the new disease-coverage extension is running. Use `tools/progress_tracker/status.json` and the latest dated reports for current compute, cost and results. Older “no training started,” “no replacement VM,” and ₺850/₺1,000 statements below are historical and superseded.

**Audit date:** 2026-11-21 (Europe/Istanbul)
**Scope:** Current Golden Retina checkout, repository artifacts and metadata, mobile model manifest, existing CCG/DR reports, local test collection, and read-only Google Cloud inventory.
**Data rule:** This Mac is not to be used for dataset downloads or extraction. Future dataset transfer, preparation and training are VM-only after cloud and cost gates pass.

## Executive status

Golden Retina is an established research/software codebase, not a blank project. It contains retinal model trainers and evaluations, a multi-field acquisition/CCG research track, an 18-artifact on-device ONNX bundle, native iOS and Android demo apps, and synthetic eye-localization work. The checkout also preserves a large historical `results/` tree. These assets are useful starting points, but their presence does not itself establish reproducibility, external validity, clinical readiness, or live app integration.

No dataset was downloaded or extracted to this Mac during this audit. The conventional `data/`, `datasets/`, and `external/` roots are absent in the active checkout, and an image-extension scan found only four research-figure PNGs. A prior closeout record describes private archives and recovered artifacts; those archives were not opened or copied as part of this audit.

## Inventory and verification

| Area | What exists | Verification and limit |
|---|---|---|
| Retinal classification | 17 RFMiD specialist ONNX artifacts and corresponding manifest entries; numerous training/evaluation scripts and historical result folders | All 18 declared ONNX files (17 specialists plus eye tracker) matched the SHA-256 values in `mobile_models/onnx/manifest.json`. A CPU smoke inference loaded DR and eye-tracker ONNX files and returned finite outputs of the declared shapes. This verifies artifact integrity/loading only, not prediction quality. |
| Reported mobile disease metrics | Four RFMiD image-split results: DR 92.81% accuracy / 86.29% sensitivity / 94.38% specificity (124 positive test images); AMD 94.84% / 74.19% / 95.89% (31); myopia 97.19% / 100.00% / 97.04% (32); BRVO 95.63% / 78.26% / 96.27% (23) | Values are from `results/expert_sweep/per_disease_metrics.csv` and are exposed image-level development results. Patient independence is unverified; the shortlist was post-hoc; none is clinical validation. Other 13 heads include findings/low-support outputs and must not be represented as validated diseases. |
| Prior multi-task RETFound work | A dated run summary records a patient-disjoint internal train/validation split and DR, glaucoma-labelled HYGD, and source-stratified validation measurements | The trained RETFound checkpoint is not a distributable artifact in this checkout. HYGD support is small (125 images), and the report is development-only, not independent external validation. The saved summary and manifest hashes are in `docs/latest_model_state_20261110.md` and `results/technical_dev_20261108_rerun/`. |
| Eye localization/tracking | Synthetic-only checkpoints, run cards and two ONNX/runtime artifacts | Run cards describe synthetic renderer data. No human, real IR camera, physical eye motion, or goggle validation is evidenced. Hashes and a finite-output CPU smoke check passed for the ONNX tracker. |
| Apps | SwiftUI iOS and Jetpack Compose Android app sources | Both READMEs describe a guided demo, not a live camera/Bluetooth/model workflow. Current output values are demo examples; the apps do not invoke these ONNX models. Hardware pairing protocol is not present. |
| CCG / adaptive acquisition | Policy/evaluation code, DeepDRiD manifests, cross-fit predictions, cost/risk frontiers, bootstrap/decision summaries, plots, and reports | A prior summary reports paired-view development evidence; `results/flagship_decision.json` explicitly labels the autonomous candidate uncertified and says only 48 positive patients were available versus 59 required for the stated zero-miss certificate. The separate Track B record `results/deepdrid_track_b_metrics.json` remains blocked for a required cross-fit file and records an inferred field-suffix mapping. Treat the CCG results as development evidence, not a completed clinically validated policy. |
| Dataset files | No conventional dataset root in the active checkout | Historical registries and manifests exist, but local availability, source terms, labels, patient/eye linkage, duplicates and exact cohort provenance must be rechecked on the VM before use. mBRSET remains credentialed-access gated. |
| Evaluation/tests | 46 Python test modules are present | `python3 -m pytest -q` initially failed collection because the package path was not configured. With `PYTHONPATH=src`, collection still failed for missing dependencies (`scipy`, `scikit-learn`, `joblib`, `torchvision`), a Python 3.9 incompatibility in an importer test, and two absent paper-reproducibility modules. The full suite is not verified as passing. Do not install large ML dependencies on this Mac; qualify the VM environment first. |

## Cloud inventory and constraints

- Google Cloud CLI authentication is present. The configured default project is not the Golden Retina project, so all future commands must name the Golden Retina project explicitly.
- In `golden-retina-dev-20261107`, the source Golden Retina L4 VM `golden-retina-l4-01` is **TERMINATED**. Its 150 GB standard persistent disk remains attached and preserved. A same-project 150 GB snapshot, `golden-retina-l4-01-zone-copy-20261121`, is now **READY** for a neighboring-zone recovery. No reserved addresses were found.
- Other visible VM instances are terminated. Conrad-named resources are unrelated and were not touched.
- Regional L4 quota (limit 1, usage 0) exists in multiple supported regions, but quota does not guarantee capacity. Requests in `us-west1-a/b/c`, `us-central1-a/b/c`, `us-east4-a/c`, and `us-east1-b/c` failed with L4 stockouts; standard and Spot requests in `us-east1-d` also failed. The unattached us-east1-d clone was deleted. A snapshot-derived disk is RESTORING in `us-east1-c` for a fresh Spot request; no replacement VM exists yet.
- The Cloud Billing API was enabled on the Golden Retina project for read-only checks; the project is linked to a billing account. An existing **₺1,400 monthly alert budget** is configured for this project; its thresholds are alerts, not a hard spend cap. The Billing Budgets API response does not expose actual accrued charges. BigQuery is enabled but no datasets were listed, so no billing-export table was available to query. The authenticated Cloud Console report scoped to this project shows TRY 0.00 net total and forecast for Nov 1–20, 2026; gross Compute Engine TRY 3,925.66 and Networking TRY 6.43 were offset by TRY 3,932.09 in savings. Nov 21 same-day charges may lag. Future estimates use list price and do not assume continued credits. L4 capacity requests failed in multiple zones through `us-east1-d`; the source VM remains stopped, a snapshot-based disk is restoring in `us-east1-c`, and no replacement VM exists yet. No dataset transfer or training has started.
- The current list-price planning scenario uses eight hours of `g2-standard-4`, one 150 GiB standard disk for 30 days, the READY snapshot's observed 50.204 GiB stored size for 24 hours, eight hours of replacement-disk overlap, and up to 8 GiB outbound transfer. At the recorded rate and Nov. 20 TCMB USD sell rate (≈₺50.49/USD), this is about **₺652 before a 25% stress allowance (≈₺816 total)**. This is a bounded estimate, not a bill; it excludes taxes, account billing adjustments and unrelated prior usage. Nov 1–20 project net was TRY 0 after savings, but this does not cap future costs and Nov 21 charges may lag. Retain the 7.5-hour STOP action, 8 GPU-hour cap, ₺850 planned cutoff, and ₺1,000 absolute ceiling; refresh the report before training.
- The original disk remains preserved. L4 requests in `us-west1-a/b/c`, `us-central1-a/b/c`, `us-east4-a/c`, and `us-east1-b/c`, plus standard/Spot in `us-east1-d`, were rejected for capacity. The unattached us-east1-d clone was deleted. A replacement disk from the READY snapshot is restoring in `us-east1-c`; no replacement VM exists yet. No training or dataset transfer has started.

## Reusable components

1. `src/golden_retina/` contains substantial DR, multi-disease, model-evaluation, split, risk-control and policy code.
2. `results/deepdrid_*`, `results/flagship_*`, `results/formal_risk_control_*`, and `plots/` provide a historical basis for adaptive-view/CCG analysis.
3. `mobile_models/onnx/manifest.json` provides versioned model contracts, hashes, output semantics, and evidence caveats.
4. `eye_finding_ai/` contains synthetic eye-rendering/training work, small checkpoints, metrics, and runtime notes.
5. `ios/` and `android/` contain separate user-facing demo applications whose live hardware and model interfaces remain future work.
6. `scripts/cloud/vm_preflight.sh` is a non-destructive remote readiness script prepared to prevent a repeat of the missing `models/` output-directory failure. It has not been run because the current billing gate is unresolved.

## Main gaps before model work can resume

1. **Cost gate:** the authenticated project report currently covers Nov 1–20 and shows net TRY 0 after savings; same-day Nov 21 charges may lag, and the ₺1,400 project alert is not a hard cap. The campaign estimate includes the observed 50.204-GiB snapshot size and is about ₺816 with the 25% stress allowance. Refresh billing after VM provisioning and before training; stop by the ₺850 planned cutoff and never cross the ₺1,000 ceiling. The cloud budget is not enforced automatically.
2. **VM-only data workflow:** confirm the authorized VM and disk contents, then stage any newly acquired datasets directly there. Do not download or extract datasets on this Mac.
3. **Provenance:** reconcile historic manifests/reports with data physically available on the VM; record source/version/terms and verify patient/eye linkage before making splits.
4. **Test environment:** use a pinned VM environment with the missing packages and resolve missing test modules or document them as unavailable. Do not infer passing status from import/compile smoke tests.
5. **Claims and apps:** retain research-only language; no disease-model app is currently shown to perform real image analysis, and synthetic eye tracking is not validated on real IR footage.
6. **PhysioNet:** mBRSET cannot be used until credentialing, required training, dataset-specific DUA, and explicit access grant are complete. The denied reference-email linkage needs a truthful explanation and user-submitted replacement application.

## Required closeout order

The tracker now has dedicated final stages: (1) pull all authorized VM outputs to the private workspace and verify checksums, (2) publish only permitted source, documentation, and aggregate evidence to GitHub, then (3) stop/delete project compute as appropriate and verify no Golden Retina VM remains running. Preserve the existing disk until its contents and billing implications are understood. Never push raw/restricted datasets, row-level predictions, credentials, or weights without confirmed redistribution rights.

## Evidence files

- `mobile_models/onnx/manifest.json`
- `results/expert_sweep/per_disease_metrics.csv`
- `results/flagship_decision.json`
- `results/deepdrid_track_b_metrics.json`
- `docs/mobile_model_artifacts.md`
- `docs/latest_model_state_20261110.md`
- `docs/deepdrid_D1_audit.md` (historical; its local-image paths need rechecking)
- `results/closeout_verification_20261114.json` (prior closeout snapshot; does not prove current billing or VM state)
- `docs/dataset_candidate_screen_20261121.md` (official-source/access triage; metadata only, no image downloads)
- `docs/vm_recovery_and_cost_gate_20261121.md` (current VM/disk/runtime metadata and safe repair order)
- `scripts/cloud/vm_preflight.sh` (prepared for remote use after cost gate)


## Follow-up: staged cohort reconciliation (2026-11-21)

A VM-only scan of the current staged directories found 3,002 DDR, 2,666 DRTiD, 1,377 DeepDRiD, and 626 HYGD images. Their separate archives contain images only; labels/group linkage were reconciled against the prior project manifest. The 7,671-row prior manifest resolves all paths and its SHA-256 columns contain no exact duplicate-content groups. DeepDRiD (346 patients), DRTiD (1,333 paired-eye groups), and HYGD (244 patients) have patient-group-verified train/validation partitions with zero overlap; those partitions were used in prior model development and are not unused tests. DDR has no verified patient groups and was training-only. DRTiD's official test has already been inspected. Zhejiang has a frozen 60/20/20 patient split, but the earlier project generated DR-vs-Healthy predictions for 200 Zhejiang images from 152 patients; therefore its DR subset is exposed. See `docs/disease_scope_and_progress_20261121.md` for the disease-specific boundaries.

## Budget revision (2026-11-21)

The current campaign budget is **₺1,900 planned operating ceiling plus ₺80 protected margin (₺1,980 hard ceiling)**. This supersedes the initial ₺850/₺1,000 planning values recorded above. The 8 GPU-hour campaign limit and 7.5-hour STOP guard remain unchanged; actual billing is not verified.
