# Golden Retina — mBRSET post-arrival execution sequence

**Status:** prepared; waiting for Burak's arrival notice and the required access gates.
**Scope:** maximize the value of mBRSET with the existing Golden Retina software suite; no new model family.
**Separate follow-up limits:** maximum 12 hours; ₺1,400 planned ceiling; up to ₺75 protected contingency only for justified closeout; ₺1,475 absolute ceiling. Stop at the time or spend limit, whichever comes first. This budget is separate from the current initial program.

## Why this sequence preserves the dataset's value

The official mBRSET record describes 5,164 fundus images from 1,291 patients. It lists image-level ICDR DR grades, edema, quality/artifact fields, patient identifiers, laterality, and macula-/optic-disc-centered image descriptions. The record's aggregate table reports a `referable_dr` count, while the displayed metadata-field list does not establish that a `referable_dr` column exists in the distributed CSV. Therefore, inspect the actual authorized release and its data dictionary; do not silently derive a binary endpoint or assume a threshold. Likewise, the published field list does not itself establish a reliable per-image view identifier or same-eye paired views. Verify those from the release and provenance before pairing. The dataset is described as derived from BRSET, so audit overlap with any BRSET-derived model/training sources before interpreting transfer. Access requires PhysioNet credentialing, approved CITI Data or Specimens Only Research training, and a signed project DUA. These facts make mBRSET valuable for testing mobile-camera transfer, DR severity, edema, quality, and—if the actual metadata verifies same-eye views—view-combination policies. They do **not** establish that every image has a paired disc/macula view or that a particular model will perform well.

The best use is two explicitly separate tracks:

1. **Frozen external evaluation:** run the final existing model artifacts and thresholds without adapting to mBRSET. This is the cleanest evidence of transfer to the mobile-camera cohort. Report all prespecified tasks, errors, uncertainty, and missing labels. Do not tune or select a winner using this full-cohort result.
2. **mBRSET development/adaptation:** only after the frozen evaluation outputs and protocol are sealed, use patient-disjoint train/validation/test partitions to adapt compatible existing heads, calibrate, or compare one-view/two-view use. These results are development estimates for the adapted versions, not independent external validation. Keep the locked test partition unseen until the candidate and thresholds are frozen.

## Arrival and authorization gates — before the 12-hour run

When Burak says mBRSET has arrived:

1. **Confirm what “arrived” means.** Check whether PhysioNet has granted access to the requesting account and which exact version is available. A message or file from the dataset creator is not itself an access grant.
2. **Clear the PhysioNet gates:** approved credentialing, accepted CITI training, the personally accepted mBRSET DUA, and actual account access. PhysioNet access is individual; never copy the account credentials or distribute the dataset to teammates.
3. **Finish the current initial-program closeout first:** stop training, pull and checksum permitted VM artifacts, publish only permitted code/docs/aggregates, stop the Golden Retina VM, and verify it is terminated. The live tracker is the source of truth for this campaign's current stage numbering.
4. **Reconcile the separate mBRSET budget:** obtain the current billing state, include the existing disk, storage, network, tax and currency effects, and confirm the 12-hour plan fits ₺1,400 planned. The ₺75 margin is not a target and is used only for justified closeout. If the total cannot be verified within the ₺1,475 hard cap, do not launch.
5. **Confirm VM readiness without destroying old work:** preserve the existing boot disk, run the remote preflight, verify the L4 and free disk space, and ensure output/checkpoint directories exist. Dataset download and extraction happen on the VM only.
6. **Preselect the model set before any mBRSET predictions are viewed:** use only prior non-mBRSET development evidence to name the frozen models and up to two adaptation candidates. If a candidate is chosen after seeing mBRSET results, label the resulting analysis exploratory and do not call it confirmatory transfer evidence.

If any gate is incomplete, record it in the tracker and continue only with non-data preparation. Do not accept terms or submit credentials on anyone's behalf.

## Timed execution — 12 hours maximum

| Time | Work | Evidence required / decision |
|---:|---|---|
| 0:00–0:30 | Start campaign clock and cost ledger; validate access, version, private VM path, output space, timer and stop guard. | Access/accessibility evidence; clock and spend baseline; no data on Mac. Abort if a gate or cap is unclear. |
| 0:30–1:00 | Download the authorized version directly to the VM; record official source, version, file list, archive hashes and transfer result. | Checksummed source manifest; raw files remain in access-controlled storage. |
| 1:00–2:00 | Schema and label audit: image-to-row joins, patient IDs, laterality, DR grade, referable DR, edema, quality/gradable fields, missingness, view/camera fields. | Counts at image, eye and patient levels; label-mapping table; discrepancies and unresolved fields. Do not guess mappings. |
| 2:00–2:45 | Integrity, duplicate and overlap audit against available development data and manifests. | Exact-hash and near-duplicate report; patient/eye grouping assessment; possible BRSET/mBRSET overlap flagged. No raw identifiers in GitHub. |
| 2:45–3:30 | Freeze analysis plan, inclusion/exclusion, patient-level partitions, primary metrics and aggregation rules before looking at model results. | Timestamped plan and split manifest on VM. If sample support is too small, revise the estimand before evaluation or mark the task descriptive. |
| 3:30–4:45 | Frozen external evaluation of compatible existing models, with preprocessing, checkpoint hashes and thresholds fixed from prior development data. | Per-image/per-eye predictions stored privately; prespecified DR, referable DR, edema and quality metrics. This first pass is the clean external-transfer result. |
| 4:45–5:45 | Error and subgroup audit for the frozen models; review false-negative/false-positive cases and image-quality strata without changing the model. | Patient-bootstrap confidence intervals; per-task confusion matrices, AUROC/AUPRC, sensitivity/specificity, calibration, coverage and failure review. Patient-level grouping for uncertainty. |
| 5:45–6:45 | View/quality analysis if mappings support it: compare single-view with same-eye paired views and quality-aware handling. | Verified pairing only; report missing/invalid pairs. If same-eye pairing or acquisition order is absent, skip fusion/CCG claims and use this hour for supported quality/error analyses. |
| 6:45–8:45 | Adaptation experiment using only existing compatible model families: patient-grouped training/validation/test; adapt at most two preselected heads or use a single candidate if compute/data support is weak. | Resumable checkpoint, full config, fixed seeds, train/validation logs; no access to the locked test while selecting. Stop an unproductive run instead of cycling variants. |
| 8:45–10:00 | Freeze adapted candidate and validation-derived threshold; evaluate once on its patient-held-out test partition. | Patient-bootstrap uncertainty, per-task results and errors. Mark this as within-mBRSET development/test evidence, not an independent external cohort. |
| 10:00–10:45 | Check calibration, subgroup support, model size/latency, inference contract, and whether any mobile/portable preprocessing mismatch invalidates deployment. | Updated model card and integration note; exploratory/underpowered subgroups explicitly marked. No clinical-diagnosis claim. |
| 10:45–12:00 | Closeout: stop jobs, create remote hash inventory, review DUA and dataset terms for each output, pull only authorized artifacts, update tracker/docs, push only permitted code and aggregates, stop VM and verify termination. | Checksums and safe transfer record; public repo excludes raw images, row-level predictions, credentials, and any checkpoint/derived artifact whose redistribution rights are unclear. Record settled-vs-estimated spend and residual disk state. |

The timeline is a ceiling and triage plan, not a promise that every branch will be scientifically supportable. If data integrity, label semantics, or sample support fails, stop that branch and document it; do not fill time with post-hoc tuning.

## Prespecified endpoints and analysis

### Primary tasks

- **Referable DR:** binary label as supplied, subject to metadata verification. Report AUROC and AUPRC as threshold-free measures; at thresholds frozen before the external evaluation, report sensitivity, specificity, PPV/NPV, coverage and confusion counts.
- **Referable-DR schema gate:** first determine whether the released row-level file contains an explicit, documented binary label. If absent, derive it only from a documented grading protocol with a prespecified mapping and validate the mapping against official counts where possible. If semantics cannot be reconciled, omit this endpoint and report only supported ordinal grading.
- **DR severity:** ordinal grades, subject to exact `final_icdr` semantics and missingness. Report quadratic-weighted kappa, ordinal error and per-grade support; do not reduce it to ordinary accuracy alone.
- **Macular edema:** binary task using only rows with an observed label. Report missing-label counts and the same threshold-free and fixed-threshold metrics.
- **Image quality:** use only documented quality labels. Separate ungradable/quality failure from disease absence; report coverage and errors by quality strata.
- **Multi-view/CCG:** evaluate only if patient, eye, laterality and view fields prove that images are multiple views of the same eye. Compare single view with paired-view use using patient-held-out data. Do not claim acquisition savings without actual acquisition sequence/cost evidence.
- **View-map gate:** macula-/disc-centered descriptions are not sufficient by themselves to pair rows. Require an explicit, reliable view field or verified source documentation, plus a same-patient and same-eye mapping. Otherwise mark paired-view fusion unsupported.

Use patient-level grouping for every split and patient-level bootstrap for confidence intervals. Keep both eyes and all views of a patient together. If multiple views are aggregated into an eye-level score, specify the aggregation rule before evaluation. Report image-level and eye/patient-level estimands separately where the labels support them. Do not treat all 5,164 images as independent observations.

Age, sex, diabetes duration, treatment and other clinical metadata may support missingness and subgroup analyses only where allowed by the DUA and sufficiently supported. Do not use clinical metadata as model inputs in the primary image-only test; that would change the task and could leak labels or care pathways. Any metadata-input experiment must be separately prespecified and clearly labeled.

### Adaptation guardrails

- No new architecture family. Reuse only the existing, compatible Golden Retina model families and preprocessing contracts.
- Freeze source-trained checkpoints, preprocessing and operating thresholds before the external pass. Save their hashes and code revision.
- Any adaptation uses patient-disjoint partitions; tuning/selection uses training/validation only. The held-out mBRSET partition is opened once after the candidate is frozen.
- Do not use full-cohort mBRSET results to select a model and then report those same results as independent test performance.
- Compare adapted performance to frozen-source performance on the same held-out patients. Report gains and regressions per task, not only a macro average.
- If the official dataset release does not provide enough independent patient support for a stable test, use grouped cross-validation for development and label results internal; do not imply independent confirmation.

## Data handling, artifacts, and publication

Keep mBRSET images, labels, row-level predictions, identifiers, and any restricted derived files in the authorized private environment. No download, extraction, or row-level data on this Mac. Do not share the PhysioNet account or dataset access with the team. The DUA governs outputs as well as raw inputs: before pulling a checkpoint, feature cache, prediction file, or detailed error list, check whether the terms permit that transfer. If not explicit, keep it on authorized storage and seek the data owner's or PhysioNet's guidance.

The repository may receive only reviewed, permitted items such as source code, a schema without patient rows, run configuration, safe hashes, and aggregate metrics. Update the tracker with actual progress, access state, spent/remaining time and spend, model hashes, split IDs only if permitted, and closeout evidence. Never represent a development estimate as independent validation or clinical utility.

## Official references

- [PhysioNet mBRSET v1.0 record](https://physionet.org/content/mbrset/1.0/)
- [PhysioNet credentialed-data access FAQ](https://physionet.org/about/faqs/)
- [mBRSET required CITI training](https://physionet.org/content/mbrset/view-required-training/1.0/)
- [mBRSET credentialed data license](https://physionet.org/content/mbrset/view-license/1.0/)
- [mBRSET data use agreement](https://physionet.org/content/mbrset/view-dua/1.0/)
