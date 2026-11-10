# Golden Horn Retina Eye-Finding and Tracking AI

## Scope

This is a new software system, separate from retinal-image analysis. Its input is an eye-facing camera frame or synchronized camera pair. Its output is a timestamped observation of each eye and pupil: eye identity, image-space pupil location or mask, confidence, and track state. It may output 3-D pupil coordinates only when the two cameras and their geometry are calibrated. It does not classify retinal disease, select retinal fields, or directly command motors.

The current project has no target camera hardware or Golden Horn Retina eye-camera footage. The first software-only version therefore uses a controllable synthetic image generator and, if terms allow, public eye-tracking data for comparison. Results on synthetic/public data are not evidence of performance on the eventual device.

## Authorized cloud sprint (2026-11-09)

- Use only the dedicated Golden Retina Cloud project and `golden-retina-l4-01` (`g2-standard-4`, one L4, `us-west1-a`). RTX is not used.
- Work window: 7 hours; reserve 30 minutes for artifact transfer and VM shutdown. A Compute Engine maximum run duration of 7 hours 30 minutes is configured with stop as the termination action.
- Budget: ₺1,900 planned ceiling, ₺80 protected margin, ₺1,980 absolute ceiling. Google Cloud's listed on-demand rate for `g2-standard-4` is $0.706832276/hour; at the 2026-11-09 USD/TRY rate of 50.178, 7.5 hours is approximately ₺267 for compute alone. This estimate excludes persistent disk, network, tax, and billing adjustments; Cloud Billing API is not enabled, so settled project spend is not available in this tracker.
- This capped sprint targets a reproducible synthetic eye renderer, compact pupil segmentation/localization, and temporal tracking feasibility. It does not complete the 24-block, 48-hour roadmap, and it cannot validate a physical eye camera or real-person performance.
- Do not enable additional cloud APIs or alter unrelated projects/resources. Stop the VM as soon as compute is no longer needed, then verify its stopped state and pull/checksum only this prototype's artifacts.

## Proposed model and system

1. A parameterized renderer generates eye-region images and short sequences with exact labels: pupil/iris masks, pupil center, eye side, gaze/pose parameters, blink and occlusion state, camera intrinsics/extrinsics, and timestamps.
2. A compact pupil segmentation/localization model predicts masks or landmarks plus confidence.
3. Geometric post-processing extracts a pupil center. A temporal tracker maintains left/right identity across frames, blinks, and short occlusions, and explicitly reports `uncertain` or `lost` rather than fabricating a location.
4. For synthetic stereo frames, calibrated geometry triangulates 3-D position. At runtime, 3-D is emitted only with real calibration supplied by the future camera system.
5. A versioned observation interface passes results to a separate motion controller. The perception model itself emits no motor commands.

Use public data as a benchmark or initialization source only after checking the exact current terms. LPW's official page specifies non-commercial scientific use. NVGaze and OpenEDS are useful near-eye IR research references, but their camera arrangements and appearance differ from the intended retinal headset. Generated images and public datasets must remain separately tagged in manifests.

## 24 two-hour development blocks

Nominal engineering effort: **48 focused hours** (24 blocks × 2 hours), excluding future hardware procurement, installation, real-camera data collection, human-subject procedures, and device-domain validation.

| Block | Work | Completion evidence / gate |
|---:|---|---|
| 1 | Freeze scope, repository boundary, and software interfaces | Separate package and written non-goals; no changes to retinal diagnosis model |
| 2 | Define observation schema and coordinate conventions | Versioned 2-D observation schema; stereo fields are optional and calibration-gated |
| 3 | Specify synthetic eye/camera parameter envelope | Explicit ranges for eye appearance, pupil, gaze, illumination, occlusion, camera pose, and motion; ranges labeled as assumptions |
| 4 | Screen accessible public eye datasets and exact terms | Source/label/terms matrix; no dataset used until exact access terms are checked |
| 5 | Implement static eye-image generator | Deterministic image samples with pixel masks, pupil-center labels, camera metadata, and reproducible seeds |
| 6 | Add temporal sequence generation | Ground-truth trajectories with blinks, saccades, smooth motion, occlusion, and dropped-frame events |
| 7 | Audit generator labels and parameter coverage | Automated geometry/label consistency report and visual sample sheet; reject invalid renders |
| 8 | Create leakage-safe dataset manifests | Splits grouped by generated subject, eye, scene seed, and camera configuration |
| 9 | Establish classical vision baseline | Threshold/contour/ellipse baseline with per-condition localization and failure metrics |
| 10 | Select compact learned segmentation model | Candidate and input resolution chosen on training/validation only; memory/latency budget recorded |
| 11 | Build training and checkpoint pipeline | Reproducible config, deterministic manifests, checkpoint/resume support, loss curves |
| 12 | Train first synthetic-data candidate | Best checkpoint selected on validation; test split unopened |
| 13 | Evaluate frame-level detection | Pixel/landmark error, detection rate, false detection rate, and confidence calibration by condition |
| 14 | Implement temporal tracker and eye identity | Track continuity, identity-switch, blink recovery, and lost-track behavior report |
| 15 | Implement simulated stereo geometry | Calibrated synthetic triangulation, reprojection checks, uncertainty, and invalid-depth rejection |
| 16 | Add confidence and out-of-distribution behavior | Explicit unknown/lost state; low-confidence observations cannot appear as valid tracks |
| 17 | Run domain-randomization ablation | Matched comparison with/without each randomization group; preserve negative findings |
| 18 | Evaluate on eligible public IR eye data | Subject-held-out benchmark, only if exact dataset terms and label mappings permit |
| 19 | Adapt using combined permitted development data | Keep sources tagged; select on validation only; no mixing test data into training |
| 20 | Optimize inference runtime | Measure latency, memory, and throughput on the selected development hardware |
| 21 | Finalize integration contract | API emits eye/pupil observations and status only; controller and fixation interfaces documented separately |
| 22 | Stress-test simulated failures | Occlusion, out-of-frame eye, blur, lighting shift, camera mismatch, timestamp gaps, and recovery traces |
| 23 | Freeze model and run locked evaluation | Freeze weights/config; one held-out evaluation with subject-level uncertainty and all condition breakdowns |
| 24 | Package software and hardware-readiness handoff | Versioned package, model card, manifests, evaluation report, and exact future camera-calibration/data requirements |

## Success criteria

Final numeric tolerances must be set before the locked evaluation and justified by the eventual optical capture envelope. Do not invent a millimeter or pixel pass threshold before the camera geometry and optical tolerance are known. Report, at minimum:

- 2-D pupil-center error and mask overlap;
- 3-D error only for calibrated stereo;
- detection and false-detection rates by condition;
- track continuity, left/right identity switches, blink/occlusion recovery, and reacquisition time;
- confidence calibration and lost/uncertain-state correctness;
- inference latency, memory, and throughput on the intended runtime hardware;
- subject/scene/camera-grouped uncertainty and all failures.

The synthetic phase passes only as a **software feasibility demonstration**. Device readiness requires actual sensor footage, verified camera calibration, and a new locked evaluation on held-out people/sessions/device conditions.

## Compute and budget status

The current authorized run uses only the dedicated Golden Retina L4 VM; the RTX remains unused. The approved ceiling is **₺1,900 planned plus ₺80 protected overhead, for a ₺1,980 absolute cap**. The tracker estimates spend from actual VM runtime at the configured hourly rate; final billing can differ because of storage, networking, and tax. The VM has a 450-minute automatic-stop limit, and artifacts must be pulled and checked before shutdown.

At the 2026-11-09 23:49 UTC checkpoint, the failed stylized renderer output (780/1,620) remained preserved and excluded. The CC0 CharMorph/Vitruvian base has 37,436 vertices and 153 shape keys but remains visibly synthetic and represents one base identity. Seeded label geometry probes passed for V6 (8/8), V9 (5/5), and V10 (7/7); V7/V8 age-morph probes failed from camera cropping and were excluded. A 64-case 512×512 cohort passed render label-presence QA and preprocessing accepted 64/64, split by seed into 45/10/9 train/validation/test. The first five-class U-Net ran 15 epochs on L4. Best validation foreground macro Dice was 0.557; the exposed morph-seed diagnostic test yielded 0.542, with pupil-pixel presence 0/9 on image-left and 9/9 image-right. This failure is recorded; that test is now development-only. Candidate v2 merges iris and pupil semantics across eyes, estimates side by image half, and was selected at epoch 24 on validation macro Dice only (0.8532). Its checkpoint was frozen and hashed before a fresh 24-seed run (seed 816170). That fresh-seed evaluation accepted 24/24 frames and reported foreground macro Dice 0.8542 (iris Dice 0.8086; pupil Dice 0.8997), pupil pixel-presence 24/24 for both image sides, and mean center error 0.434/0.539 px at 384×384 model input. This is seed-held-out evaluation on the same synthetic base mesh and renderer; it is not person-held-out or camera validation. Gaze remains geometry translation, not physical eyeball rotation. No independent human identity, real camera, blink, or demographic validation exists. A separate 32-frame fixed-face/fixed-camera translation sequence was run with the same frozen checkpoint: model pupil presence was 32/32 per side; mean center error was 1.220/1.048 model-input pixels before filtering and 1.195/1.036 pixels after the alpha-beta filter. This is controlled geometric translation, not physical eyeball rotation, blink recovery, or head/camera tracking.


## Risks and stop rules

- If generated masks/centers are inconsistent with rendered geometry, stop training and repair the generator.
- If the model performs well only on its own synthetic distribution, report that and improve the renderer or data mix; do not claim real-camera generalization.
- If public dataset terms are incompatible or unclear, exclude that source rather than infer permission.
- If confidence is low or the eye is absent/occluded, emit `lost`/`uncertain`; do not hallucinate coordinates.
- No direct motor commands are emitted by this project. The downstream controller must independently enforce travel, speed, clearance, and emergency-stop constraints.

## Tracker cadence

At each two-hour block boundary, update completed/current blocks, evidence, issues, remaining work, and compute/spend state. Report a meaningful checkpoint after each block. The page must distinguish software-only synthetic evidence from future device validation and show “no hardware / no active compute” until those states actually change.


## Frozen-candidate comparison and service package (2026-11-10)

The shared-eye `FaceEyeUNet` candidate was selected at epoch 24 using the validation split only and frozen before fresh-seed evaluation. The checkpoint SHA-256 is `57ef89060125078d9aea7f3f8a21b380a269b3f34b20ef89d7ed32872262c2ab`; the training manifest SHA-256 is `e152061a180c4a67ab38282daeb0e253edf4e3c02e132cad20c909e2e45b0104`.

A train-only constant-coordinate pupil-center baseline (estimated from training masks, no image input) was evaluated on the same synthetic probes. On the fresh 24-seed cohort, mean center error was 2.042 px left / 2.080 px right for the baseline versus 0.434 / 0.539 px for the model (error reductions 78.76% / 74.07%). On the 32-frame controlled translation sequence, baseline error was 3.087 / 3.212 px versus 1.220 / 1.048 px for raw model measurements (reductions 60.50% / 67.36%); the alpha-beta filter yielded 1.195 / 1.036 px. This is useful evidence that the segmenter responds to rendered image content instead of merely returning the average training location. It is still a same-renderer, single-base-mesh comparison and does not establish transfer to another person, camera, or environment.

Added `eye_finding_ai/realistic_3d_v3/infer_charmorph_eye.py` for local single-frame inference and `serve_charmorph_eye.py` for a loopback-only research API. The local CPU API smoke check returned a ready health response and pupil candidates for both image halves on one held-out synthetic frame. Output is explicitly marked research-only. Non-square inputs are square-resized, confidence is uncalibrated, and neither tool is suitable for driving hardware.

Machine-readable comparison: `results/eye_finding_ai/frozen_v2_model_baseline_comparison.json`. Local inference smoke output: `/private/tmp/golden_retina_v2_inference_smoke/observation.json`. Closeout archive `results/eye_finding_ai/charmorph_closeout_bundle.tar.gz` has SHA-256 `a320845fb49dc5e1ab38053f67732a02ec24e932ef80ca1935e220f1997bc423`, verified against the transferred archive.

Operational closeout: budget is ₺1,900 planned plus ₺80 reserve, ₺1,980 hard ceiling. At this update the estimated dedicated-VM compute spend is ₺203.19, calculated from elapsed time and the configured hourly estimate, not from a settled invoice. Stop only the dedicated Golden Retina VM after local artifact checks; keep the localhost tracker available.


## Sprint closeout (2026-11-10 00:31:44 UTC)

The 12-block sprint is closed. The frozen model package and outputs are local, the inference API smoke check passed on a held-out synthetic render, the train-only coordinate comparison is recorded, and `golden-retina-l4-01` was stopped and verified `TERMINATED`. The local tracker remains available at `http://127.0.0.1:8770/`. Runtime-based compute estimate at shutdown: ₺204.63; this is not the settled invoice and does not include possible disk, network, or tax charges. Budget remains ₺1,900 planned + ₺80 protected overhead = ₺1,980 hard cap.

The closeout file `results/eye_finding_ai/closeout_integrity.json` records local artifact SHA-256 hashes and file counts. No GitHub push was performed.
