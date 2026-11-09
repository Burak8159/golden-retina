# Golden Horn Retina Eye-Finding and Tracking AI

## Scope

This is a new software system, separate from retinal-image analysis. Its input is an eye-facing camera frame or synchronized camera pair. Its output is a timestamped observation of each eye and pupil: eye identity, image-space pupil location or mask, confidence, and track state. It may output 3-D pupil coordinates only when the two cameras and their geometry are calibrated. It does not classify retinal disease, select retinal fields, or directly command motors.

The current project has no target camera hardware or Golden Horn Retina eye-camera footage. The first software-only version therefore uses a controllable synthetic image generator and, if terms allow, public eye-tracking data for comparison. Results on synthetic/public data are not evidence of performance on the eventual device.

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

## Compute recommendation

For the first compact segmentation model and synthetic-data experiments, an **RTX 3050 Ti with 4 GB VRAM is suitable** using modest image sizes, small batches, mixed precision, and gradient accumulation. It is not necessary to rent an L4 for the initial prototype. An **L4 is optional** if renderer throughput, larger batches, multiple experiments, or model size becomes the bottleneck. It will not solve the simulation-to-device gap.

No compute has been launched for this new project. The new phase has no approved spend ceiling yet; cloud work must wait for a phase-specific budget and live cost estimate. RTX use also remains a separate execution decision; the tracker does not claim that it is connected or running.

## Risks and stop rules

- If generated masks/centers are inconsistent with rendered geometry, stop training and repair the generator.
- If the model performs well only on its own synthetic distribution, report that and improve the renderer or data mix; do not claim real-camera generalization.
- If public dataset terms are incompatible or unclear, exclude that source rather than infer permission.
- If confidence is low or the eye is absent/occluded, emit `lost`/`uncertain`; do not hallucinate coordinates.
- No direct motor commands are emitted by this project. The downstream controller must independently enforce travel, speed, clearance, and emergency-stop constraints.

## Tracker cadence

At each two-hour block boundary, update completed/current blocks, evidence, issues, remaining work, and compute/spend state. Report a meaningful checkpoint after each block. The page must distinguish software-only synthetic evidence from future device validation and show “no hardware / no active compute” until those states actually change.
