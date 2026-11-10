# Golden Retina eye finding and tracking

This is a perception subsystem separate from retinal disease analysis. It estimates pupil masks and image-space centers from eye-facing frames and can apply a short-gap temporal filter. It does not issue motor commands.

## Current candidate

The latest frozen prototype is [`realistic_3d_v3`](realistic_3d_v3/README.md), using a shared three-class `FaceEyeUNet` trained on a CC0-based synthetic face/eye renderer. The selected checkpoint is available at `results/eye_finding_ai/charmorph_training_v2/model_shared_v2b/best_checkpoint.pt`; it was selected at epoch 24 using validation macro Dice and has SHA-256 `57ef89060125078d9aea7f3f8a21b380a269b3f34b20ef89d7ed32872262c2ab`.

On a fresh 24-seed synthetic cohort from the same base mesh/renderer, macro Dice was 0.8542 (iris 0.8086, pupil 0.8997), with pupil detections on 24/24 frames per image side. On a separate 32-frame synthetic eye-geometry translation sequence, detection was 32/32 per side. A local, loopback-only inference API was smoke-checked. The model beat a train-only constant-coordinate baseline on these synthetic checks.

This is **not** a real-person, camera, device, or clinical validation. The face is one synthetic base identity; sequence motion translates iris/pupil geometry rather than rotating a physical eyeball. Confidence is not calibrated. Do not connect it to a camera-driven controller without a new, locked validation on representative human and device data.

`realistic_3d_v3/` is the current frozen research candidate. The top-level scripts and `full_frame_v2/` preserve earlier experiments, not separate promoted candidates. Generated images and raw run bundles stay outside Git.

## Additional compact eye-crop model

The recovered `artifacts/eye_pupil_unet.pt` is a separate TinyUNet trained on procedurally generated grayscale eye crops. It is not the frozen `realistic_3d_v3` candidate, and its metrics are not directly comparable because the renderer and split differ. On its 600-image synthetic held-out split it reached 0.9854 foreground macro Dice and 100% pupil detection; mean pupil-center error was 0.179 px (p95 0.483 px). On 50 synthetic sequences of 50 frames, it tracked all 2,050 visible frames and predicted through all 450 synthetic blink frames. These are renderer-only results, not human-eye or camera results. The L4 runtime profile (2.34 ms median, 2.47 ms p95) is hardware-specific.

Run one already-cropped eye image with:

```sh
python eye_finding_ai/predict_one.py \
  --checkpoint eye_finding_ai/artifacts/eye_pupil_unet.pt \
  --image path/to/eye_crop.png
```

Aggregate metrics and tracking/runtime summaries are in `eye_finding_ai/artifacts/`. The synthetic split manifest and generated run bundle are not published.

## Run inference

From the repository root:

```sh
python eye_finding_ai/realistic_3d_v3/infer_charmorph_eye.py \
  --checkpoint results/eye_finding_ai/charmorph_training_v2/model_shared_v2b/best_checkpoint.pt \
  --image path/to/frame.png --output-dir results/eye_finding_inference
```

For the loopback-only research API, see [`realistic_3d_v3/README.md`](realistic_3d_v3/README.md). The API binds to `127.0.0.1` by default and must not be exposed as a clinical or motor-control service.

## Reproducibility and resources

Training, rendering, audit, and evaluation scripts are in `realistic_3d_v3/`. The closeout summary is [`results/eye_finding_ai/frozen_v2_model_baseline_comparison.json`](../results/eye_finding_ai/frozen_v2_model_baseline_comparison.json); the frozen run card and checkpoint hashes are in `results/eye_finding_ai/charmorph_training_v2/frozen_candidate_run_card.json`. The Google Cloud L4 used for this sprint is stopped. Tracker spend remains an estimate, not a settled invoice.
