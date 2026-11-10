# Golden Retina eye-finding v3 (synthetic development)

This path builds a parameterized Blender full-face renderer, dense face/eye labels, a trainable full-frame segmenter, per-eye pupil localization, a same-identity motion-sequence renderer, a short-gap temporal filter, and a sequence evaluator. It is an engineering prototype evaluated on renderer-generated images only; it is not a human-subject or physical-camera validation. Current faces are smooth, stylized heads and are not photorealistic.

## Frozen shared-eye candidate (current sprint)

The current candidate is a compact `FaceEyeUNet` with shared `background/iris/pupil` classes; left/right estimates are assigned by image half. It was selected at epoch 24 from validation data and frozen before evaluation on a fresh 24-seed cohort and a separate 32-frame synthetic translation sequence. The checkpoint is kept locally at `results/eye_finding_ai/charmorph_training_v2/model_shared_v2b/best_checkpoint.pt` and is not committed to the repository.

Fresh seed cohort: macro Dice 0.8542 (iris 0.8086, pupil 0.8997), 24/24 pupil detections per side, mean center error 0.434/0.539 pixels at 384 model input. The 32-frame sequence had 32/32 detections per side; raw mean center error was 1.220/1.048 pixels and alpha-beta filtered mean error was 1.195/1.036 pixels. Against a train-only constant-coordinate baseline, the model reduced mean center error from 2.042/2.080 to 0.434/0.539 pixels on fresh seeds, and from 3.087/3.212 to 1.220/1.048 pixels on the sequence. This baseline has no image input and is intentionally simple.

These are synthetic same-renderer results from one base mesh, not independent-human or real-camera performance. The sequence moves iris/pupil geometry by translation; it does not simulate physical eyeball rotation. Do not present the detection rate as real-world coverage.

## Render

Run from the repository root in Blender 4.5+ with NumPy:

```sh
blender -b --python eye_finding_ai/realistic_3d_v3/render_dataset.py -- \
  --output results/eye_v3 --train 1000 --validation 150 --test 200 --samples 8
```

Each split uses disjoint generated subject seeds. The manifest includes the image/mask paths, subject id, blink labels, acquisition nuisance variables, and projected eye landmarks. The mask has 12 classes listed in `classes.json`.

## Train and evaluate

```sh
python eye_finding_ai/realistic_3d_v3/train.py \
  --manifest results/eye_v3/manifest.csv --output results/eye_v3_model \
  --epochs 20 --batch-size 8 --test-after-freeze
```

Run `audit_dataset.py` on the manifest before training. It verifies path integrity, subject-level split separation, exact duplicates, class ids, blink labels, and landmark-to-mask alignment. Training uses train subjects; validation selects checkpoints; `--test-after-freeze` evaluates the selected checkpoint once on test subjects. The report includes classwise overlap, open-eye pupil detection and localization, and pupil false positives on rendered blinks. The separate temporal sequence report measures visible-frame coverage, center error, blink behavior, and reacquisition on unseen generated identities.

## Single-frame inference

```sh
python eye_finding_ai/realistic_3d_v3/infer.py \
  --checkpoint results/eye_v3_model/best_checkpoint.pt \
  --image results/eye_v3/images/test_002815900.png \
  --output-dir results/eye_v3_inference
```

The result is explicitly named research inference. A missing/closed eye yields `uncertain_or_occluded`; it is never converted into an invented pupil point. The renderer's `ir_like` mode is a grayscale material approximation, not a physically accurate infrared sensor simulation. No result here establishes real-eye tracking performance.

## Frame service

```sh
python eye_finding_ai/realistic_3d_v3/serve.py \
  --checkpoint results/eye_v3_model/best_checkpoint.pt --host 127.0.0.1 --port 8780
```

`GET /health` reports model/device status. `POST /v1/frame` accepts one PNG/JPEG body and optional `X-Session-ID`; it returns per-eye detections and short-gap track states. It uses aspect-ratio-preserving letterboxing and returns coordinates in the original frame. Frames are processed in memory and not retained. The default listener is loopback; expose it to a network only behind suitable access controls. This is a research observation API, not a clinical or motor-control API.

## Frozen shared-eye model CLI and API

For the current three-class candidate, use the dedicated inference utility (the older `infer.py` and `serve.py` expect the original 12-class checkpoint):

```sh
python eye_finding_ai/realistic_3d_v3/infer_charmorph_eye.py \
  --checkpoint results/eye_finding_ai/charmorph_training_v2/model_shared_v2b/best_checkpoint.pt \
  --image path/to/frame.png --output-dir results/eye_finding_inference
```

It writes `observation.json` and a semantic mask. Inputs are resized square to match training, so non-square images may be geometrically distorted; side labels mean image-left/image-right, not anatomical laterality.

The loopback-only frame API is:

```sh
python eye_finding_ai/realistic_3d_v3/serve_charmorph_eye.py \
  --checkpoint results/eye_finding_ai/charmorph_training_v2/model_shared_v2b/best_checkpoint.pt \
  --host 127.0.0.1 --port 8781 --device auto
```

`GET /health` reports readiness and checkpoint hash. `POST /v1/frame` accepts a PNG/JPEG body up to 16 MiB and returns pupil-center candidates. Non-loopback bind addresses are rejected; uploaded frames are processed in memory. This is a research-only interface smoke test, not deployment or hardware validation.
