# Full-frame eye-finding pipeline v2

This is a clean, independently runnable synthetic prototype. It generates full-face frames with two labeled eyes, trains a seven-class semantic segmenter, evaluates a held-out synthetic split once, and provides single-frame inference. The older crop-only pipeline is not reused as this model's evidence.

## Outputs

- `data/images/` and `data/masks/`: generated full-frame grayscale inputs and class masks.
- `data/manifest.csv`: subject ID, disjoint split, label visibility, and nuisance-condition metadata.
- `artifacts/best_checkpoint.pt`, `last_checkpoint.pt`, `training_history.json`: resumable training state and validation-selected model.
- `artifacts/locked_test_metrics.json`: one-time held-out test report, with pixel and per-eye pupil metrics.
- `artifacts/inference_smoke/`: example observation JSON and predicted mask.

## Run

On the L4 environment with NumPy, Pillow, and PyTorch installed:

```sh
python eye_finding_ai/full_frame_v2/pipeline.py --run-dir results/full_frame_v2 --epochs 24
```

For a small wiring smoke run, reduce `--train`, `--validation`, `--test`, and `--epochs` on `generate.py` and `train.py` separately; do not use smoke metrics as the final result.

## Target and limits

Inputs are complete rendered face frames, not cropped eyes. Labels are background, frame-left sclera/iris/pupil, and frame-right sclera/iris/pupil. Split identity is a unique generated subject seed assigned to one split only. The held-out set contains new generated subjects but shares the same renderer, so it measures pipeline behavior under the declared synthetic distribution only. It is not evidence of real-human, physical-camera, or clinical performance. Closed-eye frames have absent pupil labels and inference emits `uncertain_or_occluded` if a pupil is absent or confidence is low.
