# L4 checkpoint recovery — 2026-11-10

The training artifacts were recovered from the original Golden Retina L4 boot disk after the VM had been stopped. The original VM and disk were left intact. A snapshot was mounted read-only on a temporary recovery VM, and the selected run outputs were transferred through an IAP tunnel.

## Integrity and completeness

- Source snapshot: `retfound-rescue-20261110`
- Transferred archive SHA-256: `95903c2cc5b74c26efeef8829ff231f333a23f0600e0c190456c7f8843889bbb`
- Recovered run files: 100, approximately 2.0 GB
- Trained checkpoints recovered: 10 (RETFound-Green variants plus RETFound-MAE adaptation)
- Each `best.pt` loaded successfully with `torch.load(weights_only=True)` and its state dictionary was inspected.
- A private per-file SHA-256 inventory is stored beside the recovered copy at `external/recovered_l4_20261108/RECOVERY_SHA256SUMS.json`.

The private recovery copy is at `external/recovered_l4_20261108/`. It includes row-level predictions, embeddings, and split manifests and is git-ignored. Checkpoint weights also remain local because the applicable foundation-model terms do not establish redistribution rights. Publicly safe checkpoint hashes and recovery metadata are in `results/technical_dev_20261108_rerun/checkpoint_recovery_manifest_20261110.json`.

## Important correction: RETFound-MAE hash

The older `checkpoint_sha256` field in `retfound_mae_adaptation/summary.json` identifies the pretrained initialization weights. It is not the trained checkpoint hash. The recovered trained `best.pt` is 1,213,344,967 bytes, selected at epoch 11, with SHA-256 `41dd0a0fea9bd33653640fa33909338c42e664e43fe1ead2508539256ebc5330`. The summary now labels both hashes explicitly.

## Evidence boundary

The recovered adaptation runs used train/validation data only (`test_rows_loaded: 0`). Their metrics are development results, not independent external or clinical validation. The recovered weights are research artifacts, not a deployable or clinically validated system.


## Additional eye-tracking and model assets recovered

A second archive restored the project-level results and code, including the 8,167-file synthetic 3D eye-sequence output (images and masks), eye-finding/tracking scripts, training logs, the ConvNeXt baseline, the OpenEye-FM pretrained weights, and the related model outputs. Its source archive SHA-256 is `ca577c2136de3aa4d57c89baf4e56714adfce1848f8a895e2a579e03cdd1e8c3`; the transferred archive matched. The extracted private copy is `external/recovered_l4_aux_20261110/`.

The eye-crop TinyUNet checkpoint (`eye_finding_ai/artifacts/eye_pupil_unet.pt`) is 1,963,389 bytes and loaded successfully. On its 600 generated held-out images it reached foreground macro Dice 0.9854 and pupil detection 100%; this is synthetic-only. The frozen `FaceEyeUNet` checkpoint already in GitHub was byte-for-byte identical to the recovered L4 v2b checkpoint (SHA-256 `57ef89060125078d9aea7f3f8a21b380a269b3f34b20ef89d7ed32872262c2ab`).

Across both bundles, 8,267 payload files (about 3.54 GB) were restored and covered by local SHA-256 inventories. The tracked recovery manifest records all 19 recovered model/weight files and the exact archive hashes. The original dataset store was not copied; the original boot disk remains preserved.
