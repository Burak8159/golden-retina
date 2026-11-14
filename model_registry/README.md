# On-device model registry contract

Golden Retina is designed for inference on the phone. The registry API is only a
model distribution/update channel; it does not accept patient images and does
not run inference. Both apps can fetch a manifest over HTTPS, download ONNX
artifacts, verify SHA-256, and ask ONNX Runtime to open the model before marking
the bundle locally available. The current demo flow is not wired to image
preprocessing or inference yet. A weight/version update that preserves this
schema does not require an app update.

## Contract

Host `manifest.example.json` as `GET <registry-base>/v1/manifest.json` over HTTPS.
The production manifest must supply `version`, `artifact_url`, `sha256`,
`input`, and `outputs` for each `ready` model. `artifact_url` must use HTTPS and
the digest must be SHA-256 of the exact ONNX file. The digest is checked against
the manifest delivered over TLS; this prototype does not digitally sign the
manifest, so it is not a production trust chain. Keep previous installed files
until the new file passes hash validation and the runtime opens it. Apps reject
unknown schema versions, untrusted URL schemes, and hash mismatches.

The tracker uses the stable ID `ir_eye_tracker`. Disease experts use dynamic
IDs such as `retinal_classifier_dr` or `retinal_classifier_amd`; the apps list
and fetch IDs from the manifest rather than hard-coding a disease roster. Model
versions, artifact URLs, hashes, task labels, thresholds, input sizes, and tensor
contracts are manifest-owned. Keep preprocessing and output semantics versioned.
If a model changes its tensor contract, increment `schema_version` and release
compatible app support before publishing it.

## Current release state

The checked-in manifest is an example/status manifest only. It intentionally
does not contain a downloadable model URL: the disease models and eye tracker
currently exist as PyTorch research checkpoints, not verified ONNX mobile
bundles. The apps have registry clients and ONNX Runtime dependencies, but no
model bundle can be installed until export and a stable HTTPS registry URL are
available. Inference is not wired into the simulated scan flow.

The present eye tracker was trained on synthetic rendered imagery and has no
validated physical IR-camera input contract. The current disease sweep uses a
set of per-disease experts and has image-level RFMiD evidence; it is not a
single clinically validated classifier. Do not label either as a clinical
diagnosis in the app.

## Model release workflow

1. Choose a supported disease-head set and an IR-compatible tracker checkpoint.
2. Export the selected checkpoints to ONNX with `tools/model_registry/export_onnx.py`.
3. Verify outputs against the source PyTorch implementation on a fixed local
   comparison set; record latency, peak memory, model size, and quantization
   effects for each phone class.
4. Sign and host artifacts plus the manifest on an HTTPS static object/CDN
   endpoint. The registry distributes weights; it does not process images.
5. Install from each app, verify hashes, open the models with ONNX Runtime, and
   only then enable a real acquisition path. The goggles/Bluetooth demo remains
   separate until hardware and its image protocol exist.
