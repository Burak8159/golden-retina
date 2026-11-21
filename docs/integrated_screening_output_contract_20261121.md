# Integrated screening output contract

**Status:** research-software contract; not a clinical specification
**Schema:** [`mobile_models/schemas/per_eye_result.schema.json`](../mobile_models/schemas/per_eye_result.schema.json)
**Model registry:** [`mobile_models/onnx/manifest.json`](../mobile_models/onnx/manifest.json)

## Findings from the integration audit

- The registry contains 18 ONNX artifacts: one synthetic-domain eye-localization model and 17 RFMiD heads. All 18 local artifact files matched the SHA-256 values in the registry at audit time.
- Disease models are distributed through the same HTTPS GitHub registry for iOS and Android. The apps execute downloaded models locally. They do not upload selected fundus photos for model inference.
- The registry's `20261118-r5` bundle version and Android's `20261114-r3` bundled fallback label refer to different release mechanisms. The fallback is only the bundled synthetic eye-localization model; downloaded disease models use each registry entry's individual version and SHA-256. This is a documentation/version-label distinction, not evidence that the disease model sets differ.
- Disease model inputs use RGB, square bilinear resize to 320x320 and ImageNet normalization. Both app inference paths use equivalent channel normalization and CHW tensor construction. The eye-localization model has its own 384x384 RGB, unnormalized input contract.
- The four post-hoc shortlist heads are DR, AMD, myopia and BRVO. They remain research classifiers trained/evaluated on RFMiD image splits; patient independence is unverified and their 0.50 cutoffs are not clinical thresholds. Other registry heads include findings (e.g. drusen and optic-disc findings), not only disease labels. Rare heads have very small positive support.
- Image-quality assessment is not implemented as a validated model. The schema therefore requires `not_assessed` until a supported quality model exists; disease scores must never be repurposed as image-quality scores.
- The demo's fixed examples remain illustrative sample values. They are not inference outputs and must not be serialized as analysis results.

## Output rules

1. Every result is tied to one eye (`left`, `right`, or `unknown`) and the exact registry version. Never swap left/right labels to match screen position.
2. Every scored output records its model ID, per-model version, SHA-256, architecture, source dataset, output kind and evidence scope.
3. `raw_score` is a bounded model output only. It is not called a probability, patient risk, confidence of disease, or diagnosis. `research_flag` is optional and only means comparison with that model's documented research cutoff.
4. A dataset condition label and a retinal image finding are separate output kinds and must be presented separately. For example, optic-disc cupping is not a glaucoma diagnosis, and drusen is not an AMD diagnosis.
5. Missing, failed, or unsupported heads are represented with an explicit status and `raw_score: null`; they are never silently converted to zero or a negative result.
6. Image quality is independent. If quality is not measured, show `not_assessed`; do not infer that an image is gradable from disease-head scores.
7. An eye with no usable image is not a negative screen. Partial model installation produces `partial` status and lists each unavailable head.
8. The research API is local to the device. A registry URL distributes model files and metadata; it is not a remote inference service. Registry updates may change model versions without changing the app, but each result remains reproducible only when its model hash and preprocessing are retained.

## Capability boundaries

The current implementation does not establish real goggle capture, Bluetooth acquisition, real-person eye tracking, infrared-camera validity, functional-vision measurements, calibrated clinical risk, clinical utility, or independent external disease performance. The synthetic eye-localization artifact must not control motors or guide a real examination. These remain release-blocking capability gaps, not fields to fill with synthetic or illustrative values.

## Release gate

Before any user-facing pilot, the product owner must choose the supported disease set using prespecified patient-level and independent-source evidence; remove or clearly segregate unsupported image findings; validate image quality and real capture; define calibration and action thresholds; and complete appropriate clinical, privacy, and regulatory review. This contract prevents data-shape ambiguity but does not satisfy those evidence requirements.
