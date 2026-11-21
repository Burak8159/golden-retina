# Golden Retina mobile product architecture

**Status:** implementation architecture for the current research app; not a clinical, regulatory, device, or release approval.
**Inference boundary:** model execution is on-device. No inference server is used. The public model registry is a version/distribution source only.

## 1. Product capability boundary

| Capability | Current state | User-visible behavior |
|---|---|---|
| Guided scan | Simulated | Persistent Demo context; no images or health record are created. A displayed sample is explicitly marked as sample data. |
| Image classifier | Installed research ONNX models | The selected image stays on the phone; show raw score, model ID/version, dataset label semantics and development-only limitations. Never call the score a probability, risk, diagnosis, or reassurance. |
| Eye localization | Synthetic-render research model | Available only in the research lab for uploaded/selected test frames. Never connect it to goggles, IR acquisition, or motor control. |
| Goggle connection/capture | Unavailable | Keep controls disabled with a concise explanation. Do not request Bluetooth permissions or simulate actual pairing. Unlock only with versioned firmware/SDK and acquisition/safe-stop evidence. |
| Clinical output, quality gate, history, reports | Unsupported or not implemented | Use explicit unavailable/not assessed states. Do not substitute zero, a negative finding, or invented session data. |

## 2. Data and execution flow

1. The user selects a local image and explicitly assigns it to Left or Right. The image picker and analysis screen do not upload or persist the image.
2. The app reads the signed-by-hash public registry over HTTPS. Registry metadata identifies task/label type, model version, expected byte size, SHA-256, architecture, preprocessing, input/output contract and evidence scope.
3. The app downloads only a user-requested model to private app storage, verifies declared size and SHA-256 before marking it installed, and leaves the previously working version intact until the new file passes verification.
4. ONNX Runtime executes locally. Each model output is shape-, finite-value- and range-checked. A malformed output is an inference error, never clamped into an apparently valid score.
5. The result is stored only in transient view state for this analysis. No backend, account, analytics upload, or examination history is implied.

Registry signatures are not currently specified. Hash checking detects corruption against the fetched manifest, but does not by itself authenticate a maliciously modified manifest. Production distribution therefore requires a pinned signing key or app-store signed registry policy before relying on remote model updates.

## 3. Result semantics

The shared `mobile_models/schemas/per_eye_result.schema.json` is the contract. `eye_side` is `left`, `right`, or `unknown`; it must not be guessed. Each output carries an `output_kind`: `dataset_condition_label` or `retinal_image_finding`. Finding labels such as optic-disc cupping or tessellation must not be rendered as diagnoses. An uninstalled model is unavailable; missing quality evidence is `not_assessed`; neither means zero or normal.

The interface may show a percentage-formatted raw score because the model emits a scalar in `[0,1]`, but it must state adjacent to results that this is an uncalibrated research score. Each scored row must also show its model version and RFMiD provenance; missing installed models must be identified as not assessed, not silently omitted as if their scores were zero. Group summaries can show the largest raw score among outputs of the same semantic group and same eye, explicitly named “highest raw score”; a group maximum is not a patient risk estimate. Never combine left and right scores or swap their physical label/order.

Dataset mappings remain faithful to dataset label names. In particular, the RFMiD category “Myopia” is not silently upgraded to pathological myopia; optic-disc cupping is not called glaucoma; and unsupported glaucoma, image-quality, lesion, or “any retinal disease” outputs remain absent.

## 4. Navigation and state model

Five top-level destinations remain: Home, Screening, History, Device and Settings. Screening contains the distinct Guided Demo and Research Image Analysis paths. Model management is reached from Settings and explains local storage, versions, hashes, download status, deletion and retry. How-to material is contextual help.

The future physical exam is an explicit state machine: `unavailable → ready → button_started → right_acquiring → right_complete → left_acquiring → left_complete → processing → result | partial | recoverable_error | safe_stop`. Only the simulated demo can currently transition through a visual analogue. No real state is enabled until device protocol and stop/fault semantics are supplied.

Recovery requirements:

- Cancel image selection without changing the selected eye or showing stale results.
- On image decode/model failure, clear that eye's previous result and identify the failed operation; offer retry.
- Interrupted local inference can be restarted; do not present stale partial outputs as complete.
- On insufficient storage or failed model checksum, do not activate the model; preserve the prior valid model and give a recovery action.
- Device unavailable, no image, no installed model, unsupported task and model error are distinct states.

## 5. Privacy, accessibility and content

No account is required. Images are processed on-device and are not uploaded. Model files are fetched from a public HTTPS registry. The app must distinguish a Demo sample from an actual local model result at every entry and result screen. Actual model results are not sample values; sample results must never masquerade as actual inference.

Use platform-native navigation, dynamic text, semantic labels, screen-reader order, 44 pt iOS / 48 dp Android target sizing, and status text that does not rely on color alone. Respect Reduce Motion and system appearance. English/Turkish content is not considered clinically reviewed until checked by fluent medical reviewers.

## 6. Unlock and release gates

1. **Device:** versioned firmware/API, discovery transport, pairing/security, button semantics, orientation/eye identity, capture metadata, error handling, and validated safe stop.
2. **Each clinical task:** intended use, independent patient-level external cohort, reference standard, predeclared thresholds, calibration, uncertainty, subgroup and failure analysis, and qualified clinical review.
3. **Local model update security:** authenticated registry policy, compatibility/rollback behavior, storage lifecycle and security review.
4. **Persistence/export:** data-retention, deletion, consent, encryption, report provenance, threat model and jurisdictional privacy review.
5. **Release:** physical device performance, battery/thermal/memory, accessibility evaluation, English/Turkish review, and platform build/signing review.

Passing code compilation or simulator launch is not evidence that any of these gates has passed.
