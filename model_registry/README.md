# On-device model registry and inference

**Current state:** iOS and Android contain a local ONNX inference path for selected fundus images and a versioned registry client. The registry distributes model files and metadata; image inference runs on the phone. The guided goggles scan remains simulated. Physical goggles and real eye tracking are not implemented.

## Artifacts in this checkout

mobile_models/onnx/manifest.json declares 18 ONNX artifacts: one synthetic eye-localization model and 17 RFMiD disease/finding specialists. Android bundles the eye-localization model and downloads selected disease experts from the trusted HTTPS GitHub raw-content path. iOS includes a subset as project resources and can download additional registered models. Installation checks the declared byte count and SHA-256, then asks ONNX Runtime to open the model before recording its version.

These artifacts support research inference, not clinical screening. The four post-hoc shortlist labels are DR, AMD/ARMD, myopia and BRVO. Other heads are exploratory, and multiple outputs are image findings rather than diseases. Glaucoma is not an installed classifier. Optic-disc cupping is not equivalent to glaucoma. The RFMiD test partition is image-level, lacks verified patient independence and was previously exposed by the project.

## App behavior

1. The user selects a fundus image locally and assigns it to Left or Right.
2. The app preprocesses it according to the model contract.
3. ONNX Runtime evaluates each installed model on-device.
4. The app shows a raw, uncalibrated research score with the model identity/version. It must not call this a probability, risk or diagnosis.
5. The selected image is not sent to the model registry or a server.

The guided scan demo is separate: its animation does not capture or analyze an image. The synthetic eye-localization model is for synthetic images only and has not been validated on people, infrared camera frames or goggles.

## Registry trust and updates

The current clients pin model URLs to the Golden Retina GitHub raw-content repository, validate HTTPS and the declared byte count, check SHA-256, and confirm ONNX Runtime can open the downloaded model. The manifest itself is not cryptographically signed. TLS plus a digest delivered in the same manifest detects transfer corruption but does not protect against a compromised repository/account serving a changed model and changed digest. Signed releases or app-store-controlled updates remain a release gate.

Schema changes that alter inputs or outputs require an explicit compatibility version. Keep a previous working model until a new download passes all checks. Never install an unverified model or silently interpret a missing head as a negative result.

## Evidence boundary

These artifacts form a research-model lab, not a validated diagnostic product. Per-head RFMiD results in docs/mobile_model_artifacts.md are image-level development measurements and include post-hoc selection. Patient-level independence, calibration, external-site performance, camera compatibility and clinical safety have not been established. The active disease-level restart requirements are in docs/product_restart_20261126.md.

## Legacy example

manifest.example.json illustrates the schema only. mobile_models/onnx/manifest.json is the active app registry catalog. Do not confuse the example with the model bundle.
