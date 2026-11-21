# On-device deployment footprint and readiness

**Audit date:** 2026-11-21
**Registry:** `20261118-r5`
**Evidence:** local model files and source inspection of iOS/Android registry and inference code

## Measured artifact footprint

| Set | Files | Bytes | MiB (2^20 bytes) |
|---|---:|---:|---:|
| Four post-hoc RFMiD shortlist classifiers (DR, AMD, myopia, BRVO) | 4 | 223,067,500 | 212.8 |
| All 17 RFMiD condition/finding heads | 17 | 939,272,663 | 895.8 |
| Synthetic eye-localization model | 1 | 1,959,735 | 1.9 |
| Shortlist plus eye localization | 5 | 225,027,235 | 214.6 |
| Entire current registry | 18 | 941,232,398 | 897.7 |

These are file sizes from the verified local artifacts, not peak resident memory or installed-app size. The four-head shortlist is a post-hoc development shortlist, not a clinical product recommendation. Each head is an independent image classifier; the scores are not calibrated probabilities or a mutually exclusive disease distribution.

## What the code supports

- Both platforms use the same GitHub-hosted manifest and install each ONNX model separately on demand. They validate HTTPS host/path, expected file length and SHA-256; model session creation is checked before activating the downloaded file. Failed downloads/inference surface an error instead of returning a negative disease classification.
- Disease inference is local after installation. The GitHub registry distributes model files and versions; it is not a remote inference service and selected fundus images are not sent to a server by these screens.
- Both disease preprocessors convert RGB pixels to channel-first tensors, square-resize to 320x320, and apply the registry's ImageNet mean/std. Android applies EXIF orientation; iOS canonicalizes orientation while creating the selected-image thumbnail.
- Current model code evaluates installed heads sequentially and opens/closes an ONNX session for each head. Individual download avoids requiring the full ~896 MiB registry on every device.

## Integration defect to fix during the final mobile phase

The Android disease path reads the first float from the output tensor and clamps it into `[0,1]`; it does not verify that the output tensor contains exactly one finite score in range. The iOS path rejects non-finite or out-of-range output and checks for exactly one value. This is a real cross-platform failure-mode mismatch. The current manifest files are hash-correct, but malformed or incompatible output should fail closed consistently rather than be silently clamped. Fix both implementations against the shared result schema and verify on each platform in the final mobile workstream.

## Not measured

No physical handset latency, peak memory, thermal throttling, battery draw, offline reinstall, or sustained repeated-inference profile is available in this campaign. L4 timing would not answer those phone-specific questions. Simulators are useful for packaging and UI integration but cannot establish physical-device performance. No speed or deployment-readiness claim is made from file size alone.

## Decision

The artifacts can be distributed incrementally for research use, subject to the existing evidence and data-use restrictions. The current evidence does not support a clinical deployment. Before a pilot, resolve the Android output-shape/range mismatch, test fresh install/update/failure paths on actual iOS and Android devices, measure memory/latency/thermal behavior, and independently validate each intended disease task on patient-disjoint external cohorts. Keep image-quality assessment explicitly unavailable and keep retinal findings separate from disease labels.
