# Functional-vision test contract and calibration gates

## Current app status

Both the iOS and Android applications mark functional-vision tests as unavailable. The source contains no implemented acuity, contrast-sensitivity, color-vision, visual-field, or clinically scored fixation test. The goggles are not available and no validated display/response envelope is recorded. Preserve this unavailable state; a simulated scan cannot produce a functional-vision result.

## Task-by-task requirements

| Test family | What a valid measurement needs | Current disposition |
|---|---|---|
| Visual acuity | Known angular optotype size at the eye, measured viewing distance, calibrated display/optics, monocular occlusion, standardized optotype/logMAR scoring and a validated response protocol. | Unavailable. Do not display Snellen/logMAR values or acuity percentages. |
| Contrast sensitivity | Calibrated luminance and contrast transfer, controlled ambient light, known spatial frequency/angular size and validated chart sequence. | Unavailable; phone/goggle display response not characterized. |
| Color vision | Characterized display primaries, gamma and luminance, standardized validated plates or stimuli, and response interpretation. | Unavailable; do not label a color plate demo as color-vision testing. |
| Visual field | Calibrated stimulus geometry and luminance, fixation monitoring, timing, response channel, monocular protocol and validated field scoring. | Unavailable; a phone screen is not a perimeter and synthetic eye tracking is not fixation validation. |
| Fixation / tracking | Real eye/IR sensor data, calibrated camera-to-eye geometry, latency and dropout characterization, defined target motion and participant-level validation. | Unavailable; current eye-tracking data are synthetic renders only. Do not expose a fixation score or use it to control hardware. |

## Required measurement record when the hardware exists

For each test, preserve the selected eye, device/display and firmware identifiers, calibration status and date, viewing distance, ambient conditions if required, stimulus sequence version, response timestamps, completion/termination reason, and scoring algorithm version. If calibration fails or is missing, the result is `unavailable`; do not substitute a population default or infer a clinical risk. Store participant responses separately from disease-classifier outputs.

## Safe software behavior now

- Keep the functional-vision feature visibly unavailable in both apps, with a short reason and a future hardware/calibration dependency.
- Do not generate plausible-looking sample scores in demo mode. Demo UI can illustrate navigation only and must not look like a completed examination result.
- Keep disease classifier model scores separate from visual-function measurements.
- Reopen implementation only after display/optics, response interface, and calibration procedures are specified; then validate the device-level measurement against an appropriate reference before exposing numeric results.

## Evidence boundary

This contract documents product dependencies; it does not assert clinical validity. Current app code already exposes an unavailable capability state. No hardware testing or patient study was possible in this work block.
