# Golden Retina Android concept app

Jetpack Compose counterpart to `../ios/GoldenRetina`. The Android experience
uses native Material 3 navigation, cards, dialogs, progress, system Back, and
responsive content widths. Top-level destinations are Home, Screening, History,
Device, and Settings. Screening separates the guided simulation from local
research image analysis.

## Run

Open this `android` folder in Android Studio and run the `app` configuration on
an emulator or device. To build, boot an installed AVD, install, and launch from
Terminal, run `scripts/run_android_emulator.sh [AVD_NAME]` from the repository
root. The default is `Medium_Phone_API_37.0`. The project uses Android Gradle
Plugin 9.5.1, Kotlin 2.4.10, Jetpack Compose BOM 2026.09.00, compile SDK 37,
and min SDK 26. Syncing requires network access to Google's and Maven Central's
package repositories.

The default emulator profile is a generic 1080x2400 Android 37 device, not a
Pixel 11. Install or create a matching Pixel profile in Android Studio Device
Manager before passing its AVD name to the script. The script reports an error
instead of silently substituting a different profile.

## What is and is not connected

The demo simulates finding a sample device and connecting. The user then taps an
on-screen control that represents pressing the goggles' physical start button;
the right-eye scan is followed automatically by the left-eye scan, analysis, and
example results. It does not use the camera, patient images, the screening
model, or a network API. The headset route is unavailable because the hardware
and connection protocol are not ready. No camera or discovery permissions are
declared.

The mobile research-model lab runs installed ONNX models on-device and is
separate from the demo. The app downloads model metadata and files from the
public HTTPS registry; image inference stays on the phone. No inference server
is used. The models produce research scores, not clinically validated
diagnoses. Real headset operation needs a versioned hardware protocol and
supported capture/state contract.

The app and every simulated stage disclose that this is not a medical diagnosis.
Do not use sample values or research-model scores for health decisions.

## Design and accessibility choices

- Material 3 navigation has five destinations. The demo and research inference
  are child screening flows; History remains empty until real examinations can
  be stored. Device explains the missing hardware/protocol and never starts a
  device scan.
- System light/dark appearance is respected; Android 12+ uses wallpaper dynamic
  color, with Golden Retina green fallbacks on older Android versions.
- Text uses Material typography in scalable sp units; layouts scroll and cap
  content width, and primary touch controls remain at least 48 dp high.
- No notification or launcher widget is included: the demo is short, user-driven,
  and has no background task or glanceable status to surface outside the app.

## Product design and handoff

See [`../docs/mobile_product_foundation.md`](../docs/mobile_product_foundation.md),
[`../docs/mobile_product_architecture_20261121.md`](../docs/mobile_product_architecture_20261121.md),
[`../docs/mobile_design_system.md`](../docs/mobile_design_system.md),
[`../docs/mobile_result_content_contract_20261121.md`](../docs/mobile_result_content_contract_20261121.md),
[`../docs/mobile_screen_inventory.md`](../docs/mobile_screen_inventory.md),
[`../docs/mobile_handoff.md`](../docs/mobile_handoff.md), and
[`../docs/mobile_accessibility_privacy_audit_20261121.md`](../docs/mobile_accessibility_privacy_audit_20261121.md)
for shared product architecture, platform implementation boundaries, and
recorded accessibility/privacy review limits.
