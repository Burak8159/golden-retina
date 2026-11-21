# Golden Retina iOS concept app

Native SwiftUI app for the consumer-facing concept. Top-level navigation is Home,
Screening, History, Device, and Settings. Screening offers a guided **simulation**
and a separate local research image-analysis route. The simulated flow has an
on-screen action representing the headset's physical start button, then an
illustrative right-eye followed by left-eye sequence and fixed sample results.
It does not capture eyes or connect to hardware. The image-analysis route can
run installed on-device research models; their raw scores are not clinical risks
or diagnoses.

The headset route is intentionally unavailable because the hardware and
connection protocol do not exist yet. No discovery permission is requested.

## Product design and handoff

See [`../../docs/mobile_product_foundation.md`](../../docs/mobile_product_foundation.md),
[`../../docs/mobile_product_architecture_20261121.md`](../../docs/mobile_product_architecture_20261121.md),
[`../../docs/mobile_design_system.md`](../../docs/mobile_design_system.md),
[`../../docs/mobile_result_content_contract_20261121.md`](../../docs/mobile_result_content_contract_20261121.md),
[`../../docs/mobile_screen_inventory.md`](../../docs/mobile_screen_inventory.md),
[`../../docs/mobile_handoff.md`](../../docs/mobile_handoff.md), and
[`../../docs/mobile_accessibility_privacy_audit_20261121.md`](../../docs/mobile_accessibility_privacy_audit_20261121.md).
These define
the cross-platform information architecture, status language, screen coverage,
future device contract boundary, and release dependencies.

## Build and run in Xcode

Open `GoldenRetina.xcodeproj` in Xcode, select an installed iPhone simulator,
and Run. The deployment target is iOS 17. If the shell's active developer path
points to Command Line Tools, select the Xcode bundle for a command with
`DEVELOPER_DIR="/path/to/Xcode.app/Contents/Developer"` or set Xcode under
Xcode > Settings > Locations > Command Line Tools.

For command-line build and launch, substitute the installed Xcode path and
simulator UDID shown by `xcrun simctl list devices available`. Using a UDID
avoids ambiguity when multiple simulators are booted:

```sh
export DEVELOPER_DIR="/path/to/Xcode.app/Contents/Developer"
SIMULATOR_UDID="<iPhone simulator UDID>"
xcodebuild -project ios/GoldenRetina.xcodeproj -scheme GoldenRetina \
  -destination 'platform=iOS Simulator,name=iPhone 18 Pro,OS=27.0' \
  -derivedDataPath /tmp/golden-retina-ios CODE_SIGNING_ALLOWED=NO build
xcrun simctl boot "$SIMULATOR_UDID" 2>/dev/null || true
xcrun simctl bootstatus "$SIMULATOR_UDID" -b
xcrun simctl install "$SIMULATOR_UDID" /tmp/golden-retina-ios/Build/Products/Debug-iphonesimulator/GoldenRetina.app
xcrun simctl launch "$SIMULATOR_UDID" org.goldenhorn8159.goldenretina
```

Exact simulator device names depend on the installed Xcode and runtimes. The
iPhone Duo profile uses two displays; its app UI appears on the inner display.

## Inference and integration boundary

The phone runs installed ONNX models locally. The public HTTPS model registry
distributes manifest metadata and model files only; the app does not send images
to an inference server. File size and SHA-256 are checked against the fetched
manifest, but the manifest itself is not signed yet, so authenticated registry
updates remain a release gate. Demo values are fixed samples and never enter the
local image-analysis path.

Headset connection and capture remain unavailable until the hardware team
supplies a versioned pairing, capture, eye-side, error and safe-stop contract.
Research scores are not clinical diagnoses. Xcode is not installed in the
current workstation environment; only Swift syntax parsing has been verified
here, not an iOS build or simulator launch.
