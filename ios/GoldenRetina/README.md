# Golden Retina iOS concept app

Native SwiftUI app for the consumer-facing concept. The only working route is a
guided **simulation**: simulated device discovery, simulated goggles connection,
right-eye and left-eye scan animations, timed analysis, and invented sample
results. Every phase labels itself as a demo. It does not use a camera, Bluetooth,
patient images, a model, or the research API.

The goggles route is intentionally locked because the hardware and Bluetooth
protocol do not exist yet. No Bluetooth permission is requested.

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

## Integration boundary

The repository's current inference services are research prototypes bound to
loopback and require research-only handling. They are not a production phone
backend and must not be presented as clinical diagnosis. Before enabling a live
route, define an authenticated, network-reachable service contract, approved
data handling/retention, device pairing protocol, image-quality behavior, model
versioning, and user-facing clinical scope. Keep demo results isolated from any
future live inference path.
