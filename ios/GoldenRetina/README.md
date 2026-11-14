# Golden Retina iOS concept app

Native SwiftUI app for the consumer-facing concept. The only working route is a
guided **simulation**: simulated device discovery, simulated goggles connection,
right-eye and left-eye scan animations, timed analysis, and invented sample
results. Every phase labels itself as a demo. It does not use a camera, Bluetooth,
patient images, a model, or the research API.

The goggles route is intentionally locked because the hardware and Bluetooth
protocol do not exist yet. No Bluetooth permission is requested.

## Open in Xcode

Open `GoldenRetina.xcodeproj` in Xcode 16 or later, select an iPhone simulator,
and Run. Deployment target: iOS 17. The current development Mac has Swift
command-line tools but no full Xcode installation, so an Xcode build and
simulator review could not be performed here.

## Integration boundary

The repository's current inference services are research prototypes bound to
loopback and require research-only handling. They are not a production phone
backend and must not be presented as clinical diagnosis. Before enabling a live
route, define an authenticated, network-reachable service contract, approved
data handling/retention, device pairing protocol, image-quality behavior, model
versioning, and user-facing clinical scope. Keep demo results isolated from any
future live inference path.
