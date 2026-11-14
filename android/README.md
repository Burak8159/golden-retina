# Golden Retina Android concept app

Jetpack Compose counterpart to `../ios/GoldenRetina`. The Android experience
uses native Material 3 navigation, cards, chips, dialog, progress, system Back,
and responsive content widths.

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

The demo simulates finding a sample device, connecting, scanning a sample right
and left eye, analyzing, and displaying invented results. It does not use the
camera, Bluetooth, patient images, the screening model, or a network API. The
goggles route is locked because the hardware and Bluetooth protocol are not
ready. No camera or Bluetooth permissions are declared.

The current Golden Retina inference services are loopback-bound research
prototypes, not a consumer mobile backend. A real connection needs an approved
authenticated network service and a defined hardware pairing/image-transfer
protocol. Demo output remains separate from any future live results.

The About screen and every simulated stage disclose that this is not a medical
device or diagnosis. Do not use example values for health decisions.

## Design and accessibility choices

- Material 3 navigation bar has two real top-level destinations: Home and About.
- The demo is a child task with an explicit Up/close action and system Back exits
  to Home. The locked goggles card remains inspectable so the unavailable state
  has an explanation.
- System light/dark appearance is respected; Android 12+ uses wallpaper dynamic
  color, with Golden Retina green fallbacks on older Android versions.
- Text uses Material typography in scalable sp units; layouts scroll and cap
  content width, and primary touch controls remain at least 48 dp high.
- No notification or launcher widget is included: the demo is short, user-driven,
  and has no background task or glanceable status to surface outside the app.
