# Mobile accessibility, privacy and localization audit

**Scope:** source review of the iOS SwiftUI and Android Compose applications, plus an Android emulator visual pass. This is not a user study, clinical approval, or complete assistive-technology certification.

## Findings and fixes

- **Scalable content:** user-facing copy uses SwiftUI semantic text styles and Material 3 scalable typography. Long model/result screens scroll; explanatory content wraps. Decorative artwork has fixed geometry, is hidden from accessibility, and is not used to carry meaning.
- **Navigation and controls:** native tab/navigation controls provide platform semantics; result eye tabs say Left/Right and include each eye's own highest raw model score. Demo state and model provenance are also given as text. Standard buttons and tabs are used instead of custom gesture-only controls. Source review cannot prove VoiceOver/TalkBack reading order on every OS version.
- **Motion:** reduced-motion behavior was already applied to iOS phase transitions. This audit found two continuous iOS demo animations that bypassed it; both now skip their repeating animation when Reduce Motion is enabled. Android demo uses Compose animations, which follow the system animator duration scale. No motor, live camera, or live scan animation is represented as operational.
- **Contrast and themes:** app surfaces use native semantic colors/system appearance or Material 3 light/dark roles. The shared primary colors have calculated contrast ratios of 6.15:1 (iOS `#146E5C` on white), 6.42:1 (Android `#176B55` on white), and 11.41:1 (Android dark primary `#A0D9C2` on `#12161C`). These checks cover only the listed solid-color pairs; they are not a full per-screen contrast certification. Status is paired with text/icons and is never encoded by color alone.
- **Image and network handling:** source review found network GETs only for the HTTPS model manifest and model artifacts. Selected images are decoded and passed to local ONNX Runtime; no image upload endpoint or inference service is implemented. Analysis results are transient view state, and demo results are not written to history. Registry hash verification does not authenticate a modified manifest; signed registry metadata or an equivalent pinned trust policy remains a release gate.
- **Language:** the app is English-only. Turkish localization and any medical instructions require fluent-speaker and qualified clinical review; no translated copy was invented.

## Remaining verification

- iOS build/simulator review is unavailable on this workstation because Xcode and `simctl` are not installed. Swift syntax parsing is a narrower check.
- Android debug build and the Pixel_11_API_37 emulator visual pass are recorded separately; this is an emulator, not a physical Pixel device.
- VoiceOver, TalkBack, large-font interaction, switch control, and low-vision usability still need hands-on review on physical devices by users. No accessibility conformance claim is made.
- Full contrast, localization, privacy threat-model, registry-signature, clinical, and hardware reviews remain release gates.
