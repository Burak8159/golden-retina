# Golden Retina Mobile Developer Handoff

## Architecture boundary

The iOS and Android apps own presentation, local user interaction, and on-device model execution. No remote inference service is part of this product direction. The public HTTPS registry distributes model metadata and files only; selected images never leave the phone. The current registry verifies file size and SHA-256 against its manifest, but manifest authenticity is not cryptographically signed yet. A pinned signing key or app-store-controlled registry policy is a release gate. The device firmware and headset safety logic remain owned by the hardware team.

## Provisional future device contract (not implemented)

Do not implement against guessed BLE UUIDs or payloads. The hardware team must supply a versioned contract before enabling device actions:

```text
DeviceCapabilities: protocolVersion, capabilities[]
DeviceStatus: connectionState, batteryState?, firmwareVersion?, calibrationState?, fault?
ExamCommand: requestId, requestedModules[]
ExamEvent: requestId, state, eye?, progress?, qualityState?, recoverableError?
CapturedImage: examId, eye, fieldId, orientation, mediaType, dimensions, captureTime, qualityMetadata
ModelResult: examId, taskId, modelId, modelVersion, rawOutput, quality/uncertainty, evidenceStatus
```

This is a discussion schema only. Optional fields must not be shown until actual support is confirmed. Firmware must retain control over motor limits, exposure safety, fault handling, and safe stop. Phone UI cannot bypass them.

## State and navigation

Top-level destinations: Home, Screening, History, Device, Settings. Demo and on-device research are child flows of Screening. Help is contextual under Settings/Device. Use native navigation and preserve navigation state when switching tabs.

## Evidence-safe data presentation

- Demo sample outputs are fixed illustrative data; label the entire flow, not only a small badge.
- Model output is a raw score associated with model ID/version, RFMiD label semantics and selected image; never call it disease probability or risk unless calibrated and validated for that claim.
- Only installed models are scored. A model that is not installed or supported is not assessed, never a zero score or normal result.
- Render dataset condition labels separately from retinal image findings. A finding is not promoted to a disease diagnosis.
- Do not persist demo results as patient/exam records.
- History remains empty until real exam persistence is implemented.
- Eye-side selectors must say Left/Right and place them in the corresponding physical order.
- No disease-positive/negative/referral result is enabled in the absence of a validated capability definition.

## Accessibility and localization handoff

- Verify VoiceOver/TalkBack names, reading order, headings, controls and dynamic content.
- Support system text scaling; controls remain operable at large sizes and narrow widths.
- Meet native minimum touch targets (44 pt iOS; 48 dp Android), visible focus, contrast, and non-color-only status communication.
- Honor Reduce Motion and OS appearance; the existing System/Light/Dark preference is retained.
- Externalize user-visible copy into localization resources; Turkish medical copy requires human and clinical review before release.

The source-level accessibility/privacy review and its limits are recorded in `docs/mobile_accessibility_privacy_audit_20261121.md`. The two iOS continuous demo animations honor Reduce Motion; Android Compose animations follow the platform animation duration scale. This does not replace VoiceOver/TalkBack or physical-device review.

## Privacy and security boundary

Current research image inference runs locally in the app. The model registry is a model distribution channel, not an inference service. Do not upload the selected image. Do not claim an account, cloud backup, secure share link, remote deletion, or consent service exists. Future persistence/export work needs data inventory, retention/deletion semantics, threat model, permission rationale, encryption/storage review, and applicable legal review (including Türkiye KVKK/GDPR applicability by counsel).

## Release checklist

- [ ] Firmware protocol and supported capability matrix supplied and versioned.
- [ ] Real exam data schema, save/retry/cancel semantics, and safe fault behavior agreed.
- [ ] Each clinical task has frozen intended use, threshold policy, validation evidence, quality handling, and approved language.
- [ ] Model registry trust/update policy reviewed; model IDs/versions and checksums shown.
- [ ] Actual-user usability and low-vision accessibility sessions completed and findings resolved.
- [ ] English/Turkish copy reviewed by fluent speakers and clinicians for consequential instructions.
- [ ] Privacy/security/legal review complete for actual data flows.
- [ ] iOS/Android builds, navigation, image selection, model inference, empty/error states and accessibility manually reviewed on target devices.

## Status of design gates

This source-based implementation is a foundation, not a claim that UX-01–UX-20 or the 200-item screen inventory is fully accepted.

| Gate | Status | Evidence / remaining work |
|---|---|---|
| UX-01 Product architecture | Foundation complete | Product foundation document; validate with users and product owners |
| UX-02 Three directions | Desk evaluation complete | Internal scoring only; not user research |
| UX-03 Design system | Foundation complete | Tokens and component behavior documented; visual contrast/runtime review open |
| UX-04 iOS experience | Source implementation complete; runtime build unverified | Five-tab navigation, demo, local research results, settings, unavailable/history states and result taxonomy updated; Swift syntax parse succeeds. Xcode/simctl unavailable on this Mac; device build remains open. |
| UX-05 Android experience | Debug build and emulator launch verified | Compose five-destination navigation, local models, separate disease/finding results, eye score tabs, sample demo and settings inspected on Pixel_11_API_37. Human usability and accessibility checks remain open. |
| UX-06 Exam journeys | Demo only | Real exam requires firmware API and validated workflow |
| UX-07 Device recovery | Unavailable-state design only | No pairing, telemetry, or hardware fault contract |
| UX-08 Evidence-aware results | Partial | Demo and research boundaries present; clinical categories are not enabled |
| UX-09 Retinal viewer | Not complete | No actual capture contract or result overlay support |
| UX-10 History/trends | Empty state only | No real exam persistence/comparability schema |
| UX-11 Functional vision tests | Not implemented | Requires calibrated display and validation |
| UX-12 Reports/sharing | Not implemented | Requires real provenance/data and security design |
| UX-13 Privacy/account management | Partial disclosure | No account or privacy backend; no data-export/deletion workflow |
| UX-14 Low-vision accessibility | Partial | Semantic platform components and scalable text; runtime assistive review outstanding |
| UX-15 Light/dark themes | Implemented as preference | Native/system theme; per-screen visual QA outstanding |
| UX-16 English/Turkish | English only | Turkish resource files and human/clinical review remain |
| UX-17 Interactive prototypes | Android emulator runtime observed; iOS source parse only | Android debug APK assembled, installed side-by-side as a `.dev` app, launched, and Home/Screening/Demo result/Settings screens inspected. iOS syntax parses; no Xcode app or `simctl` is installed, so no iOS simulator build was possible. |
| UX-18 Error/empty/loading | Partial | Research and unavailable/empty states exist; full error matrix remains |
| UX-19 Developer handoff | Foundation complete | Architecture, provisional contract, constraints, and release checklist documented |
| UX-20 Final review | Open | Requires an Xcode install for iOS build/runtime, additional platform QA, accessibility sessions, clinical and hardware review |

Device integration, validated result categories, actual retinal viewer, stored history, report/export, calibrated vision tests, multi-user accounts, clinical review, user research, localization QA, and final release approval remain open dependencies.
