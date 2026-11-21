# Golden Retina Mobile Screen Inventory and Coverage

This inventory maps the master directive to reusable product templates. “Implemented” describes code presence, not usability approval or clinical readiness.

| Product area | Reusable screen/state family | Current coverage | State / missing evidence |
|---|---|---|---|
| Launch/onboarding | Welcome, purpose, limits, privacy, consent, profile, completion | Not implemented as a complete flow | Needs product identity/account/privacy decisions and approved consent text |
| Home | First use, no device, device readiness, recent session, demo mode | Partial | Current concept home only; no real device/session data |
| Screening | Hub, demo, local image analysis, model selection | Implemented foundation | Demo is simulated; local research model is not clinical screening |
| History | No sessions, completed sessions, eye-specific detail, comparable/non-comparable trends | Empty state implemented | No real exam persistence or comparable longitudinal data |
| Device | Not paired, discover, pairing, connection, low battery, calibration, fault, maintenance | Unavailable state implemented | No verified hardware protocol/SDK; no Bluetooth discovery in app |
| Preparation | Scope selection, fit, checklist, readiness | Partial guide | No validated fit instructions or hardware state |
| Active exam | eye align, focus, capture, quality, switch, pause, fault, complete | Simulated flow only | No actual acquisition/state data; do not imply phone visibility while headset worn |
| Functional vision | acuity, contrast, field, color, Amsler, reliability, results | Not implemented | Requires calibrated headset display and validated test contracts |
| Results | processing, positive, negative, indeterminate, ungradable, incomplete, referral | Demo summary + research raw scores | No validated clinical result categories or thresholds |
| Disease details | 7 manifest-defined dataset condition labels, plus 10 separately named retinal image findings | Implemented in the research-model lab | Exploratory RFMiD raw scores only; glaucoma, clinical risk, diagnosis, and broad any-disease classification are not supported |
| Image viewer | eye/field, zoom/pan, metadata, overlays, comparison | Image picker/inference only | No actual exam image/metadata contract; no fabricated anatomy |
| History/insights | timeline, eye detail, comparable measures, missing/non-comparable | Honest empty state in current navigation | No persisted actual examination record schema |
| Reports | preview, provenance, export/share, failure | Not implemented | No verified session data/provenance or secure-sharing design |
| Education | condition explanations, limitations, urgent symptoms, preparation | Partial help/manual | Clinical review and localization needed |
| Device management | battery, firmware, diagnostics, calibration, unpair | Unavailable | Hardware/API dependency |
| Settings/privacy | appearance, model registry, appearance, privacy, accessibility, language, help | Partial | No account/cloud/privacy service exists; don’t promise export/delete backend |
| Multi-user/caregiver | identity switch, authorization, separate history | Not implemented | Requires account/data ownership and consent design |
| Localization | English/Turkish keys and medical text review | Partial / not audited | Requires localization resource audit and qualified review |
| Accessibility | dynamic text, VoiceOver/TalkBack, targets, contrast, reduce motion | Source audit complete; representative Android emulator screens inspected | Two iOS motion loops fixed; physical-device screen-reader/low-vision review remains open; see accessibility audit |

## Minimum safe functional prototype journeys

1. Home → Screening → simulated device flow → right eye then left eye → sample result, persistently marked demo.
2. Screening → choose local image → installed compatible on-device research model → raw model scores and explicit research limitations.
3. Device → unavailable reason → how-to guide; no Bluetooth permission request.
4. History → truthful no-real-session empty state.
5. Settings → appearance, model management, help/about and privacy limitations.

## Error and empty state checklist

Model missing; unsupported image; invalid manifest; failed download/checksum; inference error; no image permission; no session history; device absent; pairing unsupported; unsupported function; demo interrupted; insufficient or ungradable future image; export unavailable. Each state must explain what happened, what data is known to have been saved, and the safe next action. Unknown persistence must be described as unknown rather than guessed.

## Full directive coverage index

These IDs follow the source directive. They are reusable states/variants, not 200 separate routes or claims that each is built.

| Directive IDs | Screens/states represented | Status |
|---|---|---|
| 1–16 | Splash; introduction; product overview; assessable/not assessable scope; limitations; exam explanation; privacy; account creation/sign-in/recovery; consent; profile/medical-history optionality; device setup choice; onboarding completion | Onboarding flow not built; requires account, consent, privacy, and clinically reviewed capability scope |
| 17–37 | No device; discovery; connection instructions/permission; detection; confirm; progress/success/failure; already paired; disconnected/reconnect; compatibility/update/progress/failure; readiness/self-test; calibration; fit/cleaning; ready | Device unavailable state built; live states require firmware contract |
| 38–45 | First-time home; connected/disconnected home; empty/recent/incomplete exam; follow-up; demo home | First-time and demo concept state partially built; no connected/recent/incomplete/follow-up record state |
| 46–60 | Exam selection; full screening; imaging-only; functional test; supported module list; checklist; position/fit/alignment; correction/environment/battery/readiness; response control; confirm | Preparation guide partial; real readiness and supported modules require device/validation data |
| 61–80 | Start; left/right alignment; pupil; alignment progress/failure; focus; capture; field progress; quality; recapture; eye switch; extra field; near-complete; complete; pause; interrupt; safe failure/cancel/recovery | Demo sequence only; no live device data; no safety control exposed |
| 81–93 | Vision test selection; acuity; contrast; field; color; Amsler; eye-specific progress; fixation; response; unreliable; pause/restart; completed; summary | Unavailable; requires calibrated headset display, response contract, and validation |
| 94–110 | Processing; result unavailable; quality summary; overview; within-scope negative; positive; indeterminate; ungradable; incomplete; follow-up; urgent assessment; disease details; limitation/confidence; next steps; report generation/share | Demo summary and research raw scores only; clinical categories and report flow not enabled |
| 111–121 | DR; AMD; glaucoma suspect; hypertensive changes; pathological myopia; vascular findings; other finding; field, acuity, contrast, color result templates | Model-lab exploratory outputs only; no clinical status language or thresholds |
| 122–134 | Full-resolution image viewer; eye/field; macula/disc/superior/inferior; zoom/pan; metadata; validated overlays; annotations; comparison; export; missing image | User-selected image analysis exists; exam image viewer and metadata/overlays absent |
| 135–146 | History; timeline/calendar; detail; eye/disease/function history; image comparison; trends; missing/non-comparable data; archive | Empty History state built; no persisted sessions or trends |
| 147–155 | Report overview/preview/PDF; export/share; secure share; confirmation/failure; professional review; provenance/model versions | Not implemented; no exam data/share contract |
| 156–166 | Education hub; DR/AMD/glaucoma/vascular explanations; scope limits; professional/urgent guidance; preparation; cleaning; troubleshooting | Basic manual/limitations only; all medical education requires qualified review |
| 167–182 | Device overview; battery/charging/connectivity/firmware; updates; diagnostics; calibration; info/rename/unpair; recovery; maintenance/fault/support | Unavailable-state screen only |
| 183–200 | Profile; personal data; accessibility; notifications; language/units/region; privacy/preferences; export/delete; permissions/devices/consent; help/about/legal/version/diagnostics | Appearance, local model settings, privacy disclosures, help/about partial; account/backend and language/service controls absent |

## Program stage status

| Stage | Current evidence |
|---|---|
| 19 Discovery | Existing apps, model lab, demo behavior, repo docs inspected; missing firmware/API and clinical capability documents recorded |
| 20 Product architecture | Local inference and registry-only model distribution, five-destination IA, recovery, privacy and release gates documented |
| 21 Design system | Native themes, scalable typography, component/status semantics, eye-side behavior and motion guidance documented |
| 22 Journeys | Demo, local image analysis, model distribution, error recovery and locked headset states mapped |
| 23 iOS | SwiftUI flows and result semantics updated; syntax parse succeeds; build/runtime not verified because Xcode/simctl are unavailable |
| 24 Android | Debug build and Pixel_11_API_37 emulator review verified; this is not a physical Pixel 11 |
| 25 Advanced content | Manifest taxonomy corrected to 7 dataset condition labels and 10 image findings; unsupported clinical outputs remain absent |
| 26 Accessibility/privacy/localization | Source audit and contrast samples recorded; iOS Reduce Motion loops fixed; physical screen-reader testing, Turkish review, and full contrast certification remain open |
| 27 Developer handoff | Active: reconcile architecture, interaction, taxonomy, accessibility/privacy audit, platform readmes and release checklist |
| 28 Final review | Pending handoff completion; iOS build, physical-device review, hardware and clinical dependencies remain explicit blockers |
