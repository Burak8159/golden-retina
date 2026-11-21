# Golden Retina Mobile Design System

**Direction:** Clinical Precision, with warm human-centered language and restrained premium details. Shared semantic tokens; native components on each platform.

## Foundations

| Token | Light | Dark | Usage |
|---|---|---|---|
| Brand / primary | `#176B55` | `#A0D9C2` | Primary actions and focus; never a disease-negative signal |
| Brand / deep | `#103A34` | `#DDF2E8` | Hero surfaces and selected emphasis |
| Background | Platform system background | Platform system background | Respect System/Light/Dark preference |
| Surface | Platform grouped surface | Platform elevated surface | Content grouping, used sparingly |
| Text primary | Platform primary | Platform primary | Native semantic foreground |
| Text secondary | Platform secondary | Platform secondary | Supporting content |
| Status / info | Text + info icon | Text + info icon | Demo/research/unavailable context |
| Status / caution | Text + warning icon | Text + warning icon | Incomplete/indeterminate state when supported |
| Status / error | Text + error icon | Text + error icon | Technical failure; not routine medical urgency |
| Status / success | Text + completion icon | Text + completion icon | Technical workflow completion only |

Colors are not the sole carrier of status. Do not infer health from green or danger from color alone.

## Type and layout

- Prefer system fonts and semantic styles; preserve Dynamic Type / Android font scale.
- iOS: body defaults to system body; avoid text below footnote for explanatory or medical limitations. Android: Material 3 `bodyLarge/bodyMedium` for instructions, `labelSmall` only for nonessential metadata.
- Content max width 600–680 dp/pt for reading columns; full-width on compact screens.
- Spacing rhythm: 4, 8, 12, 16, 20, 24, 32.
- Shape rhythm: small 12, standard 18–22, hero 24–28; use cards only to express grouping/action.
- Minimum interactive target: iOS 44×44 pt preferred; Android 48×48 dp. Keep eye-side controls broad and explicitly labeled.
- Keep status text adjacent to its affected content; headings and screen-reader order follow visual order.

## Component patterns

| Component | Required states | Behavior |
|---|---|---|
| Primary action | enabled, disabled, busy | One clear action; no fake progress |
| Capability row | available, unavailable, unsupported | State text plus reason; no dead-end affordance |
| Demo/research notice | persistent | Stays visible through result review |
| Eye selector | right, left, unavailable | Label side in text; left is physically left and right physically right |
| Model row | not installed, checking, ready, failed | Model ID/version and raw-score caveat |
| Empty history | no data, data unavailable | State that no real sessions are stored; offer demo/research route |
| Error block | explanation, retry, support | Plain language, saved-data status where known |
| Result summary | raw research score, demo sample, future validated result | Distinct visual and copy treatment; never reuse one template without state label |
| Settings selector | system/light/dark | Persisted user preference, platform-native control |

## Result and research component rules

- Keep **disease-condition labels** and **retinal image findings** in separate sections with separate headings. Do not put optic-disc cupping, tessellation, scars, or other image findings inside a disease group.
- Eye tabs are always ordered and physically positioned Left then Right. The tab label includes that eye's own highest raw score only when a score exists; otherwise show an em dash. Selecting one eye never changes the other eye's result.
- Every result score is rendered as a percentage-formatted **raw model score** with a nearby persistent explanation that it is uncalibrated and not a diagnosis or clinical risk. A 0.50 research cutoff may be described as a model cutoff, never as a clinical decision threshold.
- The demo's fixed example values remain visibly identified as sample/demo data on the result itself. They are never mixed with local model results or copied into history.
- Model unavailable, label unsupported, quality not assessed, invalid output, image decode failure, download/checksum failure, and inference failure each have a distinct text state. Do not render a missing value as 0%.
- Long label groups use progressive disclosure. Group summary is “Highest raw score,” not “risk”; subitems retain their full dataset-defined label and model provenance.

## Source-grounded visual implementation

- iOS uses semantic system backgrounds, foreground styles, native tab and list controls, and the user's System/Light/Dark selection. Avoid fixed black surfaces that erase the platform's base/elevated dark distinction.
- Android uses Material 3 color roles and system bars, adaptive content width, scrollable long disease lists, and minimum 48 dp targets. Dynamic color may tint the app but must not encode positive/negative health meaning.
- Shared palette is a semantic reference, not a mandate to override platform contrast or appearance. Brand green is an action/focus accent only; it never means normal or disease-free.
- Text scales with iOS Dynamic Type and Android font scale; rows must wrap rather than truncate disease names or explanatory status. Use explicit accessibility labels for Left/Right and score context.
- Motion is reserved for actual navigation or timed demo transitions, is cancellable, and respects Reduce Motion / Android animator settings. The two continuous SwiftUI demo loops now stop when Reduce Motion is enabled; Compose animations use the platform animation duration scale.

## Motion

Motion communicates navigation or an actual timed demo transition only. Honor Reduce Motion/system animation scale. Never animate a fictional retinal scan as if acquisition were live.

## Platform mapping

- **iOS:** TabView, NavigationStack, native List/Section, native sheet/share controls, SF Symbols with labels, semantic system colors.
- **Android:** Material 3 NavigationBar, Scaffold, CenterAlignedTopAppBar, native Back handling, Material dialogs/snackbars, adaptive width and system bars.
- Shared semantics and copy; no forced pixel-identical shell.

## Content and status strings

- “Demo — simulated steps and sample values. No eye images were captured.”
- “Research model output — raw score, not a calibrated disease risk or diagnosis.”
- “Headset connection unavailable — compatible hardware and its interface are not available in this app build.”
- “No examination history yet — demo sessions are not saved as health records.”
- Avoid “all clear,” “healthy,” “disease-free,” “risk percentage,” and “validated” unless evidence and an approved output contract support the specific claim.
