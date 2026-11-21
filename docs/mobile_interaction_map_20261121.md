# Golden Retina mobile journeys and interaction map

This map specifies the current app journeys and the disabled future headset path. It describes intended behavior, not completed usability or clinical validation.

## Primary destinations

| Destination | Job | Empty/unavailable state |
|---|---|---|
| Home | Explain current readiness and route to Screening | State that goggles and clinical screening are unavailable in this build. |
| Screening | Choose Guided Demo or Research Image Analysis | Do not present either as a real headset exam. |
| History | Review real persisted examinations, if implemented later | “No examination history”; simulations are not records. |
| Device | Explain connection readiness and setup | No Bluetooth discovery or permission request until firmware interface is defined. |
| Settings | Appearance, help, privacy, installed research models and versions | Explain local storage and model registry distribution; no account or inference server. |

## Journeys and state transitions

### Guided demonstration

`Screening → Demo introduction → simulated device discovery → simulated ready → user presses in-app demo control (standing in for physical goggle button) → Right-eye simulated acquisition → Left-eye simulated acquisition → sample result → exit`

The Demo label stays visible from entry through results. The result screen states that the values are fixed sample values, no image was captured, and no model ran. Demo data are never written to History. Cancel/back exits without implying an interrupted real examination.

### Local research image analysis

`Screening → Research analysis → choose Left or Right → select an image through the system picker → decode/orientation check → choose installed task models → run ONNX locally → show results for the selected eye → switch eye or clear image`

The user explicitly assigns each image to an eye. The interface does not infer laterality from an arbitrary image. Before inference, it states that the selected image is processed on this phone. Each result records the label's kind, raw score, model version and dataset/evidence scope. Model errors clear stale results for that eye and expose Retry. Unsupported quality or absent models appear as “not assessed” or “model not installed,” not as a low score.

### Model management

`Settings → Models → choose a task → review size/version/source → download → verify byte size and SHA-256 → activate`

The previous valid model remains active while an update downloads. A failed transfer/hash leaves it unchanged and exposes Retry/Delete. A newer manifest alone does not activate a model. File deletion is explicit and reports the app-storage effect. Distribution is through the registry; inference stays on-device.

### Future headset exam (locked)

`Unavailable → (future) Ready → physical button starts → Right acquisition → Right complete → Left acquisition → Left complete → processing → result / partial / recoverable error / safe stop`

The device, not repeated phone taps, owns the capture sequence. Phone UI observes versioned status and provides understandable recovery. It must never report “connected,” “scanned,” “gradable,” or “complete” without corresponding authenticated device evidence. Button debouncing, loss-of-link, user abort, low battery, camera fault, eye-side metadata and safe-stop behavior are defined by the future firmware contract.

### Settings and help

System/Light/Dark is a native preference. Help presents a manual and a reserved video placeholder labeled as unavailable until supplied; it does not imply a video exists. Bluetooth controls remain unavailable and do not open OS permissions. Privacy explains selected-image local processing and model downloads.

## Interaction/state acceptance matrix

| Event | Required result | Must not happen |
|---|---|---|
| User chooses Left or Right | Picker opens with the chosen eye retained | Results displayed under the opposite eye |
| User cancels picker | Return to prior screen; retain valid current state | Clear the other eye or create a blank “scan” |
| Image is invalid/unreadable | Explain and allow reselection | Show inference scores or stale previous scores as current |
| No disease model installed | Link to model management and explain requirement | Show all diseases as 0% or “normal” |
| Model returns wrong shape, NaN/Inf, or value outside `[0,1]` | Fail that model result and allow retry | Clamp invalid value into a plausible score |
| Network absent during model download | Keep already-installed models usable and offer retry | Block local inference or send image to a server |
| Model checksum/size mismatch | Reject candidate and retain prior valid version | Activate a partially downloaded model |
| App backgrounds during inference | Cancel or finish safely; re-open into a known state | Reuse stale completion state for a different image/eye |
| Future headset disconnects mid-capture | Firmware-controlled safe state, mark partial, show recovery | Invent completion or continue motor activity |

## Current-vs-future claims

**Implemented concept:** five-destination navigation, simulated scan presentation, image selection, local ONNX model analysis, model registry download/hash checks, system appearance setting.
**Not established by this map:** actual Bluetooth pairing, goggle capture, physical button events, clinically useful risk, persistent real exam history, signed registry authenticity, or validated user comprehension.
