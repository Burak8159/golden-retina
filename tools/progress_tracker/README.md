# Golden Retina AI research tracker

Local, persistent dashboard for the current Golden Retina AI, clinical screening, adaptive-acquisition and software-research directive: 18 core phases, the mBRSET closeout sequence, then 10 mobile application phases.

## Start

From the repository root:

```sh
python3 tools/progress_tracker/server.py
```

The server binds only to `127.0.0.1` and defaults to port `8768`. Open the URL
printed in the terminal. Set `GOLDEN_RETINA_TRACKER_PORT` to choose another
local port. The dashboard refreshes project state every 10 seconds.

## Persistent updates

`status.json` is the local source of truth and is intentionally git-ignored.
The agent can update it by POSTing a complete JSON object to the same-origin
`/api/status` endpoint; the server replaces the file atomically. The endpoint
accepts only bounded JSON project-state objects and is reachable only through
the loopback-bound server. Keep private paths, credentials, row-level data,
and restricted dataset information out of this file.

The current operating rule is that the Mac runs the local Qwen interface and
project dashboard; the always-on ThinkPad orchestrates work, and the RTX 3050
laptop is the default training worker. Cloud L4 training is off unless the
owner explicitly changes that direction. Keep raw or restricted medical images
and patient-level records out of GitHub.

The dashboard separates historical evidence from current work. It does not
turn existing model artifacts, prior split metrics, or dataset registry entries
into newly verified results. Current campaign GPU-hours are tracked separately
from historical usage. The authenticated report through Nov 20 showed TRY 0 net after savings; same-day charges may lag, so campaign spend remains unverified until refreshed. Future cost estimates assume list price, not continuing credits.

## Home-compute controls

The top of the dashboard monitors the three-machine setup: this Mac and its
local Ollama runtime, the ThinkPad orchestrator, and the RTX 3050 training
worker. The dashboard's **Open Ollama in Terminal** action opens the native
Terminal app and runs the installed `ollama run <model>` CLI directly; prompts,
streaming, context, and terminal commands are handled by the real CLI, not a
browser-side chat imitation. The local runtime listens at `127.0.0.1:11434`.
After five idle minutes Ollama can unload the model. The stop action unloads active models, targets only the
verified Ollama listener on port 11434, checks its executable and owner, and
confirms whether the server is offline. A dashboard process started inside a
restricted runner may be allowed to unload models through Ollama's local API
but denied permission to signal the server process. In that case the UI reports
the partial result and provides a verified PID-specific `kill -TERM` command.
Relaunching the dashboard from a normal Terminal session should allow the
button to stop the runtime directly.

ThinkPad and RTX SSH targets are entered in the UI and saved only to the
git-ignored `devices.local.json`. SSH uses the Mac's existing OpenSSH keys and
agent with batch mode; it never saves a password. RTX status is probed through
the ThinkPad by default and reads `nvidia-smi` telemetry. A host key must first
be accepted interactively in Terminal. If Tailscale is unavailable on the Mac,
remote state is shown as unreachable rather than assumed online. The current
dashboard performs connection/GPU probes; a ThinkPad job-orchestration agent
is a separate integration and is not represented as installed until it is.
The complete architecture and current recovery state are documented in
[`docs/home_compute_dashboard_handoff.md`](../../docs/home_compute_dashboard_handoff.md).

The local tracker binds only to `127.0.0.1`. It is not exposed to the LAN or
tailnet. The Mac can still use local Qwen while traveling; ThinkPad/RTX controls
require a working Tailscale route from the Mac.

## mBRSET access boundary

The November 19, 2026 credentialing decision was not approved because the
reference email domain was not clearly linked to the listed organization; a
replacement application was subsequently reported as pending. The current
decision and actual dataset grant must be rechecked before use. Credentialing,
training, the dataset-specific DUA, and access grant are separate gates. Do not
download or analyze mBRSET before those gates are cleared. Its official record
also requires the actual release schema to be checked: an aggregate
`referable_dr` count does not prove the distributed CSV contains that field,
and image-center descriptions do not by themselves prove same-eye view pairs.
The handling sequence is documented in
[`docs/mbrset_followup_execution.md`](../../docs/mbrset_followup_execution.md);
the VM-only aggregate audit is at
[`scripts/cloud/audit_mbrset_metadata.py`](../../scripts/cloud/audit_mbrset_metadata.py).
The common evaluation rules are at
[`docs/evaluation_protocol_20261121.md`](../../docs/evaluation_protocol_20261121.md).
