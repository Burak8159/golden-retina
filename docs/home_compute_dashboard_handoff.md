# Home Compute Dashboard Handoff

Updated 2026-11-26.

## Intended machine roles

- **MacBook:** portable workstation; runs the local Qwen model through Ollama
  (MLX runtime) and hosts the loopback-only project dashboard. The dashboard
  opens native Terminal and launches the installed `ollama run <model>` CLI.
  Model prompts, streamed answers, conversation context, and terminal commands
  are handled by that real process; the dashboard does not proxy or reconstruct
  chat. The Ollama service listens locally at `127.0.0.1:11434`.
- **ThinkPad:** always-on home orchestrator. It should coordinate project jobs
  and provide the SSH hop to the GPU laptop. A job-orchestration agent has not
  yet been implemented or verified.
- **RTX 3050 laptop:** dedicated home GPU worker for training. The ThinkPad
  reaches it over SSH; the traveling Mac reaches the ThinkPad over Tailscale.
  RTX training is the intended default; cloud L4 training is off unless the
  owner explicitly changes that instruction.

The dashboard's device cards and SSH fields are implemented. They probe reachability
and, for the RTX, `nvidia-smi` telemetry. They do not yet submit or control remote
training jobs. Device names, IP addresses, and SSH targets belong in the ignored
`tools/progress_tracker/devices.local.json`, never in Git.

## Local AI stop behavior

The dashboard stop action is deliberately narrow:

1. Ask Ollama to unload each currently loaded model.
2. Verify the listener on local port `11434`, its executable name, and its UID.
3. Send `SIGTERM` only to the verified Ollama listener.
4. If it does not exit, re-check the same listener identity before considering
   `SIGKILL`, then verify whether the service is offline.

During the 2026-11-26 troubleshooting, Ollama was confirmed by `lsof` as the
Homebrew binary at version `0.40.18`, listening on `127.0.0.1:11434` under the
current user's UID. The dashboard's process environment returned macOS
`Operation not permitted` when it tried to signal that process. Ollama's model
unload API remains a separate operation and can release model memory even when
the dashboard cannot terminate the server.

The UI now reports model unload and runtime termination separately. On a denied
signal it provides a shell command that rechecks the exact port listener, PID,
current UID, and `ollama` executable before sending `SIGTERM`. If the button
should terminate Ollama directly, launch the dashboard from an ordinary macOS
Terminal session (outside a restricted app command runner) and retry. Do not
add `sudo`, broad `pkill`, or process-name wildcards.

## Connection state and resume checklist

Remote ThinkPad/RTX telemetry is not established until Tailscale is working on
the Mac, both SSH targets are configured locally, the host keys are accepted,
and the dashboard shows successful SSH probes. A previously observed Tailscale
command wrapper pointed to a missing app executable, and neither remote target
was configured in the dashboard at that time. Recheck the live machine before
relying on those observations.

1. Start `tools/progress_tracker/server.py` from a normal Terminal on the Mac.
2. Verify Ollama state; keep Qwen unloaded unless the user wants to chat.
3. Restore Tailscale on the Mac and confirm the ThinkPad is reachable.
4. Save the ThinkPad SSH target in the dashboard, accept its host key in
   Terminal, and verify the online state.
5. Save the RTX target as reachable through the ThinkPad, accept its host key,
   and verify both SSH and GPU telemetry.
6. Only then implement/enable job orchestration on the ThinkPad. Keep the Mac
   as the local LLM and user interface, and the RTX as the training worker.

The tracker binds only to loopback. No passwords, model prompts, local device
targets, or machine-specific addresses should be committed. Current project
progress and actual device status remain separate facts; refresh each from the
running machines before a new training round.

## Native Ollama terminal behavior

The “Open Ollama in Terminal” button validates the selected model against the
locally installed Ollama model list, builds the fixed `ollama run <model>`
command, and asks macOS Terminal to run it. It does not accept shell commands
from the browser. If macOS denies Terminal automation, the UI displays the
exact command to type into Terminal manually. The browser chat proxy was removed
so terminal sessions retain the actual Ollama CLI behavior and state.
