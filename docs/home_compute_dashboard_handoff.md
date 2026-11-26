# Home Compute Dashboard Handoff

Updated 2026-11-26.

## Current machine roles

- **MacBook:** Codex workstation and loopback-only Golden Retina dashboard.
- **RTX 3050 laptop:** the only planned GPU training worker. It replaces Google
  Cloud L4 for future training. Connect directly from the Mac over SSH/Tailscale.
  No training is running until an actual remote process is started and its
  state is observed.

## SSH and Tailscale path

The dashboard supports a direct SSH/Tailscale route from the Mac to the RTX.
The target is stored only in the ignored
`tools/progress_tracker/devices.local.json`; passwords are never stored. Use
OpenSSH keys/agent and accept the host key interactively in Terminal before the
dashboard probes the machine. The RTX is marked online only after an SSH probe
succeeds. GPU telemetry is read with `nvidia-smi` when available.

While the tracker server is running, it holds an SSH control master and sends a
harmless `hostname` heartbeat every 15 minutes. SSH protocol keepalives run
every 60 seconds, with a one-minute reconnect retry after a detected drop. The
tracker closes its managed SSH master when the server exits. This keepalive
does not start or manage training.

The UI currently provides reachability and GPU telemetry only. It does not
launch, schedule, or manage remote training jobs. Google Cloud L4 is off; old L4
runtime and cost figures shown in the tracker are historical campaign records,
not current RTX time or future training estimates.

## Resume checklist

1. Confirm Tailscale is connected on the Mac if using tailnet addresses.
2. Configure the RTX SSH target in the tracker.
3. Accept its host key from Terminal, then save and check the target.
4. Confirm SSH status and RTX `nvidia-smi` telemetry before treating the machine
   as ready for GPU work.
5. Keep dataset files, patient-level records, and model checkpoints out of GitHub
   unless a source-specific rights review explicitly permits publication.

The tracker binds only to `127.0.0.1`; it is not exposed to the LAN or tailnet.
