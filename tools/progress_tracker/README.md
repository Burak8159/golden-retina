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

The current operating rule is that this Mac is for source, metadata, and
documentation checks only. Dataset downloads, extraction, and training must
happen on the authorized VM after cloud and cost preflight pass. The last
roadmap phases explicitly retrieve and verify VM outputs, publish only
permitted material to GitHub, and then shut down and audit the VM. Raw or
restricted medical images and row-level records are not GitHub deliverables.

The dashboard separates historical evidence from current work. It does not
turn existing model artifacts, prior split metrics, or dataset registry entries
into newly verified results. Current campaign GPU-hours are tracked separately
from historical usage. The authenticated report through Nov 20 showed TRY 0 net after savings; same-day charges may lag, so campaign spend remains unverified until refreshed. Future cost estimates assume list price, not continuing credits.

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
