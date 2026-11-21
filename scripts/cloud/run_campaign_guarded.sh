#!/usr/bin/env bash
set -Eeuo pipefail

# Enforce the campaign's cumulative VM runtime cap and preserve logs. The
# training program remains responsible for saving its best checkpoint.
ROOT="${GOLDEN_RETINA_ROOT:-/data/golden-retina/workspace}"
CONFIG="${GOLDEN_RETINA_GUARD_CONFIG:-$ROOT/configs/campaign_guard_20261121.json}"
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 COMMAND [ARG ...]" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "BLOCKED: campaign guard config missing: $CONFIG" >&2
  exit 2
fi
if ! command -v timeout >/dev/null 2>&1 || ! command -v flock >/dev/null 2>&1; then
  echo "BLOCKED: GNU timeout and flock are required." >&2
  exit 2
fi
readarray -t GUARD < <(python3 - "$CONFIG" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    data = json.load(stream)
for key in ("start_utc", "gpu_runtime_cap_hours", "planned_spend_ceiling_try", "hard_spend_ceiling_try", "full_campaign_list_price_estimate_with_stress_try"):
    print(data[key])
PY
)
START_UTC="${GUARD[0]}"
CAP_HOURS="${GUARD[1]}"
PLANNED_CEILING="${GUARD[2]}"
HARD_CEILING="${GUARD[3]}"
FULL_ESTIMATE="${GUARD[4]}"
python3 - "$PLANNED_CEILING" "$HARD_CEILING" "$FULL_ESTIMATE" <<'PY'
import sys
planned, hard, estimate = map(float, sys.argv[1:])
if not (0 < planned <= hard and estimate <= planned):
    raise SystemExit("BLOCKED: recorded full-campaign estimate exceeds the spend guard.")
PY
START_EPOCH="$(date -u -d "$START_UTC" +%s)"
NOW_EPOCH="$(date -u +%s)"
CAP_SECONDS="$(python3 -c 'import sys; print(int(float(sys.argv[1])*3600))' "$CAP_HOURS")"
REMAINING="$((START_EPOCH + CAP_SECONDS - NOW_EPOCH))"
if (( REMAINING <= 0 )); then
  echo "BLOCKED: cumulative campaign runtime cap has been reached." >&2
  exit 3
fi
mkdir -p "$ROOT/logs" "$ROOT/locks"
exec 9>"$ROOT/locks/golden_retina_campaign.lock"
if ! flock -n 9; then
  echo "BLOCKED: another guarded campaign job is already running." >&2
  exit 4
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$ROOT/logs/guarded_${STAMP}.log"
printf 'Campaign runtime remaining: %ss of %ss\nEstimated full campaign spend: TRY %s (planned ceiling TRY %s; hard ceiling TRY %s)\nActual invoice: not verified by this guard\n' "$REMAINING" "$CAP_SECONDS" "$FULL_ESTIMATE" "$PLANNED_CEILING" "$HARD_CEILING" | tee "$LOG"
set +e
timeout --foreground --signal=INT --kill-after=120s "${REMAINING}s" "$@" 2>&1 | tee -a "$LOG"
STATUS="${PIPESTATUS[0]}"
set -e
printf '\nCommand exit status: %s\nFinished UTC: %s\n' "$STATUS" "$(date -u +%FT%TZ)" | tee -a "$LOG"
exit "$STATUS"
