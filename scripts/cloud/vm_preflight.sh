#!/usr/bin/env bash
set -euo pipefail

# Non-destructive readiness check for the Golden Retina L4 VM.
# Run only after the campaign's billing gate is cleared. Dataset downloads and
# extraction must remain on this VM, never on the development Mac.

ROOT="${GOLDEN_RETINA_ROOT:-/data/golden-retina/workspace}"
DATA_ROOT="${GOLDEN_RETINA_DATA_ROOT:-/data/golden-retina}"

printf '%s\n' 'Golden Retina VM preflight' "workspace=$ROOT" "host=$(hostname)"
if [[ ! -d "$ROOT" ]]; then
  echo "BLOCKED: expected existing workspace $ROOT is absent; inspect/mount the preserved disk before creating paths." >&2
  exit 2
fi
printf '\n%s\n' '--- GPU ---'
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
  echo 'BLOCKED: nvidia-smi is unavailable.' >&2
  exit 2
fi

printf '\n%s\n' '--- Python ---'
python3 --version
PYTHON_ENV=""
if [[ -x "$ROOT/venv/bin/python" ]]; then
  PYTHON_ENV="$ROOT/venv"
elif [[ -x "$DATA_ROOT/venv/bin/python" ]]; then
  PYTHON_ENV="$DATA_ROOT/venv"
fi
if [[ -n "$PYTHON_ENV" ]]; then
  "$PYTHON_ENV/bin/python" --version
  "$PYTHON_ENV/bin/python" - <<'PY'
import importlib.util
import sys

print(f"environment_python={sys.executable}")
try:
    import torch
except Exception as exc:
    print(f"torch_import=FAILED ({type(exc).__name__}: {exc})")
else:
    print(f"torch={torch.__version__}; cuda_available={torch.cuda.is_available()}; cuda_runtime={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(0)}")
for module in ("torchvision", "timm", "transformers", "sklearn", "PIL"):
    print(f"{module}_installed={importlib.util.find_spec(module) is not None}")
PY
else
  echo "NOTICE: no existing venv at $ROOT/venv or $DATA_ROOT/venv; inspect existing environments before creating one."
fi

printf '\n%s\n' '--- Disk ---'
mkdir -p "$ROOT"/{models,logs,checkpoints,reports,manifests,external}
df -h "$ROOT"

printf '\n%s\n' '--- Existing data roots (names only) ---'
for path in "$ROOT/external" "$ROOT/results" "$ROOT/models"; do
  if [[ -d "$path" ]]; then
    printf '%s: ' "$path"
    find "$path" -mindepth 1 -maxdepth 1 -type d -printf '%f ' 2>/dev/null || true
    printf '\n'
  else
    printf '%s: absent\n' "$path"
  fi
done

printf '\n%s\n' 'Preflight complete. This script does not download data, install packages, or start training.'
