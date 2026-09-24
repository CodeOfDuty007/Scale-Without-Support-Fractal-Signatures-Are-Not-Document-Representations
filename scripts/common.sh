#!/usr/bin/env bash
# Shared helpers sourced by every step script. Not meant to be run directly.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Optional user configuration.
if [[ -f .env ]]; then
  set -a; # shellcheck disable=SC1091
  source .env; set +a
fi

# Use the project virtualenv when present and none is active.
if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" && -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

PY="${PYTHON:-python}"
OUT_DIR="${FRACTALSIG_OUTPUT_DIR:-$REPO_ROOT/outputs}"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

log()  { printf '\n\033[1m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '\n\033[31m[error] %s\033[0m\n' "$*" >&2; exit 1; }

# run <log-name> <python -m module> [args...]  -- tees output to outputs/logs/<log-name>.log
run() {
  local name="$1"; shift
  log "$name: $*"
  "$PY" -m "$@" 2>&1 | tee "$LOG_DIR/$name.log"
  return "${PIPESTATUS[0]}"
}

require_python_module() {
  "$PY" -c "import $1" 2>/dev/null || fail "$2"
}

require_file() {
  [[ -e "$1" ]] || fail "missing $1 -- $2"
}

workspace() { "$PY" -c "from fractalsig import paths; print(paths.$1)"; }

# Paths resolved once per script.
WS_DATA() { workspace DATA_DIR; }
WS_FS()   { workspace FEATURE_STORE_DIR; }

require_gpu() {
  require_python_module torch "GPU environment missing: bash scripts/00_setup.sh --gpu"
  "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || fail "no CUDA device visible to torch"
}
