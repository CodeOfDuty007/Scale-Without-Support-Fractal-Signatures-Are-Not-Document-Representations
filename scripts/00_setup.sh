#!/usr/bin/env bash
# Step 0 -- create .venv and install the pinned environment.
#   bash scripts/00_setup.sh          # CPU environment (all steps except 5 and 8)
#   bash scripts/00_setup.sh --gpu    # adds torch, sentence-transformers, gudhi
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3, 11), f"Python >= 3.11 required, found {sys.version.split()[0]}"'
[[ -d .venv ]] || "$PYTHON_BIN" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
if [[ "${1:-}" == "--gpu" ]]; then
  python -m pip install -r requirements-gpu.txt
else
  python -m pip install -r requirements.txt
fi
python -m pip install pytest==9.1.1
python -m pip install --no-deps -e .
python -m fractalsig.paths
echo "[setup] done. Activate with: source .venv/bin/activate"
