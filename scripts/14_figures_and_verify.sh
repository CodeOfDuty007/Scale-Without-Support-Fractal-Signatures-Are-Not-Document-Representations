#!/usr/bin/env bash
# Step 14 -- render figures from the regenerated statistics and compare every
# regenerated JSON with reference_outputs/stats (exit status 1 on any mismatch).
source "$(dirname "$0")/common.sh"
run 14_figures experiments.make_figures
log "verify outputs/stats against reference_outputs/stats"
"$PY" -m fractalsig.verify "$@" | tee "$LOG_DIR/14_verify.log"
exit "${PIPESTATUS[0]}"
