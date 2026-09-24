#!/usr/bin/env bash
# Step 11 -- reliability battery (CPU; many cores recommended):
#   T1 split-half reliability under 8 deterministic constructions + 20 permutations,
#   T2 synthetic noise floor and real-text bracket, T4 alternative estimators,
#   T7 matched-length control, T0 variance decomposition, consolidation.
# Outputs: outputs/stats/*.json
source "$(dirname "$0")/common.sh"
SHARDS="$(workspace MANIFEST_SHARD_DIR)"
compgen -G "$SHARDS/*.parquet" > /dev/null || fail "no manifest shards in $SHARDS -- run scripts/04_build_p1_cpu.sh"
require_file "$(WS_DATA)/splits_v1.json" "run scripts/07_run_p3_detection.sh first"
run 11a_t1_deterministic  experiments.run_all_t1_deterministic
run 11b_t1_permutation    experiments.run_permutation 20
run 11c_t2_noise_floor    experiments.t2_step1_noise_floor
run 11d_t2_real_text      experiments.t2_step2_real_text_bracket
run 11e_t4_alt_estimators experiments.t4_run_alt_estimators
run 11f_t7_matched_length experiments.t7_matched_length
run 11g_t0_variance       experiments.t0_variance_components
run 11h_consolidate       experiments.consolidate_t0_t1_t2
