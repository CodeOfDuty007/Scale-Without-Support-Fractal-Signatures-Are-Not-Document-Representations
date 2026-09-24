#!/usr/bin/env bash
# Step 1 -- no data download, CPU only, ~2-3 minutes.
#   (a) unit tests for the estimators and the verifier
#   (b) T2 synthetic noise floor regenerated from scratch and checked against reference
#   (c) T0/T1/T2 consolidation and T10 geometry comparison recomputed from shipped inputs
#   (d) all 10 figures re-rendered from the shipped statistics
source "$(dirname "$0")/common.sh"
QC="$OUT_DIR/quickcheck"
rm -rf "$QC"; mkdir -p "$QC/synthetic" "$QC/consolidation" "$QC/figures"

log "unit tests"
"$PY" -m pytest -q tests 2>&1 | tee "$LOG_DIR/01_tests.log"

log "T2 step 1: synthetic fGn noise floor, regenerated from scratch"
FRACTALSIG_STATS_DIR="$QC/synthetic" run 01_t2_noise_floor experiments.t2_step1_noise_floor

log "consolidation + T10 geometry comparison, recomputed from the shipped per-run statistics"
for f in reference_outputs/stats/t0_t1_*_stats.json reference_outputs/stats/t1_permutation_stats.json \
         reference_outputs/stats/t0_variance_components.json reference_outputs/stats/t2_step1_noise_floor.json \
         reference_outputs/stats/t10_pilot_topology_raw_*.json; do
  cp "$f" "$QC/consolidation/"
done
FRACTALSIG_STATS_DIR="$QC/consolidation" run 01_consolidate experiments.consolidate_t0_t1_t2
FRACTALSIG_STATS_DIR="$QC/consolidation" run 01_t10_geometry experiments.t10_geometry_comparison

log "verify against reference_outputs/stats"
"$PY" -m fractalsig.verify --new-dir "$QC/synthetic" t2_step1_noise_floor | tee "$LOG_DIR/01_verify_synthetic.log"
"$PY" -m fractalsig.verify --new-dir "$QC/consolidation" --strict \
  t0_t1_MERGED_table t2_n_star_composition t10_pilot_geometry_comparison | tee "$LOG_DIR/01_verify_consolidation.log"

log "re-render all figures from reference statistics"
run 01_figures experiments.make_figures --stats-dir reference_outputs/stats --out-dir "$QC/figures"
log "quickcheck passed. Figures: $QC/figures"
