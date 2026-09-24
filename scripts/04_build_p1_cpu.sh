#!/usr/bin/env bash
# Step 4 -- P1 CPU pass: Families A (fractal) and C (style), the 721 manifest
# shards read by every later step (720 generator members + the deduplicated
# human shard), the GPU prefix map, the P0 reconciliation and the corpus-level
# confidence intervals. CPU, ~45 min. Resumable.
source "$(dirname "$0")/common.sh"
require_file "$(workspace GAGLE_ZIP)" "run scripts/02_download_data.sh first"
require_file "$(WS_DATA)/p0_corpus_level_SH.json" "run scripts/03_build_p0.sh first"
run 04a_p1_cpu          fractalsig.pipeline.build_p1_feature_store_cpu
run 04b_shard_prefix    fractalsig.pipeline.backfill_gpu_shard_prefix
run 04c_reconcile_p0    fractalsig.pipeline.reconcile_p1_against_p0_v2
run 04d_x02_cis         fractalsig.detection.x02_recompute_cis
