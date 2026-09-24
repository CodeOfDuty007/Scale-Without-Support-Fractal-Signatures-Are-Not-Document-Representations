#!/usr/bin/env bash
# Step 5 -- P1 GPU pass: Family B (self-scored log-probability statistics under
# Gemma-2B and Mistral-7B) + embedding dispersion, human-side rescoring, merge,
# the CPU/GPU join into feature_store/v1_gagle.parquet, and both exit gates.
# GPU: >= 24 GB (paper run: one 48 GB MIG slice). Resumable per shard.
source "$(dirname "$0")/common.sh"
require_gpu
SHARDS="$(workspace MANIFEST_SHARD_DIR)"
require_file "$SHARDS/human_dedup_x01.parquet" "run scripts/04_build_p1_cpu.sh first"
run 05a_selfscored_verify fractalsig.pipeline.build_p1_feature_store_gpu_selfscored --verify-span 40
run 05b_selfscored_full   fractalsig.pipeline.build_p1_feature_store_gpu_selfscored
run 05c_token_consistency fractalsig.pipeline.verify_selfscored_token_consistency --n 60 --n-shards 10

# Human rows: one scorer per process, reading only the deduplicated human shard.
HUMAN_DIR="$(workspace WORKSPACE)/human_shard_only"
mkdir -p "$HUMAN_DIR"; ln -sf "$SHARDS/human_dedup_x01.parquet" "$HUMAN_DIR/human_dedup_x01.parquet"
run 05d_human_gemma   fractalsig.pipeline.rescore_human_familyb --scorer gemma-2b   --manifest-dir "$HUMAN_DIR"
run 05e_human_mistral fractalsig.pipeline.rescore_human_familyb --scorer mistral-7b --manifest-dir "$HUMAN_DIR"
run 05f_merge_human   fractalsig.pipeline.merge_human_familyb

run 05g_join_preflight fractalsig.pipeline.join_preflight
run 05h_join           fractalsig.pipeline.build_p1_join_feature_store
run 05i_exit_gate_v1   fractalsig.pipeline.p1_exit_gate
run 05j_doc_texts      fractalsig.pipeline.extract_all_doc_texts
