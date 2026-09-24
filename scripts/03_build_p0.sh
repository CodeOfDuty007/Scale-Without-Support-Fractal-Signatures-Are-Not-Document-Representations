#!/usr/bin/env bash
# Step 3 -- P0: per-document H and corpus-level (pooled) S/H per (domain, label) cell.
# CPU, ~1 h. Outputs: data/p0_local_full_results.jsonl, data/p0_corpus_level_SH.json
source "$(dirname "$0")/common.sh"
require_file "$(workspace GAGLE_ZIP)" "run scripts/02_download_data.sh first"
run 03_p0 fractalsig.pipeline.build_p0_local
