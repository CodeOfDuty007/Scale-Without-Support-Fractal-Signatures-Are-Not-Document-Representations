#!/usr/bin/env bash
# Step 6 -- P2: split-half reliability of every fractal feature (interleaved
# halves, Spearman-Brown) and the pre-registered rho >= 0.60 whitelist. CPU.
# Outputs: feature_store/p2_reliability_report.json, fractal_feature_whitelist.json
source "$(dirname "$0")/common.sh"
require_file "$(WS_FS)/v1_gagle.parquet" "run scripts/05_build_p1_gpu.sh first"
run 06_p2_reliability fractalsig.detection.p2_split_half_reliability
