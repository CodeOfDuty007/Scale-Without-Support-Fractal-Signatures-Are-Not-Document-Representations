#!/usr/bin/env bash
# Step 13 -- T10 pilot: persistent homology of sentence-embedding trajectories
# under five window geometries, then the length-partialled comparison.
# Needs sentence-transformers and gudhi (GPU environment; CPU inference suffices).
source "$(dirname "$0")/common.sh"
require_python_module gudhi "install the GPU environment: bash scripts/00_setup.sh --gpu"
require_python_module sentence_transformers "install the GPU environment: bash scripts/00_setup.sh --gpu"
# n_docs window_words stride_words  (as run for the paper)
for cfg in "150 40 20" "300 40 8" "300 40 40" "300 15 15" "300 10 10"; do
  read -r n w s <<< "$cfg"
  run "13_t10_w${w}_s${s}" experiments.t10_pilot_topology "$n" "$w" "$s"
done
run 13_t10_geometry experiments.t10_geometry_comparison
