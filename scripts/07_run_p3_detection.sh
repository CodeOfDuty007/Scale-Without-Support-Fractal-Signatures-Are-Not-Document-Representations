#!/usr/bin/env bash
# Step 7 -- P3: frozen group-disjoint splits, the declared feature set, the
# LightGBM detector (class-weighted, calibrated at true prevalence) over all
# configs/arms/folds with its controls, the persisted refit, tables and the
# P3 follow-up analyses. CPU (16+ cores recommended).
source "$(dirname "$0")/common.sh"
D="$(WS_DATA)"; FS="$(WS_FS)"
require_file "$FS/v1_gagle.parquet" "run scripts/05_build_p1_gpu.sh first"
run 07a_splits fractalsig.detection.splits --store "$D/p1_feature_store_cpu.jsonl" --out "$D/splits_v1.json" --verify
run 07b_build_features fractalsig.detection.build_features
require_file "$D/p3_short_window_flags.jsonl" "per-row short-window flags (D-14) consumed by p3_train_eval; see docs/DATA.md"
run 07c_p3_train_eval   fractalsig.detection.p3_train_eval --out "$D/p3_t1_results.json"
run 07d_p1_exit_gate_v2 fractalsig.pipeline.p1_exit_gate_v2
run 07e_refit           fractalsig.detection.p3_refit_and_persist
run 07f_tables          fractalsig.detection.p3_make_tables
run 07g_logppl_arm      fractalsig.detection.p3_fractal_logppl_arm
run 07h_mlp_baseline    fractalsig.detection.p3_mlp_baseline
run 07i_seed_stability  fractalsig.detection.tabular_seed_stability
run 07j_ablation_fill   fractalsig.detection.ablation_grid_fill_billsum
run 07k_error_analysis  fractalsig.detection.part_c_error_analysis
run 07l_spread_reversal fractalsig.detection.spread_reversal_significance
run 07m_f5a_dropped     fractalsig.detection.f5a_characterize_dropped_llm_rows
run 07n_f5a_short_text  fractalsig.detection.f5a_score_short_text
