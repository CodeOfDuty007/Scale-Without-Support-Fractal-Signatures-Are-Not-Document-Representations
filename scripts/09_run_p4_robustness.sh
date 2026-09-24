#!/usr/bin/env bash
# Step 9 -- P4 attack battery: 5,000 sampled documents x 7 edit operators
# (generated, repaired, rescored under both scorers), Family D, the five-detector
# comparison on one shared population, and the cross-attacker capability ladder
# (Qwen2.5 0.5B-14B, Phi-3.5-mini, Falcon-7B-Instruct). GPU: >= 48 GB for the
# 14B attacker; all other stages fit in 24 GB.
source "$(dirname "$0")/common.sh"
require_gpu
D="$(WS_DATA)"
require_file "$D/p3_refit_summary.json" "run scripts/07_run_p3_detection.sh first"
run 09a_sample fractalsig.robustness.p4_sample_and_cache_docs
for b in 1 2; do
  run "09b_edits_batch$b"   fractalsig.robustness.build_edits --batch "$b"
  run "09c_regen_batch$b"   fractalsig.robustness.p4_regenerate_variants --batch "$b"
  run "09d_merge_batch$b"   fractalsig.robustness.p4_merge_regen --batch "$b"
  run "09e_rescore_b${b}_gemma"   fractalsig.robustness.p4_rescore_edits --batch "$b" --scorer gemma
  run "09e_rescore_b${b}_mistral" fractalsig.robustness.p4_rescore_edits --batch "$b" --scorer mistral --gen_batch_size 4
done
run 09f_family_d    fractalsig.robustness.p4_compute_family_d
run 09g_family_d_v2 fractalsig.robustness.p4_compute_family_d_v2
run 09h_population  fractalsig.robustness.p4_task5_build_population
run 09i_tabular     fractalsig.robustness.p4_task5_tabular_features
run 09j_perplexity  fractalsig.robustness.p4_task5_perplexity_threshold
run 09k_fastdetect  fractalsig.robustness.p4_task5_fastdetectgpt
run 09l_binoculars  fractalsig.robustness.p4_task5_binoculars
run 09m_roberta     fractalsig.robustness.p4_task5_roberta
require_file "$D/p4_task5_tabular_thresholds.json" "tabular operating thresholds; see docs/DATA.md (known gap)"
require_file "$D/fastdetectgpt_eval.json" "Fast-DetectGPT per-fold evaluation; see docs/DATA.md (known gap)"
run 09n_final_comparison  fractalsig.robustness.p4_task5_final_comparison
run 09o_arm_decomposition fractalsig.robustness.p4_task5_arm_decomposition
run 09p_test2_masked      fractalsig.robustness.p4_test2_masked_vs_intact
run 09q_test3_margin      fractalsig.robustness.p4_test3_margin_check
run 09r_v2_features       fractalsig.robustness.p4_build_v2_edited_features
run 09s_edited_vs_clean   fractalsig.robustness.p4_edited_vs_clean_classifier
run 09t_b0_lists          fractalsig.robustness.p4_b0_enumerated_list_check
run 09u_b1_cross_attacker fractalsig.robustness.p4_b1_cross_attacker_fdgpt
run 09v_b2_generate       fractalsig.robustness.p4_b2_ladder_generate
run 09w_b2_rescore_gemma   fractalsig.robustness.p4_b2_ladder_rescore --scorer gemma
run 09w_b2_rescore_mistral fractalsig.robustness.p4_b2_ladder_rescore --scorer mistral --gen_batch_size 4
run 09x_b2_score_all      fractalsig.robustness.p4_b2_ladder_score_all
