#!/usr/bin/env bash
# Step 10 -- P5 generalization: stratified RAID sample, rescoring under both
# scorers, the RAID feature store and its label-relationship gate, then GAGLE ->
# RAID transfer with the persisted P3 model and the three auxiliary tasks.
# GPU for 10c only.
source "$(dirname "$0")/common.sh"
D="$(WS_DATA)"
require_file "$D/raid_raw/train.csv" "place the RAID train split at data/raid_raw/train.csv (docs/DATA.md)"
run 10a_ingest  fractalsig.transfer.ingest_raid
run 10b_sample  fractalsig.transfer.p5_sample_raid
require_gpu
run 10c_rescore_gemma   fractalsig.transfer.p5_raid_rescore --scorer gemma
run 10c_rescore_mistral fractalsig.transfer.p5_raid_rescore --scorer mistral --gen_batch_size 4
run 10d_features  fractalsig.transfer.p5_raid_build_features
run 10e_exit_gate fractalsig.transfer.p5_exit_gate_raid --input "$D/p5_raid_features.jsonl"
require_file "$D/p4_task5_tabular_thresholds.json" "tabular operating thresholds; see docs/DATA.md (known gap)"
run 10f_task4_transfer   fractalsig.transfer.p5_task4_gagle_to_raid_transfer
run 10g_raid_intersection fractalsig.transfer.fractalscope_check_raid_intersection
run 10h_all_minus_fractal fractalsig.transfer.fractalscope_check_p4_raid_all_minus_fractal
run 10i_task2_generator  fractalsig.transfer.p5_task2_gagle_generator_id
run 10j_task3_prompt     fractalsig.transfer.p5_task3_prompt_type_id
require_file "$D/p5_raid_source_ids.json" "RAID source-id map for task 6; see docs/DATA.md (known gap)"
run 10k_task6_family     fractalsig.transfer.p5_task6_generator_family_id
run 10l_consolidate      fractalsig.transfer.p5_consolidate_generalization_table
