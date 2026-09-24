#!/usr/bin/env bash
# Step 12 -- controls on the frozen detection-stage artifacts (CPU):
#   T3 prompt-type identification under population control,
#   T5 RAID transfer after group-mean removal, T9 RAID complete-case,
#   T8 deployment metrics at realistic machine-text prevalence.
source "$(dirname "$0")/common.sh"
D="$(WS_DATA)"
for f in splits_v1.json p3_refit_predictions.jsonl p5_raid_features.jsonl p5_task3_prompt_type_id.json; do
  require_file "$D/$f" "run scripts/07 and scripts/10 first"
done
run 12a_t3_population_control experiments.t3_population_control
run 12b_t5_raid_group_mean    experiments.t5_raid_group_mean
run 12c_t8_deployment         experiments.t8_deployment_metrics
run 12d_t9_raid_complete_case experiments.t9_raid_complete_case
