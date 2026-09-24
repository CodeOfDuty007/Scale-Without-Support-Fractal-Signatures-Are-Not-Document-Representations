#!/usr/bin/env bash
# Step 8 -- zero-shot and supervised baselines on the P3 population:
# Fast-DetectGPT, Binoculars (Falcon-7B pair), RoBERTa-base fine-tuned per fold
# (8 folds; LODO folds re-run over multiple seeds), then paired significance.
# GPU: >= 24 GB.
source "$(dirname "$0")/common.sh"
require_gpu
require_file "$(WS_DATA)/splits_v1.json" "run scripts/07_run_p3_detection.sh first"
run 08a_fastdetectgpt fractalsig.baselines.fast_detectgpt_score
run 08b_binoculars    fractalsig.baselines.binoculars_score

for pair in "random random" "lodo big_patent" "lodo billsum" "lodo newsroom" \
            "lodo scientific_papers" "lodo wikipedia" "logo gemma_2b" "logo mistral_7b"; do
  read -r scheme fold <<< "$pair"
  run "08c_roberta_${scheme}_${fold}" fractalsig.baselines.roberta_finetune \
      --scheme "$scheme" --fold "$fold" --epochs 2 --batch-size 32
done
# Multi-seed LODO: 3 seeds per fold, 7 additional seeds for big_patent.
for fold in big_patent billsum newsroom scientific_papers wikipedia; do
  seeds="20260825 20260826 20260827"
  [[ "$fold" == big_patent ]] && seeds="$seeds 20260828 20260829 20260830 20260831 20260832 20260833 20260834"
  for seed in $seeds; do
    run "08d_roberta_lodo_${fold}_${seed}" fractalsig.baselines.roberta_finetune \
        --scheme lodo --fold "$fold" --epochs 2 --batch-size 32 --max-grad-norm 1.0 \
        --seed "$seed" --fp32-inference --diag-suffix "marginfp32_seed${seed}"
  done
done
run 08e_roberta_aggregate fractalsig.baselines.aggregate_roberta_multiseed
run 08f_paired_significance fractalsig.detection.part_d_paired_significance
run 08g_perplexity_tpr5     fractalsig.detection.perplexity_tpr5_fill
