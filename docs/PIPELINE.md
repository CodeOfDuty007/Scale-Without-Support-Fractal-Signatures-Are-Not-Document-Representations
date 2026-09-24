# Pipeline

Every step is a shell script in `scripts/` that runs the listed modules in order with
`python -m`. Paths are relative to the workspace (`$FRACTALSIG_WORKSPACE`, default
`./workspace`) unless they start with `outputs/`. Codes such as D-13, F-4 or X-01 in
module docstrings are the project's internal identifiers for design decisions (D),
fixed defects (F) and verification tasks (X); each docstring explains the item it cites.

## Stage overview

```
gagle.zip ─► P0 ─► P1 CPU ─► P1 GPU ─► join ─► P2 ─► P3 ─┬─► baselines
                    │                                   ├─► P4 attack battery
                    └─ manifest shards ─► T0–T10        └─► P5 RAID transfer
```

## Step 3 — P0 corpus-level exponents (CPU)

| Module | Output |
|---|---|
| `fractalsig.pipeline.build_p0_local` | `data/p0_local_full_results.jsonl`, `data/p0_corpus_level_SH.json` |

## Step 4 — P1 CPU feature pass (CPU)

| Module | Output |
|---|---|
| `fractalsig.pipeline.build_p1_feature_store_cpu` | `data/p1_feature_store_cpu.jsonl` (Families A and C), `gpu_manifest_shards/*.parquet` (720 member shards + `human_dedup_x01.parquet`) |
| `fractalsig.pipeline.backfill_gpu_shard_prefix` | `data/p1_gpu_shard_prefix_backfill.jsonl` |
| `fractalsig.pipeline.reconcile_p1_against_p0_v2` | `data/p1_p0_reconciliation_v2.json` |
| `fractalsig.detection.x02_recompute_cis` | `data/x02_confidence_intervals.json` |

## Step 5 — P1 GPU pass and join (GPU)

| Module | Output |
|---|---|
| `fractalsig.pipeline.build_p1_feature_store_gpu_selfscored` | `data/p1_feature_store_gpu_selfscored.jsonl` (Family B, each document scored by its own model) |
| `fractalsig.pipeline.verify_selfscored_token_consistency` | consistency report (stdout) |
| `fractalsig.pipeline.rescore_human_familyb` (×2 scorers) | `data/human_familyb_{gemma_2b,mistral_7b}.jsonl` |
| `fractalsig.pipeline.merge_human_familyb` | `data/p1_feature_store_gpu_selfscored_v2.jsonl` (223,834 rows) |
| `fractalsig.pipeline.join_preflight`, `build_p1_join_feature_store` | `feature_store/v1_gagle.parquet`, `feature_store/feature_manifest.json` |
| `fractalsig.pipeline.p1_exit_gate` | `feature_store/p1_exit_gate_report.json` |
| `fractalsig.pipeline.extract_all_doc_texts` | `data/all_doc_texts.jsonl` |

`build_p1_feature_store_gpu` is the shared model/scoring library imported by the GPU
modules; its own `main()` is the alignment-gated variant kept for audit.

## Step 6 — P2 reliability gate (CPU)

| Module | Output |
|---|---|
| `fractalsig.detection.p2_split_half_reliability` | `feature_store/p2_reliability_report.json`, `feature_store/fractal_feature_whitelist.json` |

## Step 7 — P3 detection (CPU)

| Module | Output |
|---|---|
| `fractalsig.detection.splits --verify` | `data/splits_v1.json` (group-disjoint random, 5 LODO, 2 LOGO folds) |
| `fractalsig.detection.build_features` | `data/p3_features.jsonl`, `feature_store/p3_feature_manifest.json` |
| `fractalsig.detection.p3_train_eval` | `data/p3_t1_results.json` (all configs × arms × folds, controls) |
| `fractalsig.pipeline.p1_exit_gate_v2` | `feature_store/p1_exit_gate_v2_report.json` (label-relationship checks) |
| `fractalsig.detection.p3_refit_and_persist` | `feature_store/p3_refit_models/*.joblib`, `data/p3_refit_predictions.jsonl`, `data/p3_refit_summary.json` |
| `fractalsig.detection.p3_make_tables` | `results/p3_tables.md` |
| `p3_fractal_logppl_arm`, `p3_mlp_baseline`, `tabular_seed_stability`, `ablation_grid_fill_billsum` | `data/*.json` |
| `part_c_error_analysis`, `spread_reversal_significance`, `f5a_*` | `data/*.json` |

## Step 8 — Baselines (GPU)

| Module | Output |
|---|---|
| `fractalsig.baselines.fast_detectgpt_score` | `data/fastdetectgpt_scores.jsonl` |
| `fractalsig.baselines.binoculars_score` | `data/binoculars_scores.jsonl` |
| `fractalsig.baselines.roberta_finetune` (8 folds; LODO × 3 seeds, big_patent × 10) | `data/roberta_summaries/*.json`, `data/roberta_predictions*.jsonl` |
| `aggregate_roberta_multiseed`, `part_d_paired_significance`, `perplexity_tpr5_fill` | `data/*.json` |

## Step 9 — P4 attack battery (GPU)

Seven edit operators are applied to 5,000 sampled documents in two batches; four
LLM-rewrite operators are regenerated after a repetition fix and merged back; every
variant and its clean snippet are rescored under both scorers. Five detectors are then
compared on one shared population. The capability ladder (B1, B2) repeats the attack with
six instruction-tuned attackers.

| Stage | Modules |
|---|---|
| sampling and edits | `p4_sample_and_cache_docs`, `build_edits`, `p4_regenerate_variants`, `p4_merge_regen` |
| rescoring and Family D | `p4_rescore_edits`, `p4_compute_family_d`, `p4_compute_family_d_v2` |
| five-detector comparison | `p4_task5_build_population`, `p4_task5_{tabular_features,perplexity_threshold,fastdetectgpt,binoculars,roberta}`, `p4_task5_final_comparison` |
| checks | `p4_task5_arm_decomposition`, `p4_test2_masked_vs_intact`, `p4_test3_margin_check`, `p4_build_v2_edited_features`, `p4_edited_vs_clean_classifier` |
| attacker ladder | `p4_b0_enumerated_list_check`, `p4_b1_cross_attacker_fdgpt`, `p4_b2_ladder_{generate,rescore,score_all}` |

## Step 10 — P5 RAID transfer (GPU for rescoring)

| Stage | Modules |
|---|---|
| ingest and sample | `ingest_raid`, `p5_sample_raid` |
| scoring and features | `p5_raid_rescore` (×2 scorers), `p5_raid_build_features`, `p5_exit_gate_raid` |
| transfer and checks | `p5_task4_gagle_to_raid_transfer`, `fractalscope_check_raid_intersection`, `fractalscope_check_p4_raid_all_minus_fractal` |
| auxiliary tasks | `p5_task2_gagle_generator_id`, `p5_task3_prompt_type_id`, `p5_task6_generator_family_id`, `p5_consolidate_generalization_table` |

## Steps 11–14 — Reliability battery (outputs in `outputs/stats/`)

| ID | Module | Output | Question |
|---|---|---|---|
| T1 | `experiments.run_all_t1_deterministic`, `run_permutation` | `t0_t1_<construction>_stats.json`, `t1_permutation_stats.json` | Is per-document reliability robust to how the document is split? |
| T0 | `experiments.t0_variance_components` | `t0_variance_components.json` | How large are the class/domain, document and estimator variance terms? |
| T2 | `experiments.t2_step1_noise_floor`, `t2_step2_real_text_bracket` | `t2_step1_noise_floor.json`, `t2_step2_real_text_bracket.json` | How many tokens does ρ ≥ 0.60 require? |
| — | `experiments.consolidate_t0_t1_t2` | `t0_t1_MERGED_table.json`, `t2_n_star_composition.json` | Consolidated tables |
| T4 | `experiments.t4_run_alt_estimators` | `t4_alt_estimators_reliability.json` | Does DFA or local Whittle fare better? |
| T7 | `experiments.t7_matched_length` | `t7_matched_length.json` | Does length matching change the conclusion? |
| T3 | `experiments.t3_population_control` | `t3_population_control.json` | Is prompt-type identification a population effect? |
| T5 | `experiments.t5_raid_group_mean` | `t5_raid_group_mean.json` | Is the RAID transfer gain a group-mean effect? |
| T8 | `experiments.t8_deployment_metrics` | `t8_deployment_metrics.json` | Operating points at realistic prevalence |
| T9 | `experiments.t9_raid_complete_case` | `t9_raid_complete_case.json` | RAID complete-case sensitivity |
| T10 | `experiments.t10_pilot_topology`, `t10_geometry_comparison` | `t10_pilot_*.json` | Does a topological representation escape the same limit? |

`experiments.make_figures` renders `outputs/figures/*.png`; `fractalsig.verify` compares
`outputs/stats/` with `reference_outputs/stats/`.

## Modules not on the fresh-run path

These are retained for provenance; a fresh run does not need them because the fixes they
applied are built into the modules above.

| Module | Purpose |
|---|---|
| `pipeline.recompute_style_features_gagle` | patched style columns of an existing store after the word-regex fix (now in the CPU pass) |
| `transfer.backfill_zwsp_flag_raid` | added the zero-width-space flag to an existing RAID store (now emitted on every build) |
| `pipeline.recompute_family_a_continuation_fix`, `pipeline.backfill_short_window_flag` | audit recomputations of Family A windows and short-window flags |
| `pipeline.reconcile_p1_against_p0` | reconciliation v1, superseded by v2 |
| `diagnostics.*` | alignment audits for the alignment-gated Family B variant |
| `experiments.t2_step0_validate_fgn` | generator sanity check; prints only |
