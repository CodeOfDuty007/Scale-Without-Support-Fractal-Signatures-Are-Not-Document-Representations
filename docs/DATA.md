# Data

## Sources

| Dataset | Obtain | Place at | Used in |
|---|---|---|---|
| GAGLE (`ibomohsin/gagle`, `gagle.zip`, 10.6 GB) | `bash scripts/02_download_data.sh` (accept the dataset terms on Hugging Face and run `hf auth login` first) | `workspace/gagle.zip` | steps 3–9, 11–13 |
| RAID (train split, `train.csv`) | the RAID benchmark release | `workspace/data/raid_raw/train.csv` | steps 10, 12 |

GAGLE contains one human article and its machine continuations (Gemma-2B and Mistral-7B,
three temperatures, eight prompt types) per source article, over five domains
(big_patent, billsum, newsroom, scientific_papers, wikipedia), with per-token
log-probabilities under both models. After cross-split deduplication of the human side the
corpus has 223,834 documents: 218,834 machine and 5,000 human.

## Workspace layout

```
workspace/
  gagle.zip
  gpu_manifest_shards/   721 parquet shards (per-document token scores, text, metadata)
  data/                  per-row stores (*.jsonl), splits, detector outputs, P4/P5 artifacts
  feature_store/         joined feature store (v1_gagle.parquet), manifests, gate reports, models
  results/               rendered tables
outputs/
  stats/  figures/  logs/
```

All of `workspace/` and `outputs/` are git-ignored.

## Row counts used as checks

| File | Rows |
|---|---|
| `data/p1_feature_store_cpu.jsonl` | 223,834 |
| `gpu_manifest_shards/` | 721 files |
| `data/p1_feature_store_gpu_selfscored_v2.jsonl` | 223,834 |
| `data/human_familyb_{gemma_2b,mistral_7b}.jsonl` | 5,000 each |
| `data/p3_features.jsonl` | 223,834 |

## Frozen inputs distributed with the data release

The steps below read these files as fixed inputs. They are distributed as frozen artifacts
alongside the data release; place them at the listed paths before running the step.

| File | Read by | Content |
|---|---|---|
| `data/p3_short_window_flags.jsonl` | step 7 (`p3_train_eval`) | per-id flags `Gemma_Used_Short_Window_RawLen`, `Mistral_Used_Short_Window_RawLen` |
| `data/fastdetectgpt_eval.json` | step 9 (`p4_task5_final_comparison`) | Fast-DetectGPT metrics per split/fold |
| `data/p4_task5_tabular_thresholds.json` | steps 9, 10 | operating thresholds of the tabular detector |
| `data/p5_raid_source_ids.json` | step 10 (`p5_task6_generator_family_id`) | RAID id → source-article id |

`reference_outputs/stats/t2_step2_full_doc_estimates.json` is shipped as a reference output.

## Reference outputs

`reference_outputs/stats/` holds the statistics behind every table and figure of the
reliability battery (T0–T10), and `reference_outputs/figures/` the rendered figures.
`python -m fractalsig.verify` compares regenerated statistics with these files.
