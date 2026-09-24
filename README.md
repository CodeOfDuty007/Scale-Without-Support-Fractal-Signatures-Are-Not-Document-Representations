# Scale Without Support: Fractal Signatures Are Not Document Representations

Code for the paper. It measures two scaling exponents of a document's token-surprisal
trajectory — the Hurst exponent **H** (rescaled range) and the Hölder exponent **S**
(return probability) — and tests whether they are reliable *per-document* representations
for detecting machine-generated text.

**Main result.** The human/machine gap in the exponents is real at the corpus level, but at
~400 tokens a single document's estimate is dominated by estimator variance:

| Quantity (interleaved split-half, Spearman–Brown) | Gemma-2B | Mistral-7B | Source |
|---|---|---|---|
| Reliability ρ of per-document **S** | 0.063 | 0.050 | `reference_outputs/stats/t0_t1_MERGED_table.json` → `<scorer>_S.interleave.rho_pooled` |
| Reliability ρ of per-document **H** | 0.576 | 0.547 | same file → `<scorer>_H.interleave.rho_pooled` |
| Var(document term δ) vs Var(estimator term η), **S** | 0.0105 vs 0.328 | — | `t0_variance_components.json` → `features.gemma_S` |
| Tokens needed for ρ ≥ 0.60 (fBm floor, lower bound) | S: 6,400 · H: 800 | S: 6,400 · H: 800 | `t2_n_star_composition.json` → `*.N_star_fBm_floor` |
| RAID transfer: AUROC gain from the fractal family | +0.0240 | | `t5_raid_group_mean.json` → `reference_gap_all85_minus_all_minus_fractal31.gap` |

---

## Contents

```
fractalsig/            Python package (all code used for the paper)
  features/            fractal_math.py — the H and S estimators
  pipeline/            P0–P1: data download, feature stores (CPU and GPU passes), join, exit gates
  detection/           P2–P3: reliability gate, frozen splits, LightGBM detector, analyses
  baselines/           Fast-DetectGPT, Binoculars, RoBERTa fine-tuning
  robustness/          P4: edit-operator attack battery and five-detector comparison
  transfer/            P5: RAID ingestion, feature store, GAGLE→RAID transfer, auxiliary tasks
  diagnostics/         optional alignment audits (not on the reproduction path)
  paths.py             every file location, configurable by environment variable
  verify.py            compares regenerated outputs with reference_outputs/
experiments/           T0–T10 reliability battery and figure rendering
scripts/               numbered steps 00–14; run them in order
reference_outputs/     statistics (JSON) and figures produced for the paper
tests/                 unit tests (estimators, verifier, configuration)
docs/                  PIPELINE.md (step → module → output), DATA.md (files, formats, frozen artifacts)
```

## Requirements

- Linux or macOS, Python ≥ 3.11, `bash`.
- CPU steps: 64 GB RAM recommended (the paper's runs used 48–96 GB allocations). The quick check needs 4 GB.
- GPU steps (5, 8, 9, 10): one CUDA GPU with ≥ 24 GB; the 14B attacker in step 9 needs ≥ 48 GB.
- Disk: several tens of GB for intermediates; the GAGLE archive alone is 10.6 GB.
- A Hugging Face account with access to `google/gemma-2b` and `mistralai/Mistral-7B-v0.1`.

---

## Step 1 — Quick check (no download, CPU only, ~2 minutes)

```bash
git clone <this repository> && cd <repository>
bash scripts/00_setup.sh          # creates .venv with pinned versions
bash scripts/01_quickcheck.sh
```

This runs the unit tests, regenerates the synthetic noise-floor experiment (T2) from scratch,
recomputes the consolidated reliability tables from the shipped per-construction statistics,
compares every regenerated number with `reference_outputs/`, and re-renders all ten figures
into `outputs/quickcheck/figures/`. It ends with `quickcheck passed`.

## Step 2 — Full reproduction

Run the scripts in order. Each one checks its inputs, logs to `outputs/logs/`, and can be
re-run after an interruption (long stages resume from checkpoints).

| # | Command | What it produces | Hardware |
|---|---|---|---|
| 0 | `bash scripts/00_setup.sh [--gpu]` | virtual environment | — |
| 2 | `bash scripts/02_download_data.sh` | `workspace/gagle.zip` | network |
| 3 | `bash scripts/03_build_p0.sh` | per-document H, corpus-level S/H | CPU |
| 4 | `bash scripts/04_build_p1_cpu.sh` | P1 CPU feature store, 721 manifest shards, CIs | CPU |
| 5 | `bash scripts/05_build_p1_gpu.sh` | self-scored Family B, joined store `feature_store/v1_gagle.parquet` | GPU |
| 6 | `bash scripts/06_run_p2_reliability.sh` | split-half reliability report and feature whitelist | CPU |
| 7 | `bash scripts/07_run_p3_detection.sh` | frozen splits, detector results, persisted model | CPU |
| 8 | `bash scripts/08_run_baselines.sh` | Fast-DetectGPT, Binoculars, RoBERTa | GPU |
| 9 | `bash scripts/09_run_p4_robustness.sh` | attack battery and detector comparison | GPU |
| 10 | `bash scripts/10_run_p5_transfer.sh` | RAID feature store and transfer results | GPU |
| 11 | `bash scripts/11_run_reliability_battery.sh` | T0, T1, T2, T4, T7 statistics | CPU |
| 12 | `bash scripts/12_run_detection_controls.sh` | T3, T5, T8, T9 statistics | CPU |
| 13 | `bash scripts/13_run_topology_pilot.sh` | T10 statistics | CPU/GPU |
| 14 | `bash scripts/14_figures_and_verify.sh` | figures + comparison with `reference_outputs/` | CPU |

`docs/PIPELINE.md` lists every module each step runs, its inputs and outputs, and the paper
section it supports. `make <target>` aliases exist for every step (`make help`).

## Configuration

All locations default to folders inside the repository. To place data elsewhere, copy
`.env.example` to `.env` and set any of:

| Variable | Default | Meaning |
|---|---|---|
| `FRACTALSIG_WORKSPACE` | `./workspace` | downloaded and intermediate data |
| `FRACTALSIG_GAGLE_ZIP` | `$FRACTALSIG_WORKSPACE/gagle.zip` | GAGLE archive |
| `GAGLE_MANIFEST_DIR` | `$FRACTALSIG_WORKSPACE/gpu_manifest_shards` | manifest shards |
| `FRACTALSIG_OUTPUT_DIR` | `./outputs` | regenerated statistics, figures, logs |
| `FRACTALSIG_WORKERS` | all CPUs | worker processes (results do not depend on it) |

`python -m fractalsig.paths` prints the resolved configuration.

## Verifying results

```bash
python -m fractalsig.verify                 # every regenerated file in outputs/stats
python -m fractalsig.verify t0_t1_MERGED_table t2_n_star_composition
```

Every numeric value is compared with `reference_outputs/stats/` (relative tolerance 1e-6).
Both estimators are deterministic: `calc_S` enumerates start offsets exhaustively at
document length and uses a seeded sampler otherwise. The synthetic T2 experiment reproduces
the reference to a relative difference of 3e-13.

## Data and models

- **GAGLE** (`ibomohsin/gagle` on Hugging Face): 223,834 documents after deduplication
  (218,834 machine, 5,000 human) across five domains, scored by Gemma-2B and Mistral-7B.
- **RAID**: the benchmark's train split, used only in step 10.
- Models: `google/gemma-2b`, `mistralai/Mistral-7B-v0.1` (scorers); `tiiuae/falcon-7b` and
  `falcon-7b-instruct` (Binoculars); `roberta-base`; `sentence-transformers/all-MiniLM-L6-v2`;
  attackers `Qwen/Qwen2.5-{0.5B,1.5B,7B,14B}-Instruct`, `microsoft/Phi-3.5-mini-instruct`.

See `docs/DATA.md` for file formats, the frozen artifacts used by later steps, and sizes.

## Citation

```bibtex
@inproceedings{scale_without_support_2027,
  title     = {Scale Without Support: Fractal Signatures Are Not Document Representations},
  author    = {Anonymous},
  booktitle = {Submitted to the International Conference on Learning Representations (ICLR)},
  year      = {2027}
}
```

## License

MIT (see `LICENSE`). Datasets and models are subject to their own licenses.
