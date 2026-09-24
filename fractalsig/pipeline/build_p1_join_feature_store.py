#!/usr/bin/env python3
"""
P1 Feature Store -- join. Merges the CPU pass's output (Families A + C,
data/p1_feature_store_cpu.jsonl) with the GPU pass's output (Family B +
embedding dispersion, data/p1_feature_store_gpu.jsonl, produced on the
separate GPU machine by build_p1_feature_store_gpu.py and copied back here)
into the single versioned artifact every later phase reads from:
feature_store/v1_gagle.parquet.

HARD INTEGRITY GATE -- id-set equality, not a best-effort merge
  This is the join-time half of the CPU/GPU split's integrity guarantee
  promised before any of this was built: the CPU pass emits a GPU manifest
  keyed by id, the GPU pass's own output MUST carry exactly those ids back,
  and this script asserts `set(cpu_ids) == set(gpu_ids)` before writing
  anything. A silent partial join (pandas' default inner-join-and-move-on)
  would hide exactly the failure mode this two-machine split was designed
  to guard against -- some ids present on one side and not the other,
  because a doc that failed MIN_TOKENS differently between the two passes
  (different tokenizer segmenting a boundary case differently), or a stray
  re-run that skipped a shard. If the sets don't match, this script prints
  the missing/extra ids (or a sample, if there are many) and EXITS WITHOUT
  WRITING v1_gagle.parquet, rather than writing a silently-incomplete store.

CROSS-SCORER FEATURES COMPUTED HERE, NOT IN EITHER PASS
  Cross-perplexity (Gemma vs. Mistral log-PPL ratio) needs both scorers'
  Family B rows for the same document in scope simultaneously, which never
  happens inside build_p1_feature_store_gpu.py's per-scorer batching loop.
  It belongs at the join, where both sides are finally in the same row.

RECONCILIATION AGAINST P0 (this script does NOT do this -- see
reconcile_against_p0.py, run separately after this completes)
  The P1 spec's own exit gate requires recomputing corpus-level S/H from
  this feature store's Family A columns and diffing against the P0 ground
  truth (data/p0_corpus_level_SH.json). That reconciliation pools ALL
  documents in a (domain, label) cell through calc_S's list-of-trajectories
  mode -- it needs the ORIGINAL per-document trajectories, not just the
  point-estimate Ŝ/Ĥ columns this feature store carries, so it re-reads
  the CPU pass's own output directly rather than this joined parquet.

USAGE
    python -m fractalsig.pipeline.build_p1_join_feature_store
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
import sys

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(_paths.WORKSPACE_STR, "data")
FS_DIR = os.path.join(_paths.WORKSPACE_STR, "feature_store")
CPU_PATH = os.path.join(DATA_DIR, "p1_feature_store_cpu.jsonl")
# Family B production path. Repointed 2026-08-23.
#
# This used to be `p1_feature_store_gpu.jsonl` -- the output of the
# ALIGNMENT-GATED pass. Per D-01 that is not the production path: the gated run
# is Option A, whose Family B coverage is prompt_type-correlated (MNAR), and
# D-06 keeps it as AUDIT EVIDENCE only. The production path is Option B,
# self-scored, 100% coverage. That file also does not exist on this box.
#
# `_v2` is the post-X-01 union: the 218,834 LLM rows carried over byte-identical
# from job 56, plus the 5,000 re-scored human rows from job 89. NOT the raw
# selfscored file, which still carries the 11,696 retired pre-X-01 human rows
# alongside the LLM side and would reinstate the population X-01 removed.
GPU_PATH = os.environ.get(
    "P1_GPU_STORE",
    os.path.join(DATA_DIR, "p1_feature_store_gpu_selfscored_v2.jsonl"))
OUT_PARQUET = os.path.join(FS_DIR, "v1_gagle.parquet")
MANIFEST_PATH = os.path.join(FS_DIR, "feature_manifest.json")

METADATA_COLS = ["id", "split", "domain", "Label", "prompt_type", "temp"]


def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def assert_id_sets_match(cpu_df, gpu_df):
    cpu_ids = set(cpu_df["id"])
    gpu_ids = set(gpu_df["id"])
    if cpu_ids == gpu_ids:
        print(f"[ok] id sets match exactly: {len(cpu_ids)} ids on both sides.")
        return
    missing_from_gpu = cpu_ids - gpu_ids
    extra_in_gpu = gpu_ids - cpu_ids
    print(f"[FAIL] id sets do NOT match. cpu={len(cpu_ids)} gpu={len(gpu_ids)} "
          f"missing_from_gpu={len(missing_from_gpu)} extra_in_gpu={len(extra_in_gpu)}")
    if missing_from_gpu:
        print(f"  sample missing_from_gpu: {sorted(missing_from_gpu)[:10]}")
    if extra_in_gpu:
        print(f"  sample extra_in_gpu: {sorted(extra_in_gpu)[:10]}")
    print("\n[abort] Not writing v1_gagle.parquet. This means the GPU pass either didn't "
          "process every id the CPU pass's manifest shipped, or processed ids the CPU pass "
          "never emitted -- investigate before joining; do not proceed with a partial join.")
    sys.exit(1)


def assert_no_duplicate_ids(df, name):
    dup = df["id"].duplicated()
    if dup.any():
        dup_ids = df.loc[dup, "id"].unique()[:10]
        print(f"[FAIL] {name} has {dup.sum()} duplicate id rows. Sample: {list(dup_ids)}")
        sys.exit(1)
    print(f"[ok] {name}: 0 duplicate ids across {len(df)} rows.")


def add_cross_perplexity(df):
    have_both = df["Gemma_LogPPL_Mean"].notna() & df["Mistral_LogPPL_Mean"].notna()
    df["Cross_PPL_Ratio"] = float("nan")
    # LogPPL_Mean is a MEAN NEGATIVE log-prob (see build_p1_feature_store_gpu's
    # LogPPL_Mean = -mean(log_probs)); ratio of exp(mean_neg_logprob) between
    # scorers is exp(gemma) / exp(mistral) = exp(gemma - mistral) -- a
    # documented simplification of Binoculars-style cross-scoring (see
    # build_p1_feature_store_gpu.py's docstring for why literal cross-scoring
    # isn't well-defined across two different tokenizers/vocabularies).
    df.loc[have_both, "Cross_PPL_Ratio"] = np.exp(
        df.loc[have_both, "Gemma_LogPPL_Mean"] - df.loc[have_both, "Mistral_LogPPL_Mean"]
    )
    return df


def main():
    if not os.path.exists(CPU_PATH):
        print(f"[abort] {CPU_PATH} not found -- run build_p1_feature_store_cpu.py to completion first.")
        sys.exit(1)
    if not os.path.exists(GPU_PATH):
        print(f"[abort] {GPU_PATH} not found -- run build_p1_feature_store_gpu.py on the GPU machine "
              f"and copy its output back to this path first.")
        sys.exit(1)

    os.makedirs(FS_DIR, exist_ok=True)
    cpu_df = load_jsonl(CPU_PATH)
    gpu_df = load_jsonl(GPU_PATH)
    print(f"[info] loaded {len(cpu_df)} CPU rows, {len(gpu_df)} GPU rows.")

    assert_no_duplicate_ids(cpu_df, "CPU output")
    assert_no_duplicate_ids(gpu_df, "GPU output")
    assert_id_sets_match(cpu_df, gpu_df)

    gpu_only_cols = [c for c in gpu_df.columns if c not in METADATA_COLS]
    merged = cpu_df.merge(gpu_df[["id"] + gpu_only_cols], on="id", how="inner", validate="one_to_one")
    if len(merged) != len(cpu_df):
        print(f"[FAIL] merge produced {len(merged)} rows, expected {len(cpu_df)} -- "
              f"id sets matched but merge still dropped/duplicated rows. Investigate before proceeding.")
        sys.exit(1)

    merged = add_cross_perplexity(merged)

    merged.to_parquet(OUT_PARQUET, index=False)
    print(f"[done] wrote {len(merged)} rows x {len(merged.columns)} columns to {OUT_PARQUET}")

    manifest = {
        "n_rows": len(merged),
        "n_columns": len(merged.columns),
        "columns": sorted(merged.columns.tolist()),
        "sources": {"cpu_pass": CPU_PATH, "gpu_pass": GPU_PATH},
        "families": {
            "A_fractal": "Gemma_/Mistral_ S,H point estimates + windowed summaries + scorer divergence "
                         "(from build_p1_feature_store_cpu.py, GAGLE's own precomputed token_scores). "
                         "Per-doc/per-window S is flagged Holder_PerDoc_Unreliable=True; computed anyway "
                         "per the P1 spec, P2 decides whether to keep it.",
            "B_statistical": "Gemma_/Mistral_ LogPPL moments, GLTR rank-bucket fractions, entropy, "
                              "Cross_PPL_Ratio (from build_p1_feature_store_gpu.py, exact-logit forward "
                              "pass on GAGLE's own text; Alignment_* columns record the per-doc "
                              "cross-check against GAGLE's stored scores -- filter on these before "
                              "trusting a row if alignment quality matters for a given analysis).",
            "C_style": "Sent_/Para_/TTR/Bigram/Discourse_Marker_* (text-only) + Embedding_Dispersion "
                       "(computed on the GPU pass, folded into the same loop as Family B -- see "
                       "build_p1_feature_store_cpu.py's docstring for why it isn't computed CPU-side). "
                       "Word budget is per-document, ratio-derived to match Family A's token window -- "
                       "NOT full text -- specifically to avoid a length confound between Human/LLM text.",
        },
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
