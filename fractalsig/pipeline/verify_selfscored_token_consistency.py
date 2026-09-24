#!/usr/bin/env python3
"""
Independent consistency check for the self-scored Family B pipeline
(build_p1_feature_store_gpu_selfscored.py).

WHY
  Fault 1 showed GAGLE's own stored token-count fields (gagle_mistral_total_tokens)
  don't match a fresh retokenization of the released text, and the gap scales with
  document length -- i.e. GAGLE's own counting process is suspect. The self-scored
  pipeline was built specifically to not depend on GAGLE's stored counts (D-01):
  it tokenizes row["text"] itself and uses len(ids) directly. This script re-derives
  that same number a SECOND, independent way (fresh tokenizer call, on completed
  shards only, cross-checked against the actual output file) to make sure we have
  not silently reintroduced an equivalent bug -- wrong text field, stale tokenizer,
  an off-by-one in the window math, etc.

WHAT IT CHECKS, per sampled row
  1. Re-tokenize row["text"] from the shard parquet (source of truth) with the SAME
     tokenizer the pipeline used.
  2. Recompute compute_align_window(len(ids)) -- the same call the pipeline makes.
  3. Compare the resulting scored-window token count against what the pipeline
     actually wrote to Gemma_Feature_Window_N_Scored / Mistral_Feature_Window_N_Scored
     in data/p1_feature_store_gpu_selfscored.jsonl.
  A mismatch here would mean the self-scored numbers are NOT reproducible from the
  raw text alone -- i.e. we would have replicated Fault 1's shape of bug ourselves.

USAGE
  python -m fractalsig.pipeline.verify_selfscored_token_consistency --n 40
"""
from fractalsig import paths as _paths  # noqa: E402
import argparse
import glob
import json
import os
import sys

import pyarrow.parquet as pq

from fractalsig.pipeline.build_p1_feature_store_gpu import compute_align_window  # noqa: E402

MANIFEST_DIR = os.environ.get(
    "GAGLE_MANIFEST_DIR",
    _paths.MANIFEST_SHARD_DIR_STR,
)
CHECKPOINT_PATH = os.path.join(_paths.WORKSPACE_STR, "data",
                                "p1_gpu_shard_checkpoint_selfscored.txt")
OUT_PATH = os.path.join(_paths.WORKSPACE_STR, "data",
                         "p1_feature_store_gpu_selfscored.jsonl")

TOKENIZER_IDS = {
    "gemma-2b": "google/gemma-2b",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
}


def completed_shards(limit):
    with open(CHECKPOINT_PATH) as f:
        names = [l.strip() for l in f if l.strip()]
    return names[:limit]


def load_output_rows_by_id(shard_basenames):
    """Only keep output rows whose id belongs to one of the sampled completed
    shards, to avoid loading the whole (growing) output file into memory."""
    wanted_ids = set()
    id_to_shard_text = {}
    for shard in shard_basenames:
        path = os.path.join(MANIFEST_DIR, shard)
        table = pq.read_table(path, columns=["id", "text"])
        for row in table.to_pylist():
            wanted_ids.add(row["id"])
            id_to_shard_text[row["id"]] = row["text"]

    found = {}
    with open(OUT_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d["id"] in wanted_ids:
                found[d["id"]] = d
    return found, id_to_shard_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--n-shards", type=int, default=8)
    args = ap.parse_args()

    shards = completed_shards(args.n_shards)
    if not shards:
        print("[FAIL] no completed shards found in checkpoint file", flush=True)
        return 1
    print(f"[info] sampling from {len(shards)} completed shards", flush=True)

    out_rows, id_to_text = load_output_rows_by_id(shards)
    print(f"[info] {len(out_rows)} output rows available to check", flush=True)

    from transformers import AutoTokenizer
    tokenizers = {k: AutoTokenizer.from_pretrained(v) for k, v in TOKENIZER_IDS.items()}

    sample_ids = list(out_rows.keys())[:args.n]
    mismatches = []
    checked = 0
    for doc_id in sample_ids:
        row = out_rows[doc_id]
        text = id_to_text.get(doc_id)
        if text is None:
            continue
        for scorer_key, prefix in (("gemma-2b", "Gemma"), ("mistral-7b", "Mistral")):
            selfscored_flag = row.get(f"{prefix}_SelfScored")
            if not selfscored_flag:
                continue  # this scorer didn't clear MIN_TOKENS for this row; nothing to check
            recorded_n = row.get(f"{prefix}_Feature_Window_N_Scored")
            if recorded_n is None:
                continue

            ids = tokenizers[scorer_key](text)["input_ids"]
            window = compute_align_window(len(ids))
            if window is None:
                mismatches.append((doc_id, scorer_key, "recomputed window=None but "
                                                          f"output says scored, recorded_n={recorded_n}"))
                continue
            # score_batch scores TARGETS, not window positions: position p
            # predicts ids[p+1], so target_idx==0 is unreachable (nothing
            # precedes it inside `ids`). build_p1_feature_store_gpu.score_batch
            # therefore clips to `t_start = max(start, 1)` and emits
            # `end - max(start, 1)` scores, NOT `end - start`.
            #
            # These differ only on the short-doc branch, where
            # compute_align_window returns (0, n_tokens): there the true count
            # is n_tokens - 1. On the long branch it returns
            # (WARMUP_DROP, CLIP_TO) = (64, 464) and max(64,1) == 64, so both
            # formulas give 400 and the discrepancy is invisible.
            #
            # The original `end - start` here is what produced job 70's
            # [FAIL] 8 mismatches (2026-08-22): every one was
            # `recomputed = recorded + 1` on a Used_Short_Window=True row.
            # That was this expectation being wrong, not the pipeline
            # mis-scoring -- see F-3 in PROJECT_STATE_AND_HPC_MIGRATION.md §2.2.
            expected_n = window[1] - max(window[0], 1)
            checked += 1
            if expected_n != recorded_n:
                mismatches.append((doc_id, scorer_key,
                                    f"recomputed n_scored={expected_n} != recorded {recorded_n} "
                                    f"(retokenized len={len(ids)})"))

    print(f"\n[info] checked {checked} (doc, scorer) pairs across {len(sample_ids)} sampled docs", flush=True)
    if mismatches:
        print(f"[FAIL] {len(mismatches)} mismatches found:", flush=True)
        for doc_id, scorer_key, msg in mismatches[:30]:
            print(f"  id={doc_id} scorer={scorer_key}: {msg}", flush=True)
        return 1

    print("[PASS] every checked (doc, scorer) pair's scored-window token count "
          "is independently reproducible from the raw text alone. No Fault-1-shaped "
          "bug detected in the self-scored pipeline.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
