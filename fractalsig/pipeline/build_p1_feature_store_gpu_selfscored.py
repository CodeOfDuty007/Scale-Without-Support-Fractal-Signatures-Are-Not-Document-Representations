#!/usr/bin/env python3
"""
P1 Feature Store -- GPU pass, Family B, SELF-SCORED mode (Option B, decision
D-01 in PROJECT_STATE_AND_HPC_MIGRATION.md, 2026-08-18).

WHY THIS FILE EXISTS, SEPARATE FROM build_p1_feature_store_gpu.py
  The alignment-gated script tries to REPRODUCE GAGLE's own precomputed
  score arrays (two candidates, per-document offset sweep, 0.98 Pearson
  gate). That path is now known to have two independent faults on the
  Mistral side (the internal verification notes):
    Fault 1 -- MISTRAL_MODEL_ID is probably the wrong checkpoint (r~0.5 at
      the structurally correct offset, not ~0.2 -- a sibling-checkpoint
      signature, not a wrong-model signature). Status as of this run: being
      tested in parallel by tokenizer_checkpoint_screen.py --per-row
      --dump-json; NOT a blocker for this script, since self-scoring never
      needs to match GAGLE's stored numbers.
    Fault 2 -- mistral-GENERATED documents' stored arrays are objectively
      too short to be that text's tokenization, under BOTH scorers, on
      byte-identical text. Proven (gemma's checkpoint is independently
      confirmed correct elsewhere, and still fails here). This is GAGLE's
      own defect and has NO fix via checkpoint substitution or a wider
      offset sweep (already exhaustively tried, see the report above).
  D-01's resolution: stop trying to reproduce GAGLE's arrays for Family B.
  Self-score every document, every prompt_type, both scorers, ourselves --
  this doesn't need GAGLE's stored numbers to be trustworthy at all, so
  neither fault blocks it. Literature check (reports/2026-08-18_literature_
  precedents...) confirms this is the standard fix for label-correlated
  (MNAR) feature missingness: eliminate the mechanism, don't manage it.

WHAT "SELF-SCORED" MEANS HERE, CONCRETELY
  No candidates. No prefix. No per-document offset sweep. No alignment
  gate. Per document, per scorer: tokenize `text` ALONE, take the SAME
  window Family A consumes -- WARMUP_DROP:CLIP_TO (tokens [64:464]) for
  documents whose tokenization reaches CLIP_TO=464, else the whole
  tokenization (the same short-doc fallback branch slice_scores/
  compute_align_window already use everywhere else in this pipeline),
  MIN_TOKENS=320 as the usability floor -- one real forward pass over that
  window, no more. This is D-02's instruction applied literally: "score
  the same window Family A consumes", which is what makes Family A and B
  span-identical BY CONSTRUCTION rather than by a matched-post-hoc check.

  For continue_it/continue_pt specifically: after Family A's own
  correction (recompute_family_a_continuation_fix.py), Family A's
  corrected window is `compute_boundary(prefix_len, text_len, gagle_total)
  + WARMUP_DROP` into GAGLE's PREFIX+TEXT array -- which, in the common
  (delta=0) case, lands within ~1 token of "text-alone, WARMUP_DROP:CLIP_TO"
  (boundary approx equals prefix_len, so boundary+WARMUP_DROP, reindexed
  into text-alone coordinates, approx equals WARMUP_DROP+1). This script
  does NOT recompute that per-document boundary or touch `prefix` at all --
  it scores text alone uniformly for every prompt_type and both scorers.
  `--verify-span` below exists specifically to measure, not assume, how
  close that approximation is for continue_it/continue_pt, gemma-2b (the
  only combination where Family A's window is not already exactly
  text-alone WARMUP_DROP:CLIP_TO). Reported honestly as a measured
  small-token discrepancy, not claimed as exact byte-identity.

WHAT THIS BUYS
  100% Family B coverage (every document that clears MIN_TOKENS, both
  scorers, all 8 prompt_types) instead of the alignment-gated path's
  partial, prompt_type-and-generator-correlated coverage. Also far
  cheaper: at most CLIP_TO=464 tokens of context per forward pass, vs. the
  alignment-gated path's up to MAX_CANDIDATE_TOKENS=2048 (for the
  offset-sweep candidate). Cost estimate from the master doc: ~2.13 EFLOP
  corpus-wide, ~5-6h GPU / 8-12h wall clock for both scorers -- see
  PROJECT_STATE_AND_HPC_MIGRATION.md section 3.

WHAT THIS DOES NOT FIX
  Family A for mistral-GENERATED documents (Fault 2 has no Family A fix;
  X-15 in the ledger, still undecided -- self-scoring Family A would change
  the paper's claim from reproduction to independent replication, a
  an open decision, not made here). Mistral's true checkpoint
  identity (Fault 1) -- an open provenance question, orthogonal to this
  script, tracked separately by tokenizer_checkpoint_screen.py.

REUSE, NOT RESTATEMENT
  This file imports several scorer-generic, already-tested primitives
  directly from build_p1_feature_store_gpu.py (score_batch,
  compute_align_window, family_b_feature_summary, embedding_dispersion,
  is_memory_error, empty_device_cache, load_models, MODEL_IDS,
  EMBEDDING_MODEL_ID, WARMUP_DROP/CLIP_TO/MIN_TOKENS/MAX_CANDIDATE_TOKENS/
  BATCH_SIZE_INITIAL/RANK_BUCKETS/SCORER_KEYS) rather than reimplementing
  them, so the two scripts can't silently drift on shared math -- same
  pattern the alignment-gated script itself uses for compute_boundary.
  D-04's concern (don't edit a script whose checkpointed run is in
  progress) does not block this: build_p1_feature_store_gpu.py's own run
  is already complete (gpu_output/p1_gpu_manifest.json: 720/720,
  complete=true) and this file makes ZERO edits to it -- read-only import.
  All NEW behavior (self-scoring loop, output paths, checkpoint file,
  manifest) lives entirely in this file.

CHECKPOINTING
  Independent of the alignment-gated run's checkpoint/output -- separate
  output file, separate shard-checkpoint file, separate manifest, so this
  run can proceed without touching or being blocked by the completed
  audit-evidence pass in gpu_output/.

USAGE
    python -m fractalsig.pipeline.build_p1_feature_store_gpu_selfscored --verify-span 40   # DO THIS FIRST
    python -m fractalsig.pipeline.build_p1_feature_store_gpu_selfscored                    # full run (resumable)
"""
from fractalsig import paths as _paths  # noqa: E402
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

from fractalsig.pipeline.build_p0_local import WARMUP_DROP, CLIP_TO, MIN_TOKENS
from fractalsig.pipeline.recompute_family_a_continuation_fix import compute_boundary
from fractalsig.pipeline.build_p1_feature_store_gpu import (
    MANIFEST_DIR, MODEL_IDS, EMBEDDING_MODEL_ID, SCORER_KEYS,
    BATCH_SIZE_INITIAL, RANK_BUCKETS, MAX_CANDIDATE_TOKENS,
    load_models, is_memory_error, empty_device_cache,
    compute_align_window, score_batch, family_b_feature_summary,
    embedding_dispersion, load_prefix_map,
)

DATA_DIR = os.path.join(_paths.WORKSPACE_STR, "data")
OUT_PATH = os.path.join(DATA_DIR, "p1_feature_store_gpu_selfscored.jsonl")
SHARD_CKPT_PATH = os.path.join(DATA_DIR, "p1_gpu_shard_checkpoint_selfscored.txt")
MANIFEST_PATH = os.path.join(DATA_DIR, "p1_gpu_manifest_selfscored.json")


def _ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)


def load_line_set(path):
    s = set()
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    s.add(line)
    return s


def append_line(path, value):
    with open(path, "a") as f:
        f.write(str(value) + "\n")
        f.flush()
        os.fsync(f.fileno())


def score_documents_selfscored(torch_mod, model, tok, device, rows, text_ids_by_idx):
    """Batched, ONE scorer, ONE window per document (no candidates, no
    sweep). usable = docs whose own tokenization clears MIN_TOKENS via
    compute_align_window -- the same honest-drop rule used everywhere else
    in this pipeline, not a new gate invented here."""
    usable = []
    for idx, ids in text_ids_by_idx.items():
        window = compute_align_window(len(ids))
        if window is None:
            continue
        usable.append((idx, ids[:window[1]], window))

    results = {}
    pos = 0
    batch_size = BATCH_SIZE_INITIAL
    while pos < len(usable):
        batch = usable[pos:pos + batch_size]
        batch_ids = [b[1] for b in batch]
        batch_windows = [b[2] for b in batch]
        try:
            scored_batch = score_batch(torch_mod, model, tok, device, batch_ids, batch_windows)
        except RuntimeError as e:
            if is_memory_error(e) and batch_size > 1:
                batch_size = max(1, batch_size // 2)
                empty_device_cache(torch_mod, device)
                continue
            raise
        for (idx, ids, window), scored in zip(batch, scored_batch):
            n_tokens = len(text_ids_by_idx[idx])
            results[idx] = {
                "scored": scored,
                "window": window,
                "used_short_window": n_tokens < CLIP_TO,
            }
        pos += len(batch)
    return results


def verify_span(n_docs, scorers=("gemma-2b", "mistral-7b")):
    """Diagnostic-only, exits without writing feature-store output.

    Two things checked:
    (1) coverage -- what fraction of a stratified sample clears MIN_TOKENS
        and gets a self-scored window, per prompt_type. Expected: ~100%
        minus the same small MIN_TOKENS-drop rate seen everywhere else in
        this pipeline (~3%), for EVERY prompt_type and BOTH scorers -- the
        whole point of Option B is that this number should not vary by
        prompt_type or by which model generated the document, unlike the
        alignment-gated path's pass rate.
    (2) span closeness for continue_it/continue_pt, gemma-2b -- the one
        combination where Family A's corrected window is not already
        exactly "text-alone, WARMUP_DROP:CLIP_TO". Measures, per document,
        the token offset between this script's self-scored window start
        (fixed at WARMUP_DROP) and Family A's corrected window re-expressed
        in text-alone token coordinates (compute_boundary's `boundary`
        approximates prefix_len, so text_index = (boundary+WARMUP_DROP) -
        prefix_len + 1 -- see compute_boundary's docstring for why the +1).
        Reports the delta distribution -- NOT asserted as zero, measured.
    """
    torch_mod, device, dtype, tokenizers, models, _ = load_models(scorers=scorers)
    shard_files = sorted(glob.glob(os.path.join(MANIFEST_DIR, "*.parquet")))
    if not shard_files:
        print(f"[abort] no shards found in {MANIFEST_DIR}")
        sys.exit(1)
    prefix_map = load_prefix_map()  # verify-span only -- the main run below never loads this

    rows_per_shard = max(1, -(-n_docs // len(shard_files)))
    sample_rows = []
    for shard in shard_files:
        table = pq.read_table(shard)
        for row in table.to_pylist()[:rows_per_shard]:
            row["prefix"] = prefix_map.get(row["id"])
            sample_rows.append(row)
            if len(sample_rows) >= n_docs:
                break
        if len(sample_rows) >= n_docs:
            break
    print(f"[info] checking {len(sample_rows)} documents, stratified across shards", flush=True)

    coverage = collections.Counter()
    total_by_type = collections.Counter()
    span_deltas = []

    for sc, sliced_key, total_key in [k for k in SCORER_KEYS if k[0] in scorers]:
        tok, model = tokenizers[sc], models[sc]
        text_ids_by_idx = {i: tok(r["text"])["input_ids"] for i, r in enumerate(sample_rows)}
        results = score_documents_selfscored(torch_mod, model, tok, device, sample_rows, text_ids_by_idx)
        print(f"\n=== {sc} ===", flush=True)
        for idx, row in enumerate(sample_rows):
            total_by_type[(sc, row["prompt_type"])] += 1
            r = results.get(idx)
            if r is None:
                print(f"  id={row['id']} prompt_type={row['prompt_type']}: below MIN_TOKENS, dropped", flush=True)
                continue
            coverage[(sc, row["prompt_type"])] += 1
            print(f"  id={row['id']} prompt_type={row['prompt_type']}: window={r['window']} "
                  f"short_window={r['used_short_window']}", flush=True)

            if sc == "gemma-2b" and row["prompt_type"] in ("continue_it", "continue_pt") and row.get("prefix"):
                prefix_len = len(tok(row["prefix"])["input_ids"])
                text_len = len(text_ids_by_idx[idx])
                gagle_total = row.get(total_key)
                if gagle_total:
                    boundary = compute_boundary(prefix_len, text_len, gagle_total)
                    family_a_text_start = (boundary + WARMUP_DROP) - prefix_len + 1
                    self_scored_start = r["window"][0]
                    delta = self_scored_start - family_a_text_start
                    span_deltas.append(delta)
                    print(f"       [span check] family_a_text_start={family_a_text_start} "
                          f"self_scored_start={self_scored_start} delta={delta}", flush=True)

    print("\n=== COVERAGE SUMMARY ===", flush=True)
    for (sc, pt), n in sorted(total_by_type.items()):
        got = coverage[(sc, pt)]
        print(f"  {sc:10s} {pt:20s} {got}/{n} scored", flush=True)

    if span_deltas:
        arr = np.array(span_deltas)
        print(f"\n=== SPAN CHECK (continue_it/continue_pt, gemma-2b) ===", flush=True)
        print(f"  n={len(arr)} mean_delta={arr.mean():.2f} median_delta={np.median(arr):.2f} "
              f"std={arr.std():.2f} max_abs={np.abs(arr).max():.0f}", flush=True)
        print("  delta = self_scored_start - family_a_text_start (tokens). "
              "0 = exact match. Small nonzero = known seam approximation, not a bug -- "
              "see module docstring.", flush=True)
    else:
        print("\n[warn] no continue_it/continue_pt gemma-2b rows with a usable prefix/total "
              "in this sample -- span check did not run. Not fatal, but re-run with a larger "
              "--verify-span N if this combination matters for your decision.", flush=True)

    total_checked = sum(total_by_type.values())
    total_scored = sum(coverage.values())
    frac = (total_scored / total_checked) if total_checked else 0.0
    print(f"\n[info] overall coverage this sample: {total_scored}/{total_checked} ({frac:.1%})", flush=True)
    if frac < 0.5:
        print("[FAIL] coverage under 50% -- something is badly wrong (tokenizer/model load, "
              "or compute_align_window returning None far more than the ~3% MIN_TOKENS-drop "
              "baseline this pipeline sees everywhere else). Do NOT run the full pass.", flush=True)
        sys.exit(1)
    print("[PASS] coverage sane. Safe to proceed to a full run.", flush=True)


def main(limit_shards=None, scorers=("gemma-2b", "mistral-7b")):
    _ensure_dirs()
    torch_mod, device, dtype, tokenizers, models, embedder = load_models(scorers=scorers)
    done_shards = load_line_set(SHARD_CKPT_PATH)
    shard_files = sorted(glob.glob(os.path.join(MANIFEST_DIR, "*.parquet")))
    remaining = [s for s in shard_files if s not in done_shards]
    if limit_shards:
        remaining = remaining[:limit_shards]

    stats = collections.Counter()
    with open(OUT_PATH, "a") as fout:
        for i, shard in enumerate(remaining, 1):
            table = pq.read_table(shard)
            rows = table.to_pylist()
            doc_results = [
                {"id": r["id"], "split": r["split"], "domain": r["domain"],
                 "Label": r["Label"], "prompt_type": r["prompt_type"], "temp": r["temp"]}
                for r in rows
            ]

            for sc, sliced_key, total_key in [k for k in SCORER_KEYS if k[0] in scorers]:
                tok, model = tokenizers[sc], models[sc]
                out_prefix = "Gemma" if sc == "gemma-2b" else "Mistral"
                text_ids_by_idx = {idx: tok(row["text"])["input_ids"] for idx, row in enumerate(rows)}
                results = score_documents_selfscored(torch_mod, model, tok, device, rows, text_ids_by_idx)

                for idx in range(len(rows)):
                    r = results.get(idx)
                    if r is None:
                        doc_results[idx][f"{out_prefix}_Feature_Window_Available"] = False
                        stats[f"{sc}_below_min_tokens"] += 1
                        continue
                    summary = family_b_feature_summary(r["scored"])
                    for k, v in summary.items():
                        doc_results[idx][f"{out_prefix}_{k}"] = v
                    doc_results[idx][f"{out_prefix}_Used_Short_Window"] = bool(r["used_short_window"])
                    doc_results[idx][f"{out_prefix}_Feature_Window_Available"] = True
                    doc_results[idx][f"{out_prefix}_SelfScored"] = True
                    stats[f"{sc}_scored"] += 1

            for idx, row in enumerate(rows):
                doc_results[idx]["Embedding_Dispersion"] = embedding_dispersion(
                    embedder, row["text"], row.get("gagle_gemma_total_tokens"), row.get("gagle_mistral_total_tokens"))
                stats["written"] += 1

            for row in doc_results:
                fout.write(json.dumps(row) + "\n")
            fout.flush()
            os.fsync(fout.fileno())
            append_line(SHARD_CKPT_PATH, shard)
            done_shards.add(shard)
            print(f"[shard complete] {i}/{len(remaining)} ({os.path.basename(shard)}) "
                  f"written={stats['written']} gemma_scored={stats['gemma-2b_scored']} "
                  f"mistral_scored={stats['mistral-7b_scored']}", flush=True)

    complete = len(done_shards) >= len(shard_files) and not limit_shards
    manifest = {
        "shards_total": len(shard_files), "shards_done": len(done_shards),
        "shards_missing": sorted(set(shard_files) - set(done_shards))[:20],
        "scorers_run_this_invocation": list(scorers),
        "docs_written_this_run": stats["written"],
        "gemma_2b_scored_this_run": stats["gemma-2b_scored"],
        "mistral_7b_scored_this_run": stats["mistral-7b_scored"],
        "gemma_2b_below_min_tokens_this_run": stats["gemma-2b_below_min_tokens"],
        "mistral_7b_below_min_tokens_this_run": stats["mistral-7b_below_min_tokens"],
        "complete": bool(complete),
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print("\n=== SUMMARY ===")
    for k, v in manifest.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verify-span", type=int, default=None, metavar="N",
                    help="Check N documents' coverage + (for continue_it/continue_pt gemma-2b) "
                         "span closeness to Family A's corrected window. Exits without writing "
                         "feature-store output. Run this before any full run.")
    p.add_argument("--limit-shards", type=int, default=None)
    p.add_argument("--scorers", nargs="+", choices=["gemma-2b", "mistral-7b"],
                    default=["gemma-2b", "mistral-7b"])
    args = p.parse_args()
    if args.verify_span:
        verify_span(args.verify_span, scorers=tuple(args.scorers))
    else:
        main(limit_shards=args.limit_shards, scorers=tuple(args.scorers))
