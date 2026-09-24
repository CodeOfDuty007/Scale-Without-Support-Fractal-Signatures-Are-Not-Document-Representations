#!/usr/bin/env python3
"""
P1 Feature Store -- GPU pass (Family B). Runs on a separate, higher-memory
GPU machine, consuming the CPU pass's manifest (build_p1_feature_store_cpu.py)
rather than re-reading gagle.zip or re-deriving any windowing decision --
that's the whole point of the CPU/GPU split: one read of the corpus, not two.

INPUT: parquet shards at MANIFEST_DIR (one per original zip member, written by
the CPU pass), columns: id, split, domain, Label, prompt_type, temp, side,
text, gagle_gemma_sliced_scores, gagle_mistral_sliced_scores,
gagle_gemma_total_tokens, gagle_mistral_total_tokens -- PLUS `prefix`, joined
in from data/p1_gpu_shard_prefix_backfill.jsonl by id (the 720 shards were
written before extract_docs carried `prefix`; see
fractalsig/pipeline/backfill_gpu_shard_prefix.py). Run that backfill first if the join file
doesn't exist yet.

*** RUN --verify-alignment FIRST, ON A SMALL SAMPLE, BEFORE ANY FULL RUN ***

This is the single most important operational instruction in this file, not
an optional nicety. The CPU pass shipped `text` (raw strings) rather than
token ids, because token ids depend on the exact tokenizer/checkpoint GAGLE's
own authors used to produce `gagle_*_sliced_scores` in the first place --
and that checkpoint identity is NOT written down anywhere retrievable from
this repo (checked: only "gemma-2b"/"mistral-7b" as dict keys, no HF repo id
given anywhere in GAGLE's own dataset card or this project's docs). The
GEMMA_MODEL_ID/MISTRAL_MODEL_ID constants below are best-guess HF repo ids.
If they're wrong, or even a different revision/tokenizer-version of the
right repo, this script will silently retokenize `text` into a DIFFERENT
token stream than GAGLE used, and every downstream alignment check will
fail for the right reason but be misdiagnosed as a prompt_type-alignment
problem instead of a checkpoint problem.

TWO ALIGNMENT CANDIDATES, PER DOCUMENT, PER SCORER
  the internal verification notes established that
  GAGLE's own stored scores follow ONE OF TWO conventions depending on
  prompt_type (never assumed, confirmed per-type at n=500): the human side
  and most LLM prompt_types are scored on `text` ALONE; `continue_it`/
  `continue_pt` are scored on `prefix_ids + text_ids[1:]` (prefix and text
  concatenated, one BOS dropped). Rather than hardcode that mapping (which
  would silently misclassify any prompt_type this project hasn't audited
  yet), this pass tries BOTH candidates for every document and keeps
  whichever wins -- the empirical outcome per prompt_type should reproduce
  the probe's findings, not assume them.

  For each candidate, alignment is checked with a PER-DOCUMENT offset
  sweep (not a single global GAGLE_SCORE_INDEX_OFFSET): the probe found
  real per-document offsets ranging from near-0 (continue_it/continue_pt,
  keywords/summary majority) to several hundred tokens (cot's reasoning
  preamble, correlated with document length at r=0.996). A fixed global
  offset cannot capture that; the sweep is centered on this document's own
  length-implied delta (len(candidate_ids) - gagle_total_tokens) with a
  OFFSET_SWEEP_RADIUS margin on each side, mirroring the probe's own
  per-document offset-sweep methodology (Part A, section 2) rather than
  inventing a new one.

  Whichever (candidate, offset) pair maximizes Pearson r against
  gagle_*_sliced_scores is kept. Family B features are written ONLY if that
  best r clears ALIGNMENT_PEARSON_MIN (0.98); otherwise the row gets
  Alignment_Passed=False and null Family B columns -- an honest gap, not a
  guess. Expected overall pass rate is well under 100% (the probe found
  excerpt/short_keywords/summary_keywords essentially never align under
  either candidate, and cot mostly doesn't clear 0.98 even with the
  per-document offset sweep) -- this is documented behavior, not a bug to
  chase away by loosening the gate.

TWO DIFFERENT WINDOWS, DELIBERATELY NOT THE SAME SPAN
  - ALIGNMENT window: whatever slice_scores originally produced
    (WARMUP_DROP:CLIP_TO of the reconstructed candidate's own token length,
    unmodified) -- this is a provenance check against gagle_*_sliced_scores
    AS STORED, so it must match what actually produced that stored array,
    contamination included.
  - FEATURE EXTRACTION window: for the winning candidate, if it's
    `prefix_text`, the CORRECTED per-record boundary from
    the internal verification notes
    (`compute_boundary`, imported from
    recompute_family_a_continuation_fix.py -- not restated here, so the two
    scripts can't silently drift apart) plus WARMUP_DROP. For `text_alone`
    it's the same plain WARMUP_DROP:CLIP_TO as the alignment window, since
    there's no prefix to skip. This keeps Family B's LogPPL/GLTR/entropy
    features computed over the SAME corrected span Family A now uses for
    continue_it/continue_pt, rather than reintroducing the prefix-
    contamination bug Family A was just fixed for -- and rather than
    reintroducing it in a LABEL-CORRELATED way, since only LLM-side
    continue_it/continue_pt rows have a real scored prefix at all.
  Both windows are extracted from ONE forward pass per candidate (feed
  ids[:max(align_end, feat_end)] once, slice both ranges out of the same
  logits) -- no doubled GPU cost for keeping the two windows separate.

FAMILY B FEATURES (per document, per scorer where alignment succeeds):
  - mean/var/skew/kurtosis of log-PPL (from this pass's own re-scored
    per-token log-probs, over the FEATURE EXTRACTION window above)
  - NOTE: cross-perplexity (Gemma vs. Mistral log-PPL ratio, a documented
    SIMPLIFICATION of Binoculars-style normalization -- the two scorers
    have different tokenizers, so literal cross-scoring per Hans et al.
    2024 isn't well-defined here) is a CROSS-SCORER feature and is computed
    in the join script (build_p1_join_feature_store.py), not here -- this
    script only ever has one scorer's rows in scope at a time per batch.
  - GLTR-style rank buckets: true rank of the actual next token in the
    FULL softmax distribution at each position (from captured logits, not
    approximated from the log-prob alone), bucketed into top-10/100/1000
    fractions across the document.
  - exact next-token entropy (from the same logits), doc-mean.
  - guardrail cross-check columns themselves (winning candidate, best
    offset, Pearson r, mean abs diff vs. GAGLE's stored scores) are
    WRITTEN INTO the output per document, not just printed during
    verification -- so a later, cheap pass can filter on Alignment_Passed
    without re-running the GPU pass.

EMBEDDING DISPERSION (Family C's deferred sub-feature)
  Computed HERE, in the same per-shard loop, not as a third pass -- see
  build_p1_feature_store_cpu.py's docstring for why (no torch on the CPU
  machine). Reuses the EXACT SAME word-budget/truncation logic as the CPU
  pass's Family C (imported, not reimplemented, to avoid the two families
  silently operating on different spans of the same document). Sentence
  embeddings via a small sentence-transformer model; dispersion = mean
  pairwise cosine distance from the document's own centroid embedding.
  Computed on `text` alone regardless of which candidate wins Family B's
  alignment -- Family C is a text-only family and was never contaminated
  by the prefix-conditioning issue.

CHECKPOINTING
  Same per-shard-file granularity as the CPU pass: data/p1_gpu_shard_checkpoint.txt
  records one line per fully-processed manifest shard file. A crash mid-shard
  redoes that whole shard (a few thousand docs, GPU-bound, cheap to redo
  relative to losing checkpoint granularity finer than that).

USAGE
    python -m fractalsig.pipeline.backfill_gpu_shard_prefix       # one-time, if not already run
    python -m fractalsig.pipeline.build_p1_feature_store_gpu --verify-alignment 20   # DO THIS FIRST
    python -m fractalsig.pipeline.build_p1_feature_store_gpu                          # full run (resumable)
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
from scipy.stats import skew, kurtosis

from fractalsig.pipeline.build_p0_local import WARMUP_DROP, CLIP_TO, MIN_TOKENS
from fractalsig.pipeline.build_p1_feature_store_cpu import (
    compute_word_budget, truncate_preserving_structure, _SENT_SPLIT_RE, _ZERO_WIDTH_SPACE_RE,
)
from fractalsig.pipeline.recompute_family_a_continuation_fix import compute_boundary

# Overridable via env var -- the physical SSD this path points at gets
# mounted at a different path on every machine; resolved via fractalsig.paths
# (override with $GAGLE_MANIFEST_DIR). Default is the workspace
# point, unchanged for local runs.
MANIFEST_DIR = os.environ.get("GAGLE_MANIFEST_DIR", _paths.MANIFEST_SHARD_DIR_STR)

# BEST-GUESS HF repo ids -- see the module docstring's "*** RUN
# --verify-alignment FIRST ***" section. Not confirmed against GAGLE's own
# scoring pipeline; confirm empirically before trusting any output below.
GEMMA_MODEL_ID = "google/gemma-2b"
MISTRAL_MODEL_ID = "mistralai/Mistral-7B-v0.1"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

DATA_DIR = os.path.join(_paths.WORKSPACE_STR, "data")
OUT_PATH = os.path.join(DATA_DIR, "p1_feature_store_gpu.jsonl")
SHARD_CKPT_PATH = os.path.join(DATA_DIR, "p1_gpu_shard_checkpoint.txt")
MANIFEST_PATH = os.path.join(DATA_DIR, "p1_gpu_manifest.json")
PREFIX_BACKFILL_PATH = os.path.join(DATA_DIR, "p1_gpu_shard_prefix_backfill.jsonl")

BATCH_SIZE_INITIAL = 16
RANK_BUCKETS = [10, 100, 1000]
ALIGNMENT_PEARSON_MIN = 0.98
ALIGNMENT_MEAN_ABS_DIFF_MAX = 0.10
MAX_CANDIDATE_TOKENS = 2048  # fallback ONLY, for callers that do not supply a
                             # per-row gagle_total. Not a resource limit -- see
                             # tokenize_candidates for why this is a
                             # reconstruction parameter.
MAX_CANDIDATE_TOKENS_HARD = 4096  # OOM guard. Must exceed the largest
                                  # gagle_*_total_tokens in the corpus
                                  # (GAGLE's own caps: gemma 2047,
                                  # mistral 3071).
CAP_MARGIN = 1  # `m` in the derivation; anchor lands at ~64 + m and the true
                # offset is 65, so this must stay well under
                # OFFSET_SWEEP_RADIUS.

CANDIDATE_NAMES = ["text_alone", "prefix_text"]

# compute_boundary (imported above) was derived from Gemma-2b's own
# boundary-crossover diagnostic and does NOT transfer to Mistral-7b -- Test
# 3A (the internal verification notes, Part A)
# forward-passed ~10 Mistral continue_it records and found a median
# combined_delta of 35 tokens (vs Gemma's ~1), only 0.4% landing within +/-3
# tokens of Gemma's formula. Deriving Mistral's own formula is Task 2 of
# the internal verification notes, confirmed blocked
# on hardware (17.2GB RAM, insufficient for a 7B forward pass) -- not
# attempted. Until it exists, Mistral `prefix_text` rows get NO corrected
# feature window (honest drop -- see compute_doc_score_range), rather than a
# feature window built on a formula already shown not to fit this scorer.
BOUNDARY_CORRECTED_SCORERS = {"gemma-2b"}

# Per-DOCUMENT sweep margin around this document's own length-implied delta
# (len(candidate_ids) - gagle_total_tokens), NOT a single global offset --
# see the module docstring's "TWO ALIGNMENT CANDIDATES" section. 150 covers
# the probe's observed range (continue_it/pt and keywords/summary majority
# near 0; cot up to several hundred, correlated with length at r=0.996;
# excerpt/short_keywords/summary_keywords don't align at ANY offset, so no
# radius fixes those -- confirmed in the probe, not re-litigated here).
OFFSET_SWEEP_RADIUS = 150


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


MODEL_IDS = {"gemma-2b": GEMMA_MODEL_ID, "mistral-7b": MISTRAL_MODEL_ID}

# (scorer_name, gagle_sliced_column, gagle_total_tokens_column) -- shared by
# verify_alignment and main so the two loops can't silently diverge.
SCORER_KEYS = [
    ("gemma-2b", "gagle_gemma_sliced_scores", "gagle_gemma_total_tokens"),
    ("mistral-7b", "gagle_mistral_sliced_scores", "gagle_mistral_total_tokens"),
]


def load_models(scorers=("gemma-2b", "mistral-7b")):
    """Deferred import -- this module must remain importable (for the CPU
    machine to run --help / read the docstring) even where torch isn't
    installed. Actually loading models only happens when a run starts.

    `scorers` restricts which model(s) get loaded -- mistral-7b is ~28GB in
    fp32 (this module's dtype on non-CUDA devices; see the dtype line below)
    and simply does not fit on a 16GB machine. On CUDA the full pair is the
    default and unchanged from before this flag existed."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizers = {sc: AutoTokenizer.from_pretrained(MODEL_IDS[sc]) for sc in scorers}
    models = {sc: AutoModelForCausalLM.from_pretrained(MODEL_IDS[sc], torch_dtype=dtype).to(device).eval()
              for sc in scorers}
    embedder = SentenceTransformer(EMBEDDING_MODEL_ID, device=device)
    return torch, device, dtype, tokenizers, models, embedder


def empty_device_cache(torch_mod, device):
    """torch_mod.cuda.empty_cache() raises AttributeError on MPS (no such
    method) and on CPU, turning a recoverable OOM-retry into a crash mid-
    shard -- caught during code review before ever running on real
    hardware, since only CUDA's cache-clear was ever called unconditionally."""
    if device == "cuda":
        torch_mod.cuda.empty_cache()
    elif device == "mps":
        torch_mod.mps.empty_cache()


# Substrings covering how different backends phrase an allocation failure.
# CUDA raises "CUDA out of memory"; MPS raises "Invalid buffer size: X GiB"
# (confirmed directly -- a stratified N=40 --verify-alignment run on this
# Mac hit exactly this string). The retry logic in
# score_documents_for_candidate originally matched "out of memory" only, so
# on MPS the exception propagated uncaught and crashed the whole run instead
# of backing off to a smaller batch -- caught by that crash, not guessed.
#
# A third form confirmed 2026-08-22 on a MIG-partitioned CUDA
# (RTX PRO 6000 Blackwell, 24GB slice): a real VRAM exhaustion (both
# gemma-2b + mistral-7b resident at once, ~19GB weights alone leaving no
# headroom) raised "NVML_SUCCESS == r INTERNAL ASSERT FAILED ...
# CUDACachingAllocator.cpp", NOT "CUDA out of memory" -- E-17's predicted gap
# ("is_memory_error() has only ever been triggered by MPS's string ... do
# NOT guess new strings mid-run") materializing exactly as flagged, now with
# real evidence rather than a guess.
MEMORY_ERROR_SUBSTRINGS = ("out of memory", "invalid buffer size", "cudacachingallocator")


def is_memory_error(exc):
    msg = str(exc).lower()
    return any(s in msg for s in MEMORY_ERROR_SUBSTRINGS)


def load_prefix_map():
    """id -> prefix, from the one-time backfill (see module docstring and
    fractalsig/pipeline/backfill_gpu_shard_prefix.py). The 720 shards on disk predate
    extract_docs carrying `prefix`, so it's joined in here rather than
    re-emitting all 720 parquet files."""
    if not os.path.exists(PREFIX_BACKFILL_PATH):
        print(f"[abort] {PREFIX_BACKFILL_PATH} not found. Run "
              f"fractalsig/pipeline/backfill_gpu_shard_prefix.py first.")
        sys.exit(1)
    m = {}
    with open(PREFIX_BACKFILL_PATH) as f:
        for line in f:
            r = json.loads(line)
            m[r["id"]] = r.get("prefix")
    return m


def compute_align_window(n_tokens):
    """(start, end): the range of token INDICES that slice_scores would have
    extracted for a candidate of this length. Mirrors
    build_p0_local.slice_scores's branching exactly (same CLIP_TO/
    WARMUP_DROP/MIN_TOKENS thresholds), UNCORRECTED -- this is a provenance
    check against gagle_*_sliced_scores AS STORED, so it must match
    whatever produced that array in the first place, prefix-contamination
    included. See compute_feature_window_prefix_text and
    score_documents_for_candidate for the corrected span actually used to
    compute Family B features.

    Deliberately does NOT determine what to FEED the model -- score_batch
    feeds the model the full token[:end] prefix regardless of `start`, so
    every extracted position has its real preceding context available. An
    earlier version of this function sliced the token ids to [start:end)
    BEFORE feeding them to the model, which silently discarded the 64 tokens
    of genuine prior context GAGLE's own scores had available for the
    long-doc case -- early-window log-probs would then be systematically
    worse than GAGLE's for a reason having nothing to do with tokenizer
    mismatch, and --verify-alignment would misdiagnose it as a wrong
    checkpoint. Caught before ever running this against a real model."""
    if not n_tokens:
        return None
    if n_tokens >= CLIP_TO:
        start, end = WARMUP_DROP, CLIP_TO
    else:
        start, end = 0, n_tokens
    if (end - start) < MIN_TOKENS:
        return None
    return start, end


def compute_feature_window_prefix_text(n_tokens, prefix_len, text_len, gagle_total):
    """(start, end) for FEATURE EXTRACTION on the `prefix_text` candidate --
    the CORRECTED span, so Family B's LogPPL/GLTR/entropy features are
    computed over the same tokens Family A now uses for continue_it/
    continue_pt (see the internal verification notes),
    NOT the prefix-contaminated span GAGLE's own stored array covers
    (which compute_align_window mirrors, deliberately, for the alignment
    check only -- see the module docstring's "TWO DIFFERENT WINDOWS"
    section for why these must stay separate).

    Uses compute_boundary (the SAME per-record formula
    recompute_family_a_continuation_fix.py uses for Family A, imported not
    restated) plus WARMUP_DROP, matching that script's `extra_warmup`
    variant (selected 2026-08-05). Returns None on the
    same drop conditions that script uses (negative_boundary,
    start_past_end, below_min_tokens) -- an honest drop, not a silently
    shortened window.

    text_alone has NO analog of this function -- see
    score_documents_for_candidate, which derives its feature window from
    the ALIGNMENT result (best_abs_start) instead. text_alone's own
    gagle-stored slice is not prefix-contaminated (nothing to skip), but it
    can still start at a real, document-specific offset (cot's reasoning
    preamble, up to several hundred tokens; keywords/summary's smaller
    residuals, present in over half of documents per the probe). A flat
    WARMUP_DROP:CLIP_TO window would silently ignore that offset and
    extract the wrong span even on documents where alignment correctly
    verified a different span -- caught during self-review before this
    shipped, not after."""
    if prefix_len is None or text_len is None or gagle_total is None:
        return None
    boundary = compute_boundary(prefix_len, text_len, gagle_total)
    if boundary < 0:
        return None
    start = boundary + WARMUP_DROP
    if start >= n_tokens:
        return None
    end = min(start + (CLIP_TO - WARMUP_DROP), n_tokens)
    if (end - start) < MIN_TOKENS:
        return None
    return start, end


def tokenize_candidates(tok, text, prefix, gagle_total=None):
    """Both alignment candidates for one document (see module docstring's
    "TWO ALIGNMENT CANDIDATES" section). `prefix_text` is only offered when
    a real prefix string is present -- every row in this corpus carries
    SOME prefix field (confirmed by fractalsig/pipeline/backfill_gpu_shard_prefix.py: 0/
    237,927 rows had a missing/empty one), so this is really just a
    defensive guard, not an expected branch."""
    text_ids = tok(text)["input_ids"]
    text_len = len(text_ids)
    # Per-row, per-scorer cap. `gagle_total` is this scorer's
    # gagle_<scorer>_total_tokens for THIS row. Truncating to `cap` sets
    # delta = n - gagle_total, and sweep_best_alignment anchors at
    # align_window.start + delta (~64 + delta) with radius
    # OFFSET_SWEEP_RADIUS -- so cap = gagle_total + CAP_MARGIN puts the anchor
    # at ~65, which is the empirically observed winning offset. A flat cap
    # cannot serve both scorers: GAGLE capped gemma human arrays at 2047 and
    # mistral at 3071. None -> old constant, preserving legacy callers.
    #
    # RAISE-ONLY (D-09). The outer max() is load-bearing: without it the cap
    # DROPS below 2048 for every document whose gagle_total is small, which is
    # most of them -- measured 2026-08-23 on gemma_2b_0.0 shards, 12/12 rows in
    # each of the summary, keywords and continue_pt cells have gagle_total
    # between 381 and 1148. Truncating those candidates shorter than the
    # unpatched code did is a pure regression: it cannot fix a drop (drops come
    # from the cap being too LOW for mistral's 3071-token human arrays) and it
    # breaks alignment on rows that were passing at r=1.0000.
    #
    # This is exactly the v1-vs-v2 distinction in F-B. Job 85 (2026-08-23) ran
    # the v1 form by mistake and caught it: gemma_llm_summary MUST_PASS fell
    # 12->9 while mistral_human dropped 12->0, i.e. the intended fix worked and
    # the missing max() regressed a different cell in the same run.
    if gagle_total:
        cap = max(MAX_CANDIDATE_TOKENS,
                  min(MAX_CANDIDATE_TOKENS_HARD, int(gagle_total) + CAP_MARGIN))
    else:
        cap = MAX_CANDIDATE_TOKENS
    candidates = {
        "text_alone": {"ids": text_ids[:cap], "prefix_len": None, "text_len": text_len},
    }
    if prefix:
        prefix_ids = tok(prefix)["input_ids"]
        combined = (prefix_ids + text_ids[1:])[:cap]
        candidates["prefix_text"] = {"ids": combined, "prefix_len": len(prefix_ids), "text_len": text_len}
    return candidates


def compute_doc_score_range(cand_ids, candidate_name, gagle_total, gagle_sliced, prefix_len, text_len, scorer_name):
    """The union token-index range [lo, hi) to extract predictions for, in
    ONE forward pass, covering both (a) the per-document alignment offset
    sweep (compute_align_window's start, shifted by this candidate's own
    length-implied delta, +/- OFFSET_SWEEP_RADIUS) and (b), for
    `prefix_text` only, the corrected feature-extraction window (see
    compute_feature_window_prefix_text). `text_alone`'s feature window is
    NOT precomputed here -- it's derived post-hoc from the alignment
    sweep's own result (see score_documents_for_candidate); (a) already
    covers it since best_abs_start is chosen from within the sweep range.

    `scorer_name` gates whether the corrected feature window is even
    attempted -- see BOUNDARY_CORRECTED_SCORERS.

    Returns None if unusable (mirrors slice_scores' MIN_TOKENS-style
    honest-drop philosophy, not silent padding)."""
    n = len(cand_ids)
    if gagle_sliced is None or gagle_total is None:
        return None
    align_window = compute_align_window(n)
    n_gagle = len(gagle_sliced)
    if align_window is None or n_gagle == 0:
        return None
    delta = n - gagle_total
    a_start, _ = align_window
    anchor = a_start + delta
    lo = max(1, anchor - OFFSET_SWEEP_RADIUS)
    hi = min(n, anchor + n_gagle + OFFSET_SWEEP_RADIUS)
    if hi - lo < n_gagle:
        return None

    feat_window = None
    if candidate_name == "prefix_text" and scorer_name in BOUNDARY_CORRECTED_SCORERS:
        feat_window = compute_feature_window_prefix_text(n, prefix_len, text_len, gagle_total)
        if feat_window is not None:
            lo = min(lo, feat_window[0])
            hi = max(hi, feat_window[1])
    return {"lo": lo, "hi": hi, "align_window": align_window, "feat_window": feat_window,
            "delta": delta, "n_gagle": n_gagle}


def score_batch(torch_mod, model, tokenizer, device, token_id_lists, windows):
    """Teacher-forced forward pass over the FULL token[:end] prefix for each
    doc (real preceding context throughout -- see compute_align_window),
    extracting predictions only for positions in [start, end).

    token_id_lists[i] is the full tokenization (untruncated at the front);
    windows[i] = (start, end), the union score-range from compute_doc_score_range. Left-padded to the
    batch max length so every sequence's real content ends flush with the
    right edge -- extraction offsets shift by the same pad amount for every
    position, including `start`.
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    fed_lists = [ids[:end] for ids, (start, end) in zip(token_id_lists, windows)]
    max_len = max(len(t) for t in fed_lists)
    input_ids = torch_mod.full((len(fed_lists), max_len), pad_id, dtype=torch_mod.long)
    attn_mask = torch_mod.zeros((len(fed_lists), max_len), dtype=torch_mod.long)
    for i, ids in enumerate(fed_lists):
        input_ids[i, max_len - len(ids):] = torch_mod.tensor(ids, dtype=torch_mod.long)
        attn_mask[i, max_len - len(ids):] = 1
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)

    # Request logits for only the positions this batch will actually
    # extract, not the whole padded sequence. HF's forward() defaults
    # logits_to_keep=0, which projects EVERY position through lm_head
    # (hidden_size -> vocab_size=256k) -- with max_len up to
    # MAX_CANDIDATE_TOKENS=2048 and a real batch, that's tens of GB for a
    # window we only need ~700-1000 tokens of. Because every sequence is
    # LEFT-padded (real content right-aligned to the same edge, see the
    # padding loop above), one shared `logits_to_keep=K` (keep the last K
    # positions) covers every document's own [start,end) window with a
    # single scalar, regardless of each document's individual length --
    # not an approximation, an exact cover, derived below. Caught via a
    # RuntimeError("Invalid buffer size: 10.80 GiB") on a stratified N=40
    # --verify-alignment run, before this ever reached the GPU machine.
    min_logit_pos = None
    for ids, (start, end) in zip(fed_lists, windows):
        t_start = max(start, 1)
        if t_start >= end:
            continue
        pad_offset = max_len - len(ids)
        pos = pad_offset + t_start - 1
        if min_logit_pos is None or pos < min_logit_pos:
            min_logit_pos = pos
    # +2 margin: covers the boundary position exactly needed (derivation:
    # every doc's LAST needed logit position is max_len-2, a fixed point
    # independent of that doc's own length, since fed_lists[i] has length
    # end_i and is right-aligned) plus one extra position of slack rather
    # than relying on that derivation being exactly tight.
    keep = max_len if min_logit_pos is None else max(1, min(max_len, max_len - min_logit_pos + 2))

    with torch_mod.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask, logits_to_keep=keep)
        logits = out.logits  # (batch, keep, vocab) -- positions [max_len-keep, max_len)
        log_probs = torch_mod.log_softmax(logits.float(), dim=-1)
    kept_offset = max_len - keep

    # Vectorized extraction, not a per-token Python loop with 3 GPU syncs
    # (.item() each) per position. OFFSET_SWEEP_RADIUS widens the scored
    # range well past a single doc's [WARMUP_DROP:CLIP_TO] window (up to
    # several hundred extra tokens for high-delta documents like cot), so a
    # per-token sync loop here would multiply GPU cost by the window-width
    # ratio across the whole corpus -- caught during self-review, fixed by
    # gathering/ranking/entropy-computing the WHOLE window per doc in one
    # batched tensor op and only converting to Python lists at the end.
    probs = log_probs.exp()
    entropy_all = -(probs * log_probs).sum(dim=-1)  # (batch, keep)
    results = []
    for i, (ids, (start, end)) in enumerate(zip(fed_lists, windows)):
        pad_offset = max_len - len(ids)
        # position p (0-indexed into `ids`) predicts token ids[p+1]; we only
        # want predictions for token indices in [start, end), i.e. logits at
        # positions (start-1)..(end-2) predicting targets start..(end-1).
        # target_idx==0 has no preceding token to predict it from within
        # `ids` -- only reachable when start==0 (short-doc branch), so clip
        # the range to start at max(start, 1).
        t_start = max(start, 1)
        if t_start >= end:
            results.append({"log_probs": [], "ranks": [], "entropies": []})
            continue
        target_positions = torch_mod.arange(t_start, end, device=input_ids.device)
        logit_positions = pad_offset + target_positions - 1 - kept_offset  # rebased into the kept window
        target_ids = input_ids[i, pad_offset + target_positions]  # (n,)
        lp_window = log_probs[i, logit_positions]  # (n, vocab)
        target_logprobs = lp_window.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)  # (n,)
        ranks = (lp_window > target_logprobs.unsqueeze(-1)).sum(dim=-1) + 1  # (n,)
        entropies = entropy_all[i, logit_positions]  # (n,)
        results.append({
            "log_probs": target_logprobs.tolist(),
            "ranks": ranks.tolist(),
            "entropies": entropies.tolist(),
        })
    return results


def sweep_best_alignment(log_probs, lo, hi, align_window, delta, gagle_sliced):
    """Per-document offset sweep -- see the module docstring's "TWO
    ALIGNMENT CANDIDATES" section. `log_probs` covers target_idx = lo..hi-1
    (len(log_probs) == hi-lo). Slides an n_gagle-length window across it,
    candidate absolute start positions centered on align_window's start
    shifted by this candidate's own length-implied delta
    (len(candidate_ids) - gagle_total), +/- OFFSET_SWEEP_RADIUS.

    This SUBSUMES the old fixed structural_offset special case (short-doc
    branch, align_window start==0) rather than handling it separately --
    the correct small offset is just one of the values naturally tried in
    the sweep, found empirically like everything else here, not asserted.

    Returns (best_abs_start, best_r, best_mean_abs_diff) or
    (None, nan, nan) if nothing fits (too few overlapping positions, or the
    gagle array degenerate)."""
    n_gagle = len(gagle_sliced) if gagle_sliced is not None else 0
    if n_gagle < 2 or len(log_probs) < n_gagle:
        return None, float("nan"), float("nan")
    a_start, _ = align_window
    anchor = a_start + delta
    lo_try = max(lo, anchor - OFFSET_SWEEP_RADIUS)
    hi_try = min(hi - n_gagle, anchor + OFFSET_SWEEP_RADIUS)
    if hi_try < lo_try:
        return None, float("nan"), float("nan")
    gagle_arr = np.asarray(gagle_sliced, dtype=np.float64)
    log_probs_arr = np.asarray(log_probs, dtype=np.float64)
    best_abs_start, best_r, best_diff = None, float("-inf"), float("nan")
    for abs_start in range(lo_try, hi_try + 1):
        s = abs_start - lo
        window_lp = log_probs_arr[s:s + n_gagle]
        if np.std(window_lp) == 0 or np.std(gagle_arr) == 0:
            continue
        r = float(np.corrcoef(window_lp, gagle_arr)[0, 1])
        if np.isnan(r) or r <= best_r:
            continue
        best_r = r
        best_abs_start = abs_start
        best_diff = float(np.mean(np.abs(window_lp - gagle_arr)))
    if best_abs_start is None:
        return None, float("nan"), float("nan")
    return best_abs_start, best_r, best_diff


def extract_window_arrays(scored, lo, window):
    """Slice out [start,end) from a scored dict covering [lo, lo+len), or
    None if `window` doesn't fit inside what was actually scored."""
    if window is None:
        return None
    w_start, w_end = window
    n_scored = len(scored["log_probs"])
    if w_start < lo or w_end > lo + n_scored:
        return None
    s, e = w_start - lo, w_end - lo
    return {"log_probs": scored["log_probs"][s:e], "ranks": scored["ranks"][s:e],
            "entropies": scored["entropies"][s:e]}


def family_b_feature_summary(arrays):
    """LogPPL/GLTR/entropy features over `arrays` (the FEATURE EXTRACTION
    window, already sliced out by extract_window_arrays) -- alignment
    quality itself is a SEPARATE set of columns, written by the caller from
    sweep_best_alignment's result, not folded in here."""
    log_probs = np.array(arrays["log_probs"])
    ranks = np.array(arrays["ranks"])
    entropies = np.array(arrays["entropies"])
    n_scored = len(log_probs)
    rank_fracs = {f"GLTR_top{b}_frac": float(np.mean(ranks <= b)) for b in RANK_BUCKETS}
    return {
        "LogPPL_Mean": float(-np.mean(log_probs)) if n_scored else float("nan"),
        "LogPPL_Var": float(np.var(log_probs)) if n_scored else float("nan"),
        "LogPPL_Skew": float(skew(log_probs)) if n_scored > 2 else float("nan"),
        "LogPPL_Kurtosis": float(kurtosis(log_probs)) if n_scored > 3 else float("nan"),
        "Entropy_Mean": float(np.mean(entropies)) if n_scored else float("nan"),
        **rank_fracs,
        "Feature_Window_N_Scored": n_scored,
    }


def embedding_dispersion(embedder, text, gemma_total_tokens, mistral_total_tokens):
    ref_tokens = gemma_total_tokens or mistral_total_tokens
    word_budget = compute_word_budget(text, ref_tokens)
    text_used = truncate_preserving_structure(text, word_budget)
    text_used = _ZERO_WIDTH_SPACE_RE.sub("", text_used)
    sentences = [s.strip() for s in _SENT_SPLIT_RE.split(text_used) if s.strip()]
    if len(sentences) < 2:
        return float("nan")
    embs = embedder.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
    centroid = embs.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(embs, axis=1) * np.linalg.norm(centroid)
    cos_sim = (embs @ centroid.T).flatten() / np.clip(norms, 1e-8, None)
    return float(np.mean(1.0 - cos_sim))


def score_documents_for_candidate(torch_mod, model, tok, device, rows, row_candidates,
                                   candidate_name, sliced_key, total_key, scorer_name):
    """Batched, for ONE (scorer, candidate) pair, across all `rows` in a
    shard (or a diagnostic sample). Returns {idx: result} for docs where
    this candidate was usable (mirrors slice_scores' honest-drop philosophy
    -- unusable docs are simply absent, not padded or defaulted). result =
    {best_abs_start, best_r, best_mean_abs_diff, feat_arrays, delta, n_gagle}."""
    usable = []
    for idx, row in enumerate(rows):
        cand = row_candidates[idx].get(candidate_name)
        if cand is None:
            continue
        gagle_sliced = row.get(sliced_key)
        gagle_total = row.get(total_key)
        info = compute_doc_score_range(cand["ids"], candidate_name, gagle_total, gagle_sliced,
                                        cand["prefix_len"], cand["text_len"], scorer_name)
        if info is None:
            continue
        usable.append((idx, cand["ids"], info))

    results = {}
    pos = 0
    batch_size = BATCH_SIZE_INITIAL
    while pos < len(usable):
        batch = usable[pos:pos + batch_size]
        batch_ids = [b[1] for b in batch]
        batch_windows = [(b[2]["lo"], b[2]["hi"]) for b in batch]
        try:
            scored_batch = score_batch(torch_mod, model, tok, device, batch_ids, batch_windows)
        except RuntimeError as e:
            if is_memory_error(e) and batch_size > 1:
                batch_size = max(1, batch_size // 2)
                empty_device_cache(torch_mod, device)
                continue  # retry the same `pos` at the smaller batch size
            raise
        for (idx, cand_ids, info), scored in zip(batch, scored_batch):
            gagle_sliced = rows[idx].get(sliced_key)
            best_abs_start, best_r, best_diff = sweep_best_alignment(
                scored["log_probs"], info["lo"], info["hi"], info["align_window"], info["delta"], gagle_sliced)
            if candidate_name == "prefix_text":
                feat_window = info["feat_window"]
            else:
                # text_alone: no independent correction formula (nothing to
                # skip) -- anchor the feature window at wherever alignment
                # empirically verified GAGLE's own content actually starts
                # in our reconstructed stream, not a flat WARMUP_DROP that
                # ignores a real per-document offset (cot's preamble;
                # keywords/summary's smaller residuals). Always fits inside
                # [lo,hi) by construction: best_abs_start is chosen from
                # within the sweep range, which already accounts for
                # OFFSET_SWEEP_RADIUS margin past it.
                feat_window = (best_abs_start, best_abs_start + info["n_gagle"]) if best_abs_start is not None else None
            feat_arrays = extract_window_arrays(scored, info["lo"], feat_window)
            results[idx] = {
                "best_abs_start": best_abs_start, "best_r": best_r, "best_mean_abs_diff": best_diff,
                "feat_arrays": feat_arrays, "delta": info["delta"], "n_gagle": info["n_gagle"],
            }
        pos += len(batch)
    return results


def pick_best_candidate(per_candidate):
    """per_candidate: {candidate_name: result_or_None}. Returns
    (best_name, best_result, both_passed).

    An earlier version of this function always preferred `prefix_text`
    whenever both candidates cleared the alignment gate, reasoning that on a
    continue_it/continue_pt document `text_alone` has no legitimate way to
    fit GAGLE's prefix-inclusive stored array, so a close call should be
    resolved toward the richer-context interpretation. That reasoning turns
    out not to generalize: verified directly, on real rows, on 2026-08-06,
    that for Human `continue`-type rows whose winning dedup member
    came from a continue_it/continue_pt zip file, BOTH candidates clear the
    gate, but `text_alone` is the genuine match (r=1.0000,
    mean_abs_diff~0.01, offset landing exactly at WARMUP_DROP+1) while
    `prefix_text` is a spurious fit found by the wide OFFSET_SWEEP_RADIUS
    sweep on an autocorrelated series (r=0.998 but mean_abs_diff~0.07, an
    order of magnitude worse, at a non-canonical offset). A blind
    `prefix_text` preference would have picked the wrong candidate for
    exactly this row class and then computed Family B features over the
    wrong span.

    Fixed to use `mean_abs_diff` -- the SAME metric already used to define
    "passing" -- as the actual tiebreak: among candidates that pass, keep
    whichever fits better. This isn't a heuristic bolted on top of the
    acceptance gate; it's the gate's own more discriminating half (Pearson r
    is a weak discriminator on autocorrelated surprisal series -- confirmed
    empirically above -- mean_abs_diff is not). Verified this doesn't
    regress the continue_it/continue_pt case it was originally written for:
    on genuine continue_it/continue_pt rows, `text_alone` doesn't merely
    lose the tiebreak, it fails the gate outright (measured r~0.21,
    mean_abs_diff~1.5 on real rows) -- so the "both pass" branch this
    tiebreak governs essentially never fires for the case the old comment
    worried about, and always fires as intended for the Human case it
    missed. `both_passed` is returned for auditing how often this branch
    fires either way."""
    available = {cn: r for cn, r in per_candidate.items() if r is not None}
    if not available:
        return None, None, False
    passing = {
        cn: r for cn, r in available.items()
        if not np.isnan(r["best_r"]) and r["best_r"] > ALIGNMENT_PEARSON_MIN
        and r["best_mean_abs_diff"] < ALIGNMENT_MEAN_ABS_DIFF_MAX
    }
    both_passed = len(passing) > 1
    if passing:
        name = min(passing, key=lambda cn: passing[cn]["best_mean_abs_diff"])
        return name, passing[name], both_passed
        return name, passing[name], both_passed
    best_name = max(available, key=lambda cn: available[cn]["best_r"] if not np.isnan(available[cn]["best_r"]) else -1)
    return best_name, available[best_name], False


def verify_alignment(n_docs, scorers=("gemma-2b", "mistral-7b")):
    """Diagnostic-only: checks N documents' tokenization/scoring against
    GAGLE's stored values and prints per-candidate results. Exits without
    writing feature-store output. Run this before any full run.

    `scorers` lets this run gemma-2b-only on machines that can't fit
    mistral-7b (see load_models) -- a gemma-only pass validates the
    alignment/candidate-selection/feature-window logic (it's scorer-generic
    code, exercised the same way regardless of which model runs it), but
    does NOT validate anything mistral-specific -- there is nothing
    mistral-specific to validate yet, since BOUNDARY_CORRECTED_SCORERS
    excludes it until Task 2's derivation is unblocked."""
    torch_mod, device, dtype, tokenizers, models, _ = load_models(scorers=scorers)
    shard_files = sorted(glob.glob(os.path.join(MANIFEST_DIR, "*.parquet")))
    if not shard_files:
        print(f"[abort] no shards found in {MANIFEST_DIR}")
        sys.exit(1)
    prefix_map = load_prefix_map()

    # Stratified across shards (a few rows per shard, walking every shard in
    # the manifest until n_docs is reached), NOT drained from shard_files[0]
    # alone. shard_files[0] is always the same one (sorted) prompt_type --
    # draining n_docs from it would silently defeat this function's own
    # documented purpose (comparing against the expected PER-PROMPT_TYPE
    # pattern: human/continue_it/pt/keywords/summary should pass, cot mostly
    # fails, excerpt/short_keywords/summary_keywords never pass) with a
    # single-prompt_type sample that can't exercise that pattern at all --
    # the same sampler-collapse failure mode already caught once in Test 2
    # (the internal verification notes).
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
    print(f"[info] checking {len(sample_rows)} documents, stratified across "
          f"{min(len(shard_files), -(-n_docs // rows_per_shard))} shards "
          f"({rows_per_shard}/shard)", flush=True)

    passed_counts = collections.Counter()
    candidate_win_counts = collections.Counter()
    for sc, sliced_key, total_key in [k for k in SCORER_KEYS if k[0] in scorers]:
        tok, model = tokenizers[sc], models[sc]
        row_candidates = [tokenize_candidates(tok, row["text"], row.get("prefix"), row.get(total_key)) for row in sample_rows]
        per_candidate_results = {
            cn: score_documents_for_candidate(torch_mod, model, tok, device, sample_rows, row_candidates,
                                               cn, sliced_key, total_key, sc)
            for cn in CANDIDATE_NAMES
        }
        print(f"\n=== {sc} ===", flush=True)
        for idx, row in enumerate(sample_rows):
            per_candidate = {cn: per_candidate_results[cn].get(idx) for cn in CANDIDATE_NAMES}
            best_name, best, both_passed = pick_best_candidate(per_candidate)
            if best is None:
                print(f"  id={row['id']} prompt_type={row['prompt_type']}: no candidate usable "
                      f"(too short / no fit)", flush=True)
                continue
            passed = (not np.isnan(best["best_r"])) and best["best_r"] > ALIGNMENT_PEARSON_MIN \
                and best["best_mean_abs_diff"] < ALIGNMENT_MEAN_ABS_DIFF_MAX
            passed_counts[(sc, passed)] += 1
            candidate_win_counts[(sc, row["prompt_type"], best_name)] += 1
            if both_passed:
                print(f"       [note] BOTH candidates passed for this doc -- kept {best_name} "
                      f"(lower mean_abs_diff, per pick_best_candidate).", flush=True)
            print(f"  id={row['id']} prompt_type={row['prompt_type']}: best_candidate={best_name} "
                  f"best_offset={best['best_abs_start']} r={best['best_r']:.4f} "
                  f"mean_abs_diff={best['best_mean_abs_diff']:.4f} {'PASS' if passed else 'FAIL'}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    for sc in scorers:
        n_pass = passed_counts[(sc, True)]
        n_fail = passed_counts[(sc, False)]
        total = n_pass + n_fail
        print(f"  {sc}: {n_pass}/{total} passed (r > {ALIGNMENT_PEARSON_MIN})", flush=True)
    print("\n  candidate wins by prompt_type (which candidate won, when any did):", flush=True)
    for (sc, pt, cn), n in sorted(candidate_win_counts.items()):
        print(f"    {sc:10s} {pt:20s} {cn:12s} n={n}", flush=True)
    if sum(passed_counts[(sc, True)] for sc in scorers) == 0:
        print("\n[FAIL] Nothing passed for either scorer. GEMMA_MODEL_ID/MISTRAL_MODEL_ID are likely "
              "wrong (or need a different revision/tokenizer) -- this is a checkpoint problem, not a "
              "prompt_type-alignment problem, if even continue_it/continue_pt or the human side fail. "
              "Do NOT run the full pass until this passes.", flush=True)
        sys.exit(1)
    print("\n[info] Expected (per the internal verification notes): "
          "human side + continue_it/continue_pt + majority of keywords/summary should pass; "
          "cot mostly fails even with the wide sweep; excerpt/short_keywords/summary_keywords "
          "essentially never pass. A materially different pattern here means something drifted "
          "(tokenizer/checkpoint) and should be investigated before a full run.", flush=True)


def main(limit_shards=None, scorers=("gemma-2b", "mistral-7b")):
    _ensure_dirs()
    torch_mod, device, dtype, tokenizers, models, embedder = load_models(scorers=scorers)
    prefix_map = load_prefix_map()
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
            for row in rows:
                row["prefix"] = prefix_map.get(row["id"])
            doc_results = [
                {"id": r["id"], "split": r["split"], "domain": r["domain"],
                 "Label": r["Label"], "prompt_type": r["prompt_type"], "temp": r["temp"]}
                for r in rows
            ]

            # Batched per scorer, per candidate, across documents in this
            # shard -- NOT one document at a time. score_batch already pads
            # variable-length sequences within a batch; OOM shrinks the
            # CURRENT batch and retries just that slice, not the whole shard.
            for sc, sliced_key, total_key in [k for k in SCORER_KEYS if k[0] in scorers]:
                tok, model = tokenizers[sc], models[sc]
                out_prefix = "Gemma" if sc == "gemma-2b" else "Mistral"
                row_candidates = [tokenize_candidates(tok, row["text"], row.get("prefix"), row.get(total_key)) for row in rows]
                per_candidate_results = {
                    cn: score_documents_for_candidate(torch_mod, model, tok, device, rows, row_candidates,
                                                       cn, sliced_key, total_key, sc)
                    for cn in CANDIDATE_NAMES
                }

                for idx in range(len(rows)):
                    per_candidate = {cn: per_candidate_results[cn].get(idx) for cn in CANDIDATE_NAMES}
                    best_name, best, both_passed = pick_best_candidate(per_candidate)
                    if best is None:
                        doc_results[idx][f"{out_prefix}_Alignment_Passed"] = False
                        doc_results[idx][f"{out_prefix}_Alignment_Best_Candidate"] = None
                        doc_results[idx][f"{out_prefix}_Alignment_Best_R"] = float("nan")
                        doc_results[idx][f"{out_prefix}_Alignment_Both_Passed"] = False
                        doc_results[idx][f"{out_prefix}_Feature_Window_Available"] = False
                        stats[f"{sc}_no_candidate_usable"] += 1
                        continue
                    passed = (not np.isnan(best["best_r"])) and best["best_r"] > ALIGNMENT_PEARSON_MIN \
                        and best["best_mean_abs_diff"] < ALIGNMENT_MEAN_ABS_DIFF_MAX
                    doc_results[idx][f"{out_prefix}_Alignment_Passed"] = bool(passed)
                    doc_results[idx][f"{out_prefix}_Alignment_Best_Candidate"] = best_name
                    doc_results[idx][f"{out_prefix}_Alignment_Best_R"] = best["best_r"]
                    doc_results[idx][f"{out_prefix}_Alignment_Mean_Abs_Diff"] = best["best_mean_abs_diff"]
                    doc_results[idx][f"{out_prefix}_Alignment_Best_Offset"] = best["best_abs_start"]
                    doc_results[idx][f"{out_prefix}_Alignment_Both_Passed"] = bool(both_passed)
                    feat_available = best["feat_arrays"] is not None and len(best["feat_arrays"]["log_probs"]) > 0
                    doc_results[idx][f"{out_prefix}_Feature_Window_Available"] = bool(feat_available)
                    # Alignment_Passed=False and Feature_Window_Available=False are DIFFERENT facts
                    # (provenance vs. corrected-span-too-short) -- both written, neither collapsed
                    # into the other, so a later filter can tell them apart.
                    if passed and feat_available:
                        summary = family_b_feature_summary(best["feat_arrays"])
                        for k, v in summary.items():
                            doc_results[idx][f"{out_prefix}_{k}"] = v
                        stats[f"{sc}_passed"] += 1
                    else:
                        stats[f"{sc}_failed_alignment_or_no_feature_window"] += 1

            for idx, row in enumerate(rows):
                doc_results[idx]["Embedding_Dispersion"] = embedding_dispersion(
                    embedder, row["text"], row.get("gagle_gemma_total_tokens"), row.get("gagle_mistral_total_tokens"))
                stats["written"] += 1

            out_rows = doc_results
            for row in out_rows:
                fout.write(json.dumps(row) + "\n")
            fout.flush()
            os.fsync(fout.fileno())
            append_line(SHARD_CKPT_PATH, shard)
            done_shards.add(shard)
            print(f"[shard complete] {i}/{len(remaining)} ({os.path.basename(shard)}) written={stats['written']} "
                  f"gemma_passed={stats['gemma-2b_passed']} mistral_passed={stats['mistral-7b_passed']}")

    complete = len(done_shards) >= len(shard_files) and not limit_shards
    manifest = {
        "shards_total": len(shard_files), "shards_done": len(done_shards),
        "shards_missing": sorted(set(shard_files) - set(done_shards))[:20],
        "scorers_run_this_invocation": list(scorers),
        "docs_written_this_run": stats["written"],
        "gemma_2b_alignment_passed_this_run": stats["gemma-2b_passed"],
        "mistral_7b_alignment_passed_this_run": stats["mistral-7b_passed"],
        "complete": bool(complete),
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print("\n=== SUMMARY ===")
    for k, v in manifest.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verify-alignment", type=int, default=None, metavar="N",
                    help="Check N documents' tokenization/scoring against GAGLE's stored values "
                         "and exit WITHOUT writing feature-store output. Run this before any full run.")
    p.add_argument("--limit-shards", type=int, default=None)
    p.add_argument("--scorers", nargs="+", choices=["gemma-2b", "mistral-7b"],
                    default=["gemma-2b", "mistral-7b"],
                    help="Which scorer model(s) to load and run. Default is both, for the GPU "
                         "machine. On a machine that can't fit mistral-7b (~28GB in this module's "
                         "fp32 non-CUDA dtype), pass '--scorers gemma-2b' to run gemma-2b only; "
                         "mistral-7b's rows simply aren't processed this invocation, not defaulted "
                         "or approximated.")
    args = p.parse_args()
    if args.verify_alignment:
        verify_alignment(args.verify_alignment, scorers=tuple(args.scorers))
    else:
        main(limit_shards=args.limit_shards, scorers=tuple(args.scorers))
