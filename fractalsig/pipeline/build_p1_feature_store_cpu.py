#!/usr/bin/env python3
"""
P1 Feature Store -- CPU pass. Reads gagle.zip LOCALLY (no network), single
pass over all 720 members, and for every document computes:

  Family A (fractal, per scorer: gemma-2b, mistral-7b): point-estimate S/H
    (with bootstrap-free CI half-width via calc_S/calc_H's return_std_err),
    windowed intra-document mean/var/min/max/drift-slope of S and H, and
    cross-scorer divergence |S_gemma-S_mistral|, |H_gemma-H_mistral|. Uses
    GAGLE's own precomputed token_scores directly -- no re-scoring, no GPU.

  Family C (style/discourse, text-only): sentence/paragraph length stats,
    sentence-transition entropy, discourse-marker frequency, lexical
    diversity (TTR), bigram repetition rate. Embedding dispersion is
    DEFERRED to the GPU pass (see note below) rather than computed here.

  GPU-job manifest: per document, the raw text + GAGLE's own sliced scores
    for both scorers, written as parquet shards -- so the separate Family B
    pass (GLTR ranks, exact entropy, cross-perplexity; needs Gemma-2B/
    Mistral-7B forward passes) can run on different hardware without ever
    re-deriving its own tokenization/windowing decision. See
    build_p1_feature_store_gpu.py's docstring for the join contract.

WHY EMBEDDING DISPERSION ISN'T COMPUTED HERE
  This machine has no torch/transformers/sentence-transformers installed (a
  hardware/dependency reality, not a design preference -- 16GB unified
  memory, no CUDA, 9GB free internal disk). Installing an embedding model
  here just for one Family C sub-feature would mean reading text a second
  time on top of the already-planned GPU pass, contradicting the "read
  every zip member exactly once" constraint. Instead it's folded into the
  SAME per-member loop as Family B, on the GPU machine, where the model
  stack already has to exist -- one extra column written alongside Family
  B's output, not a third pass over the corpus. Flagged explicitly per doc
  (Embedding_Dispersion_Deferred_To_GPU=True) so this is never silently
  mistaken for "not planned" vs. "computed elsewhere."

WHY FAMILY C USES A PER-DOCUMENT, RATIO-DERIVED WORD BUDGET, NOT FULL TEXT
  A direct sample of real GAGLE records showed tokens-per-word ratio varies
  a lot -- by domain (0.47 on scientific_papers human text to 1.39 on
  big_patent LLM text) AND by side (LLM text is consistently shorter than
  human text for the same record: e.g. one record had 735 LLM tokens vs.
  2047 human tokens). Computing Family C features (sentence length, TTR,
  n-gram repetition rate) on FULL, untruncated text would make every one of
  those features a partial proxy for raw document length rather than a
  style signal -- TTR falls monotonically with length, repetition rises --
  and since LLM/Human texts differ systematically in length, that's a
  length-confound baked into the label, exactly the kind of thing that
  inflates in-corpus accuracy and collapses under cross-domain testing
  (which is what this project's own Part II extension already found with
  a different feature set). Fix: convert Family A's fixed TOKEN window
  (up to CLIP_TO-WARMUP_DROP=400 tokens, or the whole doc if shorter) into
  an approximately-equivalent WORD count using THAT document's own real
  tokens-per-word ratio (computed directly from its own raw score-array
  length and its own raw text's word count -- no tokenizer needed, no
  global constant, no cross-document averaging), then truncate to that
  many words from the start. This targets the same effective length Family
  A operates on, per document, without needing token-to-character offsets.

CHECKPOINTING
  Same per-zip-member pattern as build_p0_local.py: data/p1_member_checkpoint.txt
  records one line per fully-processed member; a crash mid-member redoes
  that whole member (cheap local read) rather than tracking per-doc
  progress. No corpus-level pooling state needs rebuilding on resume here
  (unlike build_p0_local.py) since every P1 feature is per-document, so
  resume just skips already-done members -- no replay pass needed.

USAGE
    python -m fractalsig.pipeline.build_p1_feature_store_cpu                   # full run (resumable)
    python -m fractalsig.pipeline.build_p1_feature_store_cpu --limit-members 2  # smoke test
"""
from fractalsig import paths as _paths  # noqa: E402
import argparse
import collections
import fnmatch
import json
import os
import re
import sys
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fractalsig.features.fractal_math import generate_trajectory, calc_H, calc_S
from fractalsig.pipeline.build_p0_local import slice_scores, WARMUP_DROP, CLIP_TO, MIN_TOKENS, is_deviant_member

ZIP_PATH = os.environ.get("GAGLE_ZIP_PATH", _paths.GAGLE_ZIP_STR)
MEMBER_PATTERN = "gagle3/*/*.jsonl"
SCORERS = ["gemma-2b", "mistral-7b"]
REFERENCE_SCORER_FOR_RATIO = "gemma-2b"  # used to derive Family C's per-doc word budget

DATA_DIR = os.path.join(_paths.WORKSPACE_STR, "data")
FS_DIR = os.path.join(_paths.WORKSPACE_STR, "feature_store")
# GPU manifest shards carry per-doc raw text + both scorers' full score arrays
# (~2GB projected for the full corpus) -- written to the SSD (141GB free),
# not the internal disk (9GB free, shared with OS/swap during an hours-long
# run), and conveniently already on the drive that has to reach the GPU
# machine, avoiding a separate copy step. GAGLE_MANIFEST_DIR is the SAME env
# var build_p1_feature_store_gpu.py reads this directory from -- one knob
# controls both the write side (here) and the read side (GPU pass).
GPU_SHARD_DIR = os.environ.get("GAGLE_MANIFEST_DIR", _paths.MANIFEST_SHARD_DIR_STR)
OUT_PATH = os.path.join(DATA_DIR, "p1_feature_store_cpu.jsonl")
MEMBER_CKPT_PATH = os.path.join(DATA_DIR, "p1_member_checkpoint.txt")
MANIFEST_PATH = os.path.join(DATA_DIR, "p1_manifest.json")

WINDOW_SIZE = 200
WINDOW_STRIDE = 100
WINDOW_SCALE = [8, 16, 32, 48, 64, 96]  # validated separately: bias/R^2 comparable to full-doc H;
                                         # S at this window length is noisier than per-doc S (expected,
                                         # computed anyway, flagged unreliable -- see report)

DISCOURSE_MARKER_GROUPS = {
    "contrastive": ["however", "nevertheless", "nonetheless", "in contrast",
                    "on the other hand", "whereas", "conversely"],
    "additive": ["moreover", "furthermore", "in addition", "additionally",
                 "similarly", "likewise"],
    "causal": ["therefore", "thus", "consequently", "as a result",
               "accordingly", "hence"],
    "exemplifying": ["for example", "for instance", "specifically",
                      "namely", "in other words"],
}
# Word-boundary matching, not substring: "thus" must not fire inside
# "enthusiasm". \b works on the phrase as a whole since every marker here is
# space-separated plain words (no regex metacharacters to escape).
_DISCOURSE_MARKER_RES = {
    group: re.compile(r"\b(?:" + "|".join(re.escape(m) for m in markers) + r")\b")
    for group, markers in DISCOURSE_MARKER_GROUPS.items()
}

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Unicode-aware (2026-08-29, code review): the old [A-Za-z0-9']+ class is
# ASCII-only, so ANY non-ASCII letter -- not just RAID's homoglyph attack --
# breaks a word into fragments the same way U+200B did before the
# zero_width_space fix (e.g. "Müller" -> "M"/"ller", or a Cyrillic
# look-alike substituted mid-word). [^\W_] is "any Unicode word character
# except underscore" (re.UNICODE is Python 3's default), so a homoglyph
# substitution -- a REAL Unicode letter, just visually confusable, unlike
# U+200B's invisible/control nature -- is counted as part of the word
# instead of splitting it. Verified byte-identical to the old class on pure
# ASCII text (contractions, digits, underscore-exclusion, apostrophe
# handling all preserved) -- NOT a no-op like the ZWSP fix, though: GAGLE
# itself has non-ASCII content (measured 2026-08-29, see ledger), so this
# corrects already-reported GAGLE feature values too, not just RAID
# hygiene. Affects THIRTEEN columns -- corrected 2026-08-30 (code review);
# this list previously said twelve and omitted N_Sentences, which is exactly
# the enumeration the first targeted recompute copied and got wrong:
#   N_Words_Used, N_Sentences, Sent_Len_Mean/Std, TTR,
#   Bigram_Repetition_Rate, Para_Len_Mean/Std, Sent_Transition_Entropy,
#   and the four Discourse_Marker_*_Per100W columns (shared n_tokens
#   denominator).
# N_Sentences is gated by this regex NON-OBVIOUSLY: sent_word_counts is
# filtered by `if c > 0` (below), so a sentence the old ASCII-only class saw
# as zero-word was DROPPED entirely -- the regex decides which sentences
# survive, not just how words inside them are counted.
# Do not hand-maintain this list. Per the project notes, derive the affected set
# empirically (run old vs new regex over real non-ASCII samples and diff
# every returned key) before any future selective recompute.
# truncate_preserving_structure() uses _WORD_TOKEN_RE (whitespace-based),
# not this regex, so the truncation point is unaffected -- only counting.
_WORD_RE = re.compile(r"(?:[^\W_]|')+")
_PARA_SPLIT_RE = re.compile(r"\n\s*\n")
_WORD_TOKEN_RE = re.compile(r"\S+")
# Found 2026-08-28 via RAID's zero_width_space attack: _WORD_RE matches
# CONTIGUOUS runs of [A-Za-z0-9'], and U+200B (zero-width space) is not in
# that class, so "T​h​e" (the attack inserts it between every
# character) tokenizes as three separate one-character "words" instead of
# one three-letter word -- inflating N_Words_Used/Sent_Len_Mean by roughly
# the average word length on any zero_width_space-attacked document.
# GAGLE has no such attack, so stripping this is a no-op on all existing
# GAGLE data (verified: p1_feature_store_cpu.jsonl unchanged on a sample --
# see the module self-test). Strip before any _WORD_RE-based tokenization.
_ZERO_WIDTH_SPACE_RE = re.compile("\u200b")


def truncate_preserving_structure(text, word_budget):
    """Cut `text` to its first `word_budget` whitespace-delimited tokens,
    returning the ORIGINAL substring (not a rejoin) so newlines/paragraph
    breaks survive. A naive `" ".join(text.split()[:budget])` destroys every
    newline (text.split() splits on all whitespace including \\n), which
    silently zeroed out paragraph-length features and degraded sentence
    splitting on any doc without \\n-adjacent \".\"/\"?\"/\"!\" (heading-like
    breaks, patent claim numbering -- common in this corpus's big_patent/
    billsum domains). Caught by inspecting smoke-test output before the full
    run: Para_Len_Std was identically 0.0 for every row."""
    if word_budget is None:
        return text
    matches = list(_WORD_TOKEN_RE.finditer(text))
    if len(matches) <= word_budget:
        return text
    end = matches[word_budget - 1].end()
    return text[:end]


def _ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FS_DIR, exist_ok=True)
    os.makedirs(GPU_SHARD_DIR, exist_ok=True)


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


def make_windows(arr, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE):
    """Overlapping windows covering the full array, last window always
    flush with the end (never drops the tail). Falls back to the whole
    array as a single "window" if it's shorter than window_size -- shouldn't
    happen given MIN_TOKENS=320 > WINDOW_SIZE=200, kept as a defensive guard
    rather than an assumption."""
    n = len(arr)
    if n < window_size:
        return [arr]
    starts = list(range(0, n - window_size + 1, stride))
    if starts[-1] != n - window_size:
        starts.append(n - window_size)
    return [arr[s:s + window_size] for s in starts]


def window_summary(values):
    """mean/var/min/max/drift-slope across a list of per-window scalar
    estimates. drift-slope is the OLS slope of value vs. window index
    (linregress needs >=2 points; NaN otherwise)."""
    vals = np.array([v for v in values if not np.isnan(v)], dtype=float)
    if len(vals) == 0:
        return {"mean": float("nan"), "var": float("nan"), "min": float("nan"),
                "max": float("nan"), "drift": float("nan"), "n_windows": 0}
    drift = float("nan")
    if len(vals) >= 2:
        idx = np.arange(len(vals))
        from scipy.stats import linregress
        drift = linregress(idx, vals).slope
    return {
        "mean": float(np.mean(vals)), "var": float(np.var(vals)),
        "min": float(np.min(vals)), "max": float(np.max(vals)),
        "drift": float(drift), "n_windows": int(len(vals)),
    }


def family_a_for_scorer(sliced_scores):
    """Point estimate + windowed summaries for one document, one scorer."""
    trajectory, increment = generate_trajectory(sliced_scores)
    S, S_r2, S_se = calc_S(trajectory, return_std_err=True)
    H, H_r2, H_se = calc_H(increment, return_std_err=True)

    traj_windows = make_windows(trajectory)
    incr_windows = make_windows(increment)
    S_win_vals, S_win_r2s = [], []
    H_win_vals, H_win_r2s = [], []
    for tw in traj_windows:
        s_w, s_r2_w = calc_S(tw, scale_param=WINDOW_SCALE)
        S_win_vals.append(s_w)
        S_win_r2s.append(s_r2_w)
    for iw in incr_windows:
        h_w, h_r2_w = calc_H(iw, scale_param=WINDOW_SCALE)
        H_win_vals.append(h_w)
        H_win_r2s.append(h_r2_w)

    return {
        "S": float(S), "S_R2": float(S_r2), "S_CI_halfwidth": float(1.96 * S_se) if not np.isnan(S_se) else float("nan"),
        "H": float(H), "H_R2": float(H_r2), "H_CI_halfwidth": float(1.96 * H_se) if not np.isnan(H_se) else float("nan"),
        "S_window": window_summary(S_win_vals),
        "H_window": window_summary(H_win_vals),
    }


def family_c_style(text, word_budget):
    """Text-only style/discourse features on a per-document, ratio-matched
    word budget (see module docstring). word_budget=None means use all
    words (ratio unattainable -- flagged via Family_C_Full_Text_Fallback)."""
    fallback = word_budget is None
    # Computed on the PRE-truncation `text`, not `text_used`: the flag means
    # "this document's raw source text contains U+200B", independent of
    # where the word budget happens to cut it off -- a backfill computed
    # from the full raw text (no word_budget available) must agree with a
    # freshly-built value for every document, including ones where U+200B
    # falls past the truncation point. Also reused as the guard below so a
    # GAGLE-scale corpus pass (never contains U+200B) skips the regex
    # engine entirely instead of paying for a no-op .sub() on every doc.
    zwsp_present = _ZERO_WIDTH_SPACE_RE.search(text) is not None
    text_used = truncate_preserving_structure(text, word_budget)
    if zwsp_present:
        text_used = _ZERO_WIDTH_SPACE_RE.sub("", text_used)

    sentences = [s for s in _SENT_SPLIT_RE.split(text_used) if s.strip()]
    sent_word_counts = [len(_WORD_RE.findall(s)) for s in sentences]
    sent_word_counts = [c for c in sent_word_counts if c > 0]

    paragraphs = [p for p in _PARA_SPLIT_RE.split(text_used) if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [p for p in text_used.split("\n") if p.strip()]
    para_word_counts = [len(_WORD_RE.findall(p)) for p in paragraphs]
    para_word_counts = [c for c in para_word_counts if c > 0]

    tokens_lower = [w.lower() for w in _WORD_RE.findall(text_used)]
    n_tokens = len(tokens_lower)
    ttr = (len(set(tokens_lower)) / n_tokens) if n_tokens > 0 else float("nan")

    if n_tokens >= 2:
        bigrams = list(zip(tokens_lower[:-1], tokens_lower[1:]))
        seen = set()
        repeats = 0
        for bg in bigrams:
            if bg in seen:
                repeats += 1
            seen.add(bg)
        bigram_repetition_rate = repeats / len(bigrams)
    else:
        bigram_repetition_rate = float("nan")

    text_lower = text_used.lower()
    n_words_for_marker_rate = max(n_tokens, 1)
    # Word-boundary regex per group, not substring count: a naive
    # text.count("thus") fires inside "enthusiasm", and a single lumped sum
    # across 25 markers throws away which discourse function is present.
    marker_freqs = {
        f"Discourse_Marker_{group.capitalize()}_Per100W":
            100.0 * len(pattern.findall(text_lower)) / n_words_for_marker_rate
        for group, pattern in _DISCOURSE_MARKER_RES.items()
    }

    # Sentence-transition entropy: bucket each sentence's word count into
    # short(<10)/medium(10-25)/long(>25) bins, build the empirical
    # consecutive-sentence transition matrix, report the bin-frequency-
    # weighted average conditional entropy. NaN if <2 sentences.
    def _bin(c):
        if c < 10:
            return 0
        if c <= 25:
            return 1
        return 2

    transition_entropy = float("nan")
    if len(sent_word_counts) >= 2:
        bins = [_bin(c) for c in sent_word_counts]
        trans_counts = collections.defaultdict(lambda: collections.Counter())
        for a, b in zip(bins[:-1], bins[1:]):
            trans_counts[a][b] += 1
        entropies, weights = [], []
        for src, counter in trans_counts.items():
            total = sum(counter.values())
            probs = np.array([c / total for c in counter.values()])
            ent = -np.sum(probs * np.log2(probs))
            entropies.append(ent)
            weights.append(total)
        weights = np.array(weights, dtype=float)
        transition_entropy = float(np.average(entropies, weights=weights))

    return {
        "Sent_Len_Mean": float(np.mean(sent_word_counts)) if sent_word_counts else float("nan"),
        "Sent_Len_Std": float(np.std(sent_word_counts)) if sent_word_counts else float("nan"),
        "Para_Len_Mean": float(np.mean(para_word_counts)) if para_word_counts else float("nan"),
        "Para_Len_Std": float(np.std(para_word_counts)) if para_word_counts else float("nan"),
        "Sent_Transition_Entropy": transition_entropy,
        **marker_freqs,
        "TTR": float(ttr),
        "Bigram_Repetition_Rate": float(bigram_repetition_rate),
        "N_Sentences": len(sent_word_counts),
        "N_Words_Used": n_tokens,
        "Family_C_Full_Text_Fallback": fallback,
        "ZWSP_Present_In_Source_Text": zwsp_present,
    }


def compute_word_budget(text, n_tokens_total):
    """Per-document tokens-per-word ratio, applied to the SAME effective
    token length Family A uses (CLIP_TO-WARMUP_DROP, or the whole doc if
    shorter than CLIP_TO -- mirroring slice_scores's own branching exactly).
    Returns None (meaning: use full text) if the ratio can't be computed.

    Takes the token COUNT directly (not a raw score array) so callers that
    only have a count (e.g. build_p1_feature_store_gpu.py's
    embedding_dispersion, which has gagle_gemma_total_tokens from the
    manifest but not the original array) don't need to materialize a
    throwaway list purely to satisfy a len() call."""
    if not n_tokens_total or not text:
        return None
    n_words_total = len(text.split())
    if n_words_total == 0:
        return None
    ratio = n_tokens_total / n_words_total
    if ratio <= 0:
        return None
    if n_tokens_total >= CLIP_TO:
        target_tokens = CLIP_TO - WARMUP_DROP
    else:
        target_tokens = n_tokens_total
    budget = max(1, round(target_tokens / ratio))
    return min(budget, n_words_total)


def extract_docs(entry, split):
    """Yield 0-2 doc dicts (llm side, human side). Mirrors build_p0_local's
    extract_rows exactly for id/dedup-key construction (domain MUST be in
    the human key -- see build_p0_local's docstring for the collision bug
    this guards against; split must NOT be, per X-01 -- see build_p0_local's
    docstring for why dropping it still doesn't let a deviant-source row
    win), but carries BOTH scorers' raw scores and the raw text, which
    build_p0_local's gemma-2b-only extractor doesn't need."""
    domain = entry.get("dataset_name")
    scores_block = entry.get("scores", {})

    # entry["prefix"] is the prompt-type-specific instruction/context GAGLE's
    # own scores were computed with (as real preceding LM context for
    # llm_text, and not used at all for original_text -- confirmed via
    # the internal verification notes). It was not
    # captured here before that investigation; carried through to the GPU
    # manifest now since the GPU pass needs it to reproduce GAGLE's scores.
    prefix = entry.get("prefix")

    llm_text = entry.get("llm_text")
    llm_raw = {sc: scores_block.get(sc, {}).get("llm_text") for sc in SCORERS}
    if llm_text and any(llm_raw.values()):
        yield {
            "id": str(entry.get("_id")),
            "split": split, "domain": domain,
            "label": entry.get("model", "Unknown_LLM"),
            "prompt_type": entry.get("prompt_type"),
            "temp": entry.get("temperature"),
            "text": llm_text,
            "prefix": prefix,
            "raw_scores": llm_raw,
        }

    human_text = entry.get("original_text")
    human_raw = {sc: scores_block.get(sc, {}).get("original_text") for sc in SCORERS}
    if human_text and any(human_raw.values()):
        yield {
            "id": f"{domain}_{entry.get('dataset_id')}_human",
            "split": split, "domain": domain,
            "label": "Human",
            "prompt_type": "continue",
            "temp": "Human",
            "text": human_text,
            "prefix": prefix,
            "raw_scores": human_raw,
        }


def process_doc(doc, stats):
    """Returns (feature_row, gpu_manifest_row). feature_row is None if
    neither scorer meets MIN_TOKENS (doc unusable for Family A at all --
    Family C alone isn't a reason to keep a row, since the whole point of
    this feature store is the fractal+style+GLTR join)."""
    sliced = {sc: slice_scores(doc["raw_scores"].get(sc)) for sc in SCORERS}
    if not any(v is not None for v in sliced.values()):
        stats["skipped_no_scorer_meets_min_tokens"] += 1
        return None, None

    row = {
        "id": doc["id"], "split": doc["split"], "domain": doc["domain"],
        "Label": doc["label"], "prompt_type": doc["prompt_type"], "temp": doc["temp"],
        "Holder_PerDoc_Unreliable": True,
    }

    scorer_point_S = {}
    scorer_point_H = {}
    for sc in SCORERS:
        prefix = "Gemma" if sc == "gemma-2b" else "Mistral"
        if sliced[sc] is None:
            row[f"{prefix}_available"] = False
            continue
        row[f"{prefix}_available"] = True
        fa = family_a_for_scorer(sliced[sc])
        row[f"{prefix}_S"] = fa["S"]
        row[f"{prefix}_S_R2"] = fa["S_R2"]
        row[f"{prefix}_S_CI_halfwidth"] = fa["S_CI_halfwidth"]
        row[f"{prefix}_H"] = fa["H"]
        row[f"{prefix}_H_R2"] = fa["H_R2"]
        row[f"{prefix}_H_CI_halfwidth"] = fa["H_CI_halfwidth"]
        for stat_name, val in fa["S_window"].items():
            row[f"{prefix}_S_window_{stat_name}"] = val
        for stat_name, val in fa["H_window"].items():
            row[f"{prefix}_H_window_{stat_name}"] = val
        scorer_point_S[sc] = fa["S"]
        scorer_point_H[sc] = fa["H"]

    if len(scorer_point_S) == 2:
        row["Scorer_Divergence_S"] = float(abs(scorer_point_S["gemma-2b"] - scorer_point_S["mistral-7b"]))
        row["Scorer_Divergence_H"] = float(abs(scorer_point_H["gemma-2b"] - scorer_point_H["mistral-7b"]))
    else:
        row["Scorer_Divergence_S"] = float("nan")
        row["Scorer_Divergence_H"] = float("nan")

    ratio_ref_raw = doc["raw_scores"].get(REFERENCE_SCORER_FOR_RATIO) or doc["raw_scores"].get(
        [s for s in SCORERS if s != REFERENCE_SCORER_FOR_RATIO][0])
    word_budget = compute_word_budget(doc["text"], len(ratio_ref_raw) if ratio_ref_raw else None)
    row.update(family_c_style(doc["text"], word_budget))
    row["Embedding_Dispersion_Deferred_To_GPU"] = True
    stats["written"] += 1

    # Total (unsliced) token counts for both scorers travel alongside the
    # sliced arrays specifically so the GPU pass can verify its OWN
    # retokenization of `text` produces the same length before trusting that
    # its [WARMUP_DROP:CLIP_TO] window lines up with GAGLE's. Without this,
    # a silent tokenizer-version mismatch on the GPU machine would shift the
    # window by some offset and there would be no way to catch it from the
    # manifest alone -- the sliced arrays' length (320-400, by construction)
    # can't reveal that, since it's clipped to the same range regardless.
    gpu_row = {
        "id": doc["id"], "split": doc["split"], "domain": doc["domain"],
        "Label": doc["label"], "prompt_type": doc["prompt_type"], "temp": doc["temp"],
        "side": "human" if doc["label"] == "Human" else "llm",
        "text": doc["text"],
        "prefix": doc.get("prefix"),
        "gagle_gemma_sliced_scores": sliced.get("gemma-2b"),
        "gagle_mistral_sliced_scores": sliced.get("mistral-7b"),
        "gagle_gemma_total_tokens": len(doc["raw_scores"].get("gemma-2b") or []) or None,
        "gagle_mistral_total_tokens": len(doc["raw_scores"].get("mistral-7b") or []) or None,
    }
    return row, gpu_row


def process_member(zf, member, stats):
    """Returns LLM-side (out_rows, gpu_rows) only (empty if none this member).
    Human docs are buffered into HUMAN_BEST_GLOBAL, not computed or emitted
    here -- see main()'s final flush and build_p0_local.py's matching
    docstring for why (X-01: picking the best of several duplicate copies
    needs every candidate seen first, which first-seen-wins during a single
    streaming pass cannot guarantee -- verified false empirically 2026-08-22,
    see build_p0_local.py's process_member docstring)."""
    split = member.split("/")[1]
    out_rows, gpu_rows = [], []
    with zf.open(member) as fh:
        for line_bytes in fh:
            line = line_bytes.decode("utf-8").strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                stats["json_errors"] += 1
                continue
            stats["raw_entries_seen"] += 1
            for doc in extract_docs(entry, split):
                if doc["label"] == "Human":
                    deviant = is_deviant_member(member)
                    existing = HUMAN_BEST_GLOBAL.get(doc["id"])
                    if existing is None:
                        HUMAN_BEST_GLOBAL[doc["id"]] = {"doc": doc, "member": member, "deviant": deviant}
                    else:
                        stats["skipped_dup_human"] += 1
                        if existing["deviant"] and not deviant:
                            HUMAN_BEST_GLOBAL[doc["id"]] = {"doc": doc, "member": member, "deviant": deviant}
                    continue
                row, gpu_row = process_doc(doc, stats)
                if row is not None:
                    out_rows.append(row)
                    gpu_rows.append(gpu_row)
    return out_rows, gpu_rows


HUMAN_BEST_GLOBAL = {}  # X-01: id -> {"doc":..., "member":..., "deviant": bool}


def write_gpu_shard(member, gpu_rows):
    if not gpu_rows:
        return
    safe_name = member.replace("/", "__")
    tmp_path = os.path.join(GPU_SHARD_DIR, safe_name + ".parquet.tmp")
    final_path = os.path.join(GPU_SHARD_DIR, safe_name + ".parquet")
    table = pa.Table.from_pylist(gpu_rows)
    pq.write_table(table, tmp_path)
    os.replace(tmp_path, final_path)


def rebuild_human_dedup_state(zf, done_members):
    """On resume, replay already-done members' Human docs (full doc dicts,
    via extract_docs -- not just ids: the final flush needs the actual text
    and raw scores to run process_doc on whichever candidate wins, and the
    winner might be sitting in an already-checkpointed member) so
    HUMAN_BEST_GLOBAL is correct before processing new members. Mirrors
    build_p0_local.py's replay-on-resume pattern. This is still cheap
    relative to a fresh run -- it skips process_doc (Family A/C computation),
    just not the doc-dict construction extract_docs already does.

    Uses extract_docs itself (not a hand-rolled reimplementation) so the id
    scheme and preference logic can't silently drift from process_member's."""
    if not done_members:
        return
    print(f"[resume] replaying {len(done_members)} already-done members to rebuild human-dedup state...")
    for i, member in enumerate(sorted(done_members), 1):
        split = member.split("/")[1]
        with zf.open(member) as fh:
            for line_bytes in fh:
                line = line_bytes.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for doc in extract_docs(entry, split):
                    if doc["label"] != "Human":
                        continue
                    deviant = is_deviant_member(member)
                    existing = HUMAN_BEST_GLOBAL.get(doc["id"])
                    if existing is None:
                        HUMAN_BEST_GLOBAL[doc["id"]] = {"doc": doc, "member": member, "deviant": deviant}
                    elif existing["deviant"] and not deviant:
                        HUMAN_BEST_GLOBAL[doc["id"]] = {"doc": doc, "member": member, "deviant": deviant}
        if i % 50 == 0:
            print(f"[resume] replayed {i}/{len(done_members)}")


def main(limit_members=None):
    if not os.path.exists(ZIP_PATH):
        print(f"[abort] {ZIP_PATH} not found.")
        sys.exit(1)

    _ensure_dirs()
    done_members = load_line_set(MEMBER_CKPT_PATH)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        all_members = sorted(m for m in zf.namelist() if fnmatch.fnmatch(m, MEMBER_PATTERN))
        print(f"[info] {len(all_members)} members match {MEMBER_PATTERN}; "
              f"{len(done_members)} already checkpointed done.")

        rebuild_human_dedup_state(zf, done_members)

        remaining = [m for m in all_members if m not in done_members]
        if limit_members:
            remaining = remaining[:limit_members]
            print(f"[smoke-test] limiting to first {limit_members} not-yet-done members")

        stats = collections.Counter()
        with open(OUT_PATH, "a") as fout:
            for i, member in enumerate(remaining, 1):
                rows, gpu_rows = process_member(zf, member, stats)
                for row in rows:
                    fout.write(json.dumps(row) + "\n")
                fout.flush()
                os.fsync(fout.fileno())
                write_gpu_shard(member, gpu_rows)  # LLM-side rows only now
                append_line(MEMBER_CKPT_PATH, member)
                done_members.add(member)
                print(f"[member complete] {i}/{len(remaining)} ({member}) "
                      f"written={stats['written']} raw_seen={stats['raw_entries_seen']}")

            complete = (len(done_members) >= len(all_members)) and not limit_members
            coverage_ok = set(done_members) == set(all_members)

            if complete and coverage_ok:
                print(f"\n[info] all members scanned -- flushing {len(HUMAN_BEST_GLOBAL)} "
                      f"deduplicated human docs now that every candidate has been seen...")
                human_gpu_rows = []
                for id_, entry in sorted(HUMAN_BEST_GLOBAL.items()):
                    row, gpu_row = process_doc(entry["doc"], stats)
                    if row is not None:
                        fout.write(json.dumps(row) + "\n")
                        human_gpu_rows.append(gpu_row)
                fout.flush()
                os.fsync(fout.fileno())
                # One dedicated shard for every winning human doc, rather than
                # trying to retroactively inject rows into the per-member
                # shards already written above (those are keyed by member
                # name and may not be the member the winning copy came from).
                # The GPU pass reads every *.parquet in GPU_SHARD_DIR as one
                # dataset, so which physical file a row lives in doesn't matter.
                write_gpu_shard("human_dedup_x01", human_gpu_rows)
            else:
                print(f"\n[incomplete] {len(remaining)} members left unprocessed this run -- "
                      f"human docs are NOT written yet (dedup needs the full corpus scanned "
                      f"first). Rerun this exact command to continue.")

        # X-01 invariant, dedup site 2 (the consequential one -- this is the
        # file that actually feeds Family A/C into the feature store). See
        # build_p0_local.py's matching check for the full rationale.
        deviant_survivors = {
            id_: entry["member"] for id_, entry in HUMAN_BEST_GLOBAL.items() if entry["deviant"]
        }
        if deviant_survivors:
            print(f"\n[X-01 INVARIANT VIOLATION] {len(deviant_survivors)}/"
                  f"{len(HUMAN_BEST_GLOBAL)} surviving human rows are sourced from a "
                  f"deviant member (summary_keywords or mistral_7b_*). Examples: "
                  f"{list(deviant_survivors.items())[:10]}")
        else:
            print(f"\n[X-01 invariant holds] 0/{len(HUMAN_BEST_GLOBAL)} surviving human "
                  f"rows sourced from summary_keywords or mistral_7b_* members.")

        manifest = {
            "members_total": len(all_members),
            "members_done": len(done_members),
            "members_missing": sorted(set(all_members) - set(done_members))[:20],
            "raw_entries_seen_this_run": stats["raw_entries_seen"],
            "docs_written_this_run": stats["written"],
            "skipped_dup_human_this_run": stats["skipped_dup_human"],
            "skipped_no_scorer_meets_min_tokens_this_run": stats["skipped_no_scorer_meets_min_tokens"],
            "json_errors_this_run": stats["json_errors"],
            "human_rows_total": len(HUMAN_BEST_GLOBAL),
            "x01_deviant_survivors": len(deviant_survivors),
            "x01_deviant_survivor_ids": list(deviant_survivors.keys())[:20],
            "complete": bool(complete),
            "member_coverage_gap_free": bool(coverage_ok),
        }
        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2)
        print("\n=== SUMMARY ===")
        for k, v in manifest.items():
            print(f"  {k}: {v}")
        if complete and coverage_ok:
            print(f"\n[complete] CPU pass done. Feature rows: {OUT_PATH}. "
                  f"GPU manifest shards: {GPU_SHARD_DIR}/ (read as a pyarrow dataset).")
        else:
            print("\n[incomplete] Rerun this exact command to continue.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit-members", type=int, default=None)
    args = p.parse_args()
    main(limit_members=args.limit_members)
