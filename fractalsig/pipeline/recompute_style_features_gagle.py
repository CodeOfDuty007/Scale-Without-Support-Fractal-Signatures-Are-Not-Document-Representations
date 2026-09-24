#!/usr/bin/env python3
"""Targeted recompute of GAGLE's 12 Unicode-_WORD_RE-affected STYLE columns
in feature_store/v1_gagle.parquet, WITHOUT a full build_p1_feature_store_cpu.py
rebuild (43min) -- approved targeted path (2026-08-29), following this
project's own light-backfill precedent (fractalsig/pipeline/backfill_short_window_flag.py:
read only what's needed from gagle.zip, join by id, never rewrite an
existing output in place).

Inputs:
  - data/p1_word_regex_token_counts.jsonl: per-id {gemma_total, mistral_total}
    token counts, extracted via a dedicated light pass over gagle.zip
    (data-extraction only, no Family A/B/C compute) -- see the ledger entry
    for its own timing (5.00 min, full corpus).
  - data/all_doc_texts.jsonl: per-id cached raw text (already existed).
  - feature_store/v1_gagle.parquet.pre_zwsp_flag_backup: the CURRENT store
    (pre-_WORD_RE-fix), used for row order / all other columns / id set.

Recomputes the 13 columns family_c_style() returns that depend on
_WORD_RE: N_Words_Used, N_Sentences, Sent_Len_Mean, Sent_Len_Std, TTR,
Bigram_Repetition_Rate, Para_Len_Mean, Para_Len_Std, Sent_Transition_Entropy,
and the 4 Discourse_Marker_*_Per100W columns. Family A/B/Embedding_Dispersion
and the two provenance flags are copied through UNCHANGED from the existing
store -- this script does not touch them.

CORRECTED 2026-08-29 (second pass): the first run of this script omitted
N_Sentences, on the hand-reasoned assumption that only word-COUNTING columns
depended on _WORD_RE. That reasoning was wrong -- N_Sentences is computed as
len(sent_word_counts) AFTER `sent_word_counts = [c for c in sent_word_counts
if c > 0]` (build_p1_feature_store_cpu.py:290-291), so _WORD_RE gates which
sentences even SURVIVE, not just how words within them are counted. Caught
via RAID's full rebuild (which recomputes every family_c_style() key fresh,
unlike this script's selective column list) showing 179 real N_Sentences
diffs concentrated in `homoglyph` rows. The affected-column set is now
derived EMPIRICALLY (diff every returned key, old regex vs new, on 500
real non-ASCII GAGLE texts), not reasoned about by inspection -- see the
ledger entry for the derivation script and its exact result (this exact
13-column set, nothing else, confirming Family_C_Full_Text_Fallback and
ZWSP_Present_In_Source_Text never move, as expected since both are computed
before tokenization).
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = _paths.WORKSPACE_STR
D = lambda *p: os.path.join(HERE, "data", *p)
FS = lambda *p: os.path.join(HERE, "feature_store", *p)

from fractalsig.pipeline.build_p1_feature_store_cpu import compute_word_budget, family_c_style  # noqa: E402

STYLE_RECOMPUTE_COLS = [
    "N_Words_Used", "N_Sentences", "Sent_Len_Mean", "Sent_Len_Std", "TTR",
    "Bigram_Repetition_Rate", "Para_Len_Mean", "Para_Len_Std",
    "Sent_Transition_Entropy",
    "Discourse_Marker_Contrastive_Per100W", "Discourse_Marker_Additive_Per100W",
    "Discourse_Marker_Causal_Per100W", "Discourse_Marker_Exemplifying_Per100W",
]


def main():
    print("[recompute] loading token counts...", flush=True)
    token_counts = {}
    with open(D("p1_word_regex_token_counts.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            # first-seen-wins for duplicate human ids across members --
            # matches the pipeline's own dedup precedent; F-D established
            # duplicates are feature-identical, so which copy wins doesn't
            # matter for these token counts in practice.
            if r["id"] not in token_counts:
                token_counts[r["id"]] = r
    print(f"[recompute] {len(token_counts)} unique ids with token counts", flush=True)

    print("[recompute] loading cached text...", flush=True)
    texts = {}
    with open(D("all_doc_texts.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            texts[r["id"]] = r["text"]
    print(f"[recompute] {len(texts)} ids with cached text", flush=True)

    print("[recompute] loading current store...", flush=True)
    df = pd.read_parquet(FS("v1_gagle.parquet"))  # the CURRENT promoted store
    # (first pass's bug: read .pre_zwsp_flag_backup here, an 88-col pre-flag
    # store, silently losing ZWSP_Present_In_Source_Text -- caught before
    # promotion and fixed via a merge; this second pass reads the correct,
    # current 89-col store directly instead.)
    print(f"[recompute] {len(df)} rows, {len(df.columns)} cols in current store", flush=True)

    n_ok, n_missing_tokens, n_missing_text = 0, 0, 0
    # Seed with the store's EXISTING values, not None. A row we cannot
    # recompute must keep what it already had -- pre-filling with None and
    # then assigning the whole column back (below) would overwrite good data
    # with nulls AND flip N_Words_Used/N_Sentences from int64 to object.
    # (Found by code review 2026-08-30. The 2026-08-29 run happened to hit
    # missing_text=0/missing_tokens=0 so nothing was corrupted -- verified
    # post-hoc: both int columns are int64 with 0 nulls in the promoted
    # store -- but the hazard was real and is closed here, not just noted.)
    new_cols = {c: list(df[c]) for c in STYLE_RECOMPUTE_COLS}

    for i, rid in enumerate(df["id"].to_numpy()):
        text = texts.get(rid)
        if text is None:
            n_missing_text += 1
            continue  # existing value already seeded above; leave it alone
        tc = token_counts.get(rid)
        if tc is None:
            # No real token count -> compute_word_budget would return None ->
            # family_c_style would silently take its FULL-TEXT fallback path,
            # producing length-inflated values that do not correspond to what
            # the original build wrote, while Family_C_Full_Text_Fallback
            # (not in STYLE_RECOMPUTE_COLS) would still read False. Carrying
            # the existing value through is the only honest option.
            n_missing_tokens += 1
            continue
        ref_total = tc["gemma_total"] or tc["mistral_total"]
        word_budget = compute_word_budget(text, ref_total)
        feats = family_c_style(text, word_budget)
        for c in STYLE_RECOMPUTE_COLS:
            new_cols[c][i] = feats[c]
        n_ok += 1
        if (i + 1) % 50000 == 0:
            print(f"[recompute]   {i+1}/{len(df)} rows processed", flush=True)

    print(f"[recompute] DONE: ok={n_ok} missing_text={n_missing_text} missing_tokens={n_missing_tokens}", flush=True)

    for c in STYLE_RECOMPUTE_COLS:
        df[c] = new_cols[c]

    out_path = FS("v1_gagle_v2.parquet")
    df.to_parquet(out_path, index=False)
    print(f"[recompute] written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
