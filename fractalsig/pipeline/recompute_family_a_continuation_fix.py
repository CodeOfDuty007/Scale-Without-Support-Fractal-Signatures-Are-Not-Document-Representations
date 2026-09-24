#!/usr/bin/env python3
"""
Recomputes Family A (gemma-2b only) for the continue_it/continue_pt rows
affected by the prefix-contamination defect confirmed in
the internal verification notes and
the internal verification notes: GAGLE's stored gemma-2b
score array for these two prompt_types covers the prompt PREFIX concatenated
with the generated text, not the generated text alone, and the pipeline's
flat WARMUP_DROP=64 doesn't clear prefixes that average ~94-99 tokens.

Uses the per-record boundary formula (not a fixed constant -- see the
correction-results report's "Seam-offset prerequisite, closed" section for
why a fixed constant was wrong for the ~90% of records with zero token-count
delta):

    delta = (prefix_len_tokens + text_len_tokens - 2) - gagle_total_tokens
    boundary = prefix_len_tokens - delta   # == gagle_total_tokens - text_len_tokens + 2
    start = boundary + WARMUP_DROP         # "extra_warmup" variant, chosen
                                            # 2026-08-05: matches the paper's literal
                                            # "drop first 64 tokens for warm-up" applied
                                            # to the generation itself, not the prefix.

Scope: gemma-2b only. Part I's headline table and this whole investigation
never scored/verified mistral-7b's prefix-conditioning -- P1's Mistral_S/
Mistral_H columns for these rows are UNTOUCHED here and remain an open
question, not silently assumed fine or silently "fixed" by extension.

Output: a NEW file (FAMILY_A_CORRECTED_PATH), NOT an in-place edit of
data/p1_feature_store_cpu.jsonl -- old and corrected values are both
auditable, per the verification checklist's explicit instruction. Join
against the main feature store on `id` downstream.

USAGE
    python -m fractalsig.pipeline.recompute_family_a_continuation_fix
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
import sys
import zipfile

import numpy as np

from fractalsig.pipeline.build_p0_local import WARMUP_DROP, CLIP_TO, MIN_TOKENS
from fractalsig.pipeline.build_p1_feature_store_cpu import family_a_for_scorer

ZIP_PATH = _paths.GAGLE_ZIP_STR
MEMBER_PATTERN_PREFIX = "gagle3/"
PREFIX_CONDITIONED_TYPES = {"continue_it", "continue_pt"}

DATA_DIR = os.path.join(_paths.WORKSPACE_STR, "data")
OUT_PATH = os.path.join(DATA_DIR, "p1_family_a_corrected_continuation.jsonl")
MANIFEST_PATH = os.path.join(DATA_DIR, "p1_family_a_corrected_continuation_manifest.json")


def compute_boundary(prefix_len_tokens, text_len_tokens, gagle_total_tokens):
    """Where the text portion begins within GAGLE's stored prefix+text score
    array, per-record (not a fixed constant -- see the correction-results
    report's "Seam-offset prerequisite, closed" section). Shared by this
    script's Family A recompute and build_p1_feature_store_gpu.py's Family B
    feature-extraction window, so the two never drift apart."""
    delta = (prefix_len_tokens + text_len_tokens - 2) - gagle_total_tokens
    return prefix_len_tokens - delta


def slice_scores_per_record(raw, prefix_len_tokens, text_len_tokens):
    """Returns (sliced_array_or_None, drop_reason_or_None)."""
    g = len(raw)
    boundary = compute_boundary(prefix_len_tokens, text_len_tokens, g)
    if boundary < 0:
        return None, "negative_boundary"
    start = boundary + WARMUP_DROP
    if start >= g:
        return None, "start_past_end"
    end = min(start + (CLIP_TO - WARMUP_DROP), g)
    sliced = raw[start:end]
    if len(sliced) < MIN_TOKENS:
        return None, "below_min_tokens"
    return sliced, None


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/gemma-2b")

    os.makedirs(DATA_DIR, exist_ok=True)
    n_seen = 0
    n_written = 0
    drop_reasons = {}

    with zipfile.ZipFile(ZIP_PATH) as zf, open(OUT_PATH, "w") as fout:
        members = sorted(m for m in zf.namelist() if m.startswith(MEMBER_PATTERN_PREFIX) and m.endswith(".jsonl"))
        print(f"[info] scanning {len(members)} members for continue_it/continue_pt rows", flush=True)
        for i, member in enumerate(members, 1):
            with zf.open(member) as fh:
                for line in fh:
                    entry = json.loads(line)
                    pt = entry.get("prompt_type")
                    if pt not in PREFIX_CONDITIONED_TYPES:
                        continue
                    prefix = entry.get("prefix")
                    raw = entry.get("scores", {}).get("gemma-2b", {}).get("llm_text")
                    text = entry.get("llm_text")
                    doc_id = str(entry.get("_id"))
                    if not prefix or not raw or not text:
                        drop_reasons["missing_field"] = drop_reasons.get("missing_field", 0) + 1
                        n_seen += 1
                        continue
                    n_seen += 1

                    prefix_len = len(tok(prefix)["input_ids"])
                    text_len = len(tok(text)["input_ids"])
                    sliced, reason = slice_scores_per_record(raw, prefix_len, text_len)
                    if sliced is None:
                        drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                        continue

                    fa = family_a_for_scorer(sliced)
                    row = {
                        "id": doc_id,
                        "prompt_type": pt,
                        "domain": entry.get("dataset_name"),
                        "label": entry.get("model"),
                        "Gemma_S_corrected": fa["S"],
                        "Gemma_S_R2_corrected": fa["S_R2"],
                        "Gemma_H_corrected": fa["H"],
                        "Gemma_H_R2_corrected": fa["H_R2"],
                        "boundary_used": prefix_len - ((prefix_len + text_len - 2) - len(raw)),
                        "sliced_window_len": len(sliced),
                    }
                    fout.write(json.dumps(row) + "\n")
                    n_written += 1
            if i % 150 == 0:
                fout.flush()
                print(f"[info] {i}/{len(members)} members scanned, seen={n_seen} written={n_written}", flush=True)

    manifest = {
        "n_seen_continue_it_or_pt": n_seen,
        "n_written": n_written,
        "n_dropped": n_seen - n_written,
        "drop_reasons": drop_reasons,
        "drop_rate": (n_seen - n_written) / n_seen if n_seen else None,
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print("\n=== SUMMARY ===", flush=True)
    for k, v in manifest.items():
        print(f"  {k}: {v}", flush=True)
    print(f"\n[info] wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
