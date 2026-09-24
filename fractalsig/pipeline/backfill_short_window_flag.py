#!/usr/bin/env python3
"""
Backfills a Used_Short_Window flag per (id, scorer), for the P1 CPU feature
store's independent-review Finding 2: the no-warmup-drop fallback branch in
slice_scores (raw array shorter than CLIP_TO=464 -> keep the whole array,
warmup included, rather than dropping the first 64 tokens) is deliberate
and consistently implemented, but is label-correlated (measured: 4.85% of
rows, 100% LLM-labeled in the reviewed sample) with no flag to filter on.

A LIGHT pass -- reads only raw score-array LENGTHS from gagle.zip (no
Family A/B/C recompute, no full JSON field extraction beyond what's needed
for the id/length), joined to the existing feature store by id. Does NOT
modify data/p1_feature_store_cpu.jsonl -- writes a separate, small
id-keyed flags file that a later step can join in, consistent with never
overwriting an already-written pipeline output.

USAGE
    python -m fractalsig.pipeline.backfill_short_window_flag
"""
from fractalsig import paths as _paths  # noqa: E402
import fnmatch
import json
import os
import sys
import zipfile

from fractalsig.pipeline.build_p0_local import CLIP_TO, MIN_TOKENS

ZIP_PATH = _paths.GAGLE_ZIP_STR
MEMBER_PATTERN = "gagle3/*/*.jsonl"
DATA_DIR = os.path.join(_paths.WORKSPACE_STR, "data")
OUT_PATH = os.path.join(DATA_DIR, "p1_short_window_flags.jsonl")


def used_short_window(raw_len):
    """True iff this array hit slice_scores's no-warmup-drop fallback
    branch (raw_len < CLIP_TO), given it was long enough to be kept at all
    (raw_len >= MIN_TOKENS, mirroring slice_scores exactly)."""
    if raw_len is None or raw_len < MIN_TOKENS:
        return None  # not represented in the feature store at all for this scorer
    return raw_len < CLIP_TO


def main():
    human_ids_seen = set()  # mirrors build_p1_feature_store_cpu.py's HUMAN_IDS_SEEN_GLOBAL:
    # keep only the FIRST (sorted-member-order) occurrence per human id, so
    # the flag written here matches the SAME realization the feature store
    # itself kept -- not an independent, possibly-different selection.
    # (Finding 1 established that different prompt_type members can carry
    # different raw array lengths for the "same" human id, so this ordering
    # match matters, not just a formality.)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        members = sorted(m for m in zf.namelist() if fnmatch.fnmatch(m, MEMBER_PATTERN))
        print(f"[info] {len(members)} members")
        n_written = 0
        with open(OUT_PATH, "w") as fout:
            for i, member in enumerate(members, 1):
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
                        domain = entry.get("dataset_name")
                        scores_block = entry.get("scores", {})

                        if entry.get("llm_text"):
                            gemma_raw = scores_block.get("gemma-2b", {}).get("llm_text")
                            mistral_raw = scores_block.get("mistral-7b", {}).get("llm_text")
                            fout.write(json.dumps({
                                "id": str(entry.get("_id")),
                                "Gemma_Used_Short_Window": used_short_window(len(gemma_raw) if gemma_raw else None),
                                "Mistral_Used_Short_Window": used_short_window(len(mistral_raw) if mistral_raw else None),
                            }) + "\n")
                            n_written += 1

                        if entry.get("original_text"):
                            human_id = f"{split}_{domain}_{entry.get('dataset_id')}_human"
                            if human_id not in human_ids_seen:
                                human_ids_seen.add(human_id)
                                gemma_raw = scores_block.get("gemma-2b", {}).get("original_text")
                                mistral_raw = scores_block.get("mistral-7b", {}).get("original_text")
                                fout.write(json.dumps({
                                    "id": human_id,
                                    "Gemma_Used_Short_Window": used_short_window(len(gemma_raw) if gemma_raw else None),
                                    "Mistral_Used_Short_Window": used_short_window(len(mistral_raw) if mistral_raw else None),
                                }) + "\n")
                                n_written += 1
                if i % 100 == 0:
                    print(f"[info] {i}/{len(members)} members, {n_written} rows written so far")
        print(f"[done] wrote {n_written} rows to {OUT_PATH}")
        print(f"[note] one row per id, matching the SAME realization build_p1_feature_store_cpu.py "
              f"kept (first-in-sorted-member-order for Human ids). Should now join 1:1 against "
              f"data/p1_feature_store_cpu.jsonl by id.")


if __name__ == "__main__":
    main()
