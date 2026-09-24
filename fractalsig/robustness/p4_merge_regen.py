#!/usr/bin/env python3
"""Merge a completed regeneration pass into the canonical batch file
(2026-08-26). Per the merge-semantics decision (internal review): the
`_regen.jsonl` file is authoritative for the 4 regenerated types
(paraphrase_light, paraphrase_heavy, style_transfer, summarize_then_expand);
the original `p4_edits_batchN.jsonl` remains authoritative for the 3
untouched types (human_revision, sentence_reorder, typo_noise). This script
merges them into ONE canonical file (overwrites `p4_edits_batchN.jsonl` in
place) so nothing downstream can accidentally read the pre-fix data or mix
the two -- the same discipline `build_p1_join_feature_store.py` uses
(assert id-set equality before writing anything, never a silent partial
write).
"""
from fractalsig import paths as _paths  # noqa: E402
import argparse
import json
import os

HERE = _paths.WORKSPACE_STR
REGEN_TYPES = ["paraphrase_light", "paraphrase_heavy", "style_transfer", "summarize_then_expand"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, required=True, choices=[1, 2])
    args = ap.parse_args()

    orig_path = os.path.join(HERE, "data", f"p4_edits_batch{args.batch}.jsonl")
    regen_path = os.path.join(HERE, "data", f"p4_edits_batch{args.batch}_regen.jsonl")

    orig = {r["id"]: r for r in (json.loads(l) for l in open(orig_path))}
    regen = {r["id"]: r for r in (json.loads(l) for l in open(regen_path))}

    if len(orig) != len(regen):
        raise SystemExit(f"[FATAL] record count mismatch: orig={len(orig)} regen={len(regen)}")
    if set(orig) != set(regen):
        missing = set(orig) - set(regen)
        extra = set(regen) - set(orig)
        raise SystemExit(f"[FATAL] id-set mismatch: {len(missing)} missing from regen, "
                          f"{len(extra)} extra in regen (e.g. {list(missing)[:3] or list(extra)[:3]})")

    n_not_flagged = 0
    for doc_id, r in regen.items():
        for etype in REGEN_TYPES:
            if not r["variants"][etype].get("regenerated"):
                n_not_flagged += 1
    if n_not_flagged:
        raise SystemExit(f"[FATAL] {n_not_flagged} regenerated variant records missing "
                          f"regenerated=true -- refusing to merge an inconsistent file")

    merged = []
    for doc_id in orig:
        o, r = orig[doc_id], regen[doc_id]
        variants = dict(o["variants"])
        for etype in REGEN_TYPES:
            variants[etype] = r["variants"][etype]
        merged_row = dict(o)
        merged_row["variants"] = variants
        merged.append(merged_row)

    backup_path = orig_path + ".pre_regen_merge.bak"
    os.rename(orig_path, backup_path)
    with open(orig_path, "w") as f:
        for row in merged:
            f.write(json.dumps(row) + "\n")

    print(f"[merge] batch {args.batch}: {len(merged)} records merged", flush=True)
    print(f"[merge] pre-merge original backed up to {backup_path}", flush=True)
    print(f"[merge] canonical file rewritten: {orig_path}", flush=True)
    print(f"[merge] regenerated types: {REGEN_TYPES}", flush=True)
    print(f"[merge] unchanged types: human_revision, sentence_reorder, typo_noise", flush=True)


if __name__ == "__main__":
    main()
