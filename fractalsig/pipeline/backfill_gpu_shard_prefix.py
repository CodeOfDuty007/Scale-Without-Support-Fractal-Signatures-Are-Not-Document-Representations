#!/usr/bin/env python3
"""
Backfills a `prefix` field, keyed by id, for the 720 GPU manifest shards
already written to $FRACTALSIG_WORKSPACE/gpu_manifest_shards/ before
build_p1_feature_store_cpu.py's extract_docs/gpu_row were edited to carry
`prefix` (see the internal verification notes --
the Family B GPU scorer needs `prefix` to reproduce GAGLE's own scoring for
continue_it/continue_pt, and any prompt_type where prefix+text turns out to
be the right candidate during the alignment sweep).

Same LIGHT pass pattern as backfill_short_window_flag.py: reads only `id`
and `prefix` from gagle.zip (no Family A/B/C recompute), joined to the
existing shards by id downstream. Does NOT rewrite the 720 parquet shards
in place -- writes a separate, small id-keyed file, consistent with never
overwriting an already-written pipeline output.

USAGE
    python -m fractalsig.pipeline.backfill_gpu_shard_prefix
"""
from fractalsig import paths as _paths  # noqa: E402
import fnmatch
import json
import os
import sys
import zipfile


ZIP_PATH = _paths.GAGLE_ZIP_STR
MEMBER_PATTERN = "gagle3/*/*.jsonl"
DATA_DIR = os.path.join(_paths.WORKSPACE_STR, "data")
OUT_PATH = os.path.join(DATA_DIR, "p1_gpu_shard_prefix_backfill.jsonl")


def main():
    human_ids_seen = set()  # mirrors build_p1_feature_store_cpu.py's HUMAN_IDS_SEEN_GLOBAL:
    # keep only the FIRST (sorted-member-order) occurrence per human id, so
    # this backfill matches the SAME realization the GPU shards themselves
    # kept -- not an independent, possibly-different selection.
    with zipfile.ZipFile(ZIP_PATH) as zf:
        members = sorted(m for m in zf.namelist() if fnmatch.fnmatch(m, MEMBER_PATTERN))
        print(f"[info] {len(members)} members")
        n_written = 0
        n_missing_prefix = 0
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
                        prefix = entry.get("prefix")
                        if prefix is None:
                            n_missing_prefix += 1

                        if entry.get("llm_text"):
                            fout.write(json.dumps({
                                "id": str(entry.get("_id")),
                                "prefix": prefix,
                            }) + "\n")
                            n_written += 1

                        if entry.get("original_text"):
                            human_id = f"{split}_{domain}_{entry.get('dataset_id')}_human"
                            if human_id not in human_ids_seen:
                                human_ids_seen.add(human_id)
                                fout.write(json.dumps({
                                    "id": human_id,
                                    "prefix": prefix,
                                }) + "\n")
                                n_written += 1
                if i % 100 == 0:
                    print(f"[info] {i}/{len(members)} members, {n_written} rows written so far")
        print(f"[done] wrote {n_written} rows to {OUT_PATH}")
        print(f"[note] {n_missing_prefix} entries had no `prefix` field at all (expected for "
              f"prompt_types where GAGLE doesn't carry one -- not an error).")
        print(f"[note] one row per id, matching the SAME realization build_p1_feature_store_cpu.py "
              f"kept (first-in-sorted-member-order for Human ids). Should join 1:1 against the "
              f"existing GPU manifest shards by id (left join -- shards only contain ids that "
              f"passed MIN_TOKENS on at least one scorer, a subset of what's written here).")


if __name__ == "__main__":
    main()
