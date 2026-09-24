#!/usr/bin/env python3
"""Backfill `ZWSP_Present_In_Source_Text` into the already-built RAID
feature store (`data/p5_raid_features.jsonl`, 68,212 rows), without a
pipeline rerun -- follows `fractalsig/pipeline/backfill_short_window_flag.py`'s exact
pattern: stream the raw corpus once, read only what's needed (id + text),
write a small id-keyed join file, then patch it into a NEW copy of the
feature store (never rewrite the existing output in place).

`family_c_style()` (fractalsig/pipeline/build_p1_feature_store_cpu.py, fixed 2026-08-29)
now emits this flag on every FRESH build, but data/p5_raid_features.jsonl
was built before that fix landed -- this script computes the exact same
value (`_ZERO_WIDTH_SPACE_RE.search(text) is not None`, imported not
restated) against the same raw text, for id-exact parity with a fresh
build.
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
import sys

HERE = _paths.WORKSPACE_STR
D = lambda *p: os.path.join(HERE, "data", *p)

from fractalsig.pipeline.build_p1_feature_store_cpu import _ZERO_WIDTH_SPACE_RE  # noqa: E402


def main():
    wanted_ids = set()
    for l in open(D("p5_raid_features.jsonl")):
        wanted_ids.add(json.loads(l)["id"])
    print(f"[backfill-zwsp] {len(wanted_ids)} ids to backfill", flush=True)

    flags = {}
    n_read = 0
    with open(D("raid_ingested.jsonl")) as f:
        for line in f:
            n_read += 1
            if n_read % 1_000_000 == 0:
                print(f"[backfill-zwsp]   scanned {n_read:,} raw rows, "
                      f"{len(flags)}/{len(wanted_ids)} matched so far", flush=True)
            row = json.loads(line)
            rid = row["id"]
            if rid in wanted_ids:
                flags[rid] = _ZERO_WIDTH_SPACE_RE.search(row["text"]) is not None

    print(f"[backfill-zwsp] scanned {n_read:,} raw rows total, "
          f"{len(flags)}/{len(wanted_ids)} matched", flush=True)

    missing = wanted_ids - set(flags)
    extra = set(flags) - wanted_ids
    if missing or extra:
        print(f"[backfill-zwsp] FATAL: id-set mismatch -- missing={len(missing)} "
              f"extra={len(extra)}", flush=True)
        if missing:
            print(f"[backfill-zwsp]   sample missing: {list(missing)[:5]}", flush=True)
        sys.exit(1)
    print("[backfill-zwsp] id-set exact match confirmed: 0 missing, 0 extra", flush=True)

    with open(D("p5_zwsp_flags_raid.jsonl"), "w") as out:
        for rid, present in flags.items():
            out.write(json.dumps({"id": rid, "ZWSP_Present_In_Source_Text": present}) + "\n")
    print(f"[backfill-zwsp] written to data/p5_zwsp_flags_raid.jsonl "
          f"({sum(flags.values())} True, {len(flags) - sum(flags.values())} False)", flush=True)

    # Patch into a NEW copy of the feature store -- never rewrite in place.
    n_patched = 0
    with open(D("p5_raid_features.jsonl")) as fin, \
         open(D("p5_raid_features_v2.jsonl"), "w") as fout:
        for l in fin:
            row = json.loads(l)
            row["ZWSP_Present_In_Source_Text"] = flags[row["id"]]
            fout.write(json.dumps(row) + "\n")
            n_patched += 1
    print(f"[backfill-zwsp] patched {n_patched} rows -> data/p5_raid_features_v2.jsonl", flush=True)


if __name__ == "__main__":
    main()
