#!/usr/bin/env python3
"""P5 track A, feature-store build. Reuses `build_feature_row` from
`p4_task5_tabular_features.py` AS-IS -- it already assembles Family A
(fractal, via `features_for`), Family B (stat, via `family_b_feature_summary`),
Family C (style, via `family_c_style`), and embedding dispersion from
exactly the `{scores, ranks, entropies, window}` schema `p5_raid_rescore.py`
produces (same schema `p4_rescore_edits.py`/`p4_b2_ladder_rescore.py` use).
No new feature code, per this project's standing reuse-not-restate
discipline.

Complete-case only (F-5(a) precedent): a row is scored only if BOTH scorers
have a real rescore record (not `skipped_below_min_tokens`) -- carrying a
NaN-riddled row into the RAID feature store would reproduce exactly the
missingness-as-signal confound F-6/F-5 exist to catch, on a fresh dataset
where nobody has looked for it yet.

Shardable (--shard-index/--num-shards), matching P4 task5's own 16-way
parallel pattern for this exact CPU-bound step -- **use setsid, not bare
nohup, for any parallel launch** (standing lesson, see ledger F-... on the
9-of-16-shards-silently-killed incident).
"""
from fractalsig import paths as _paths  # noqa: E402
import argparse
import json
import os
import sys

import numpy as np

HERE = _paths.WORKSPACE_STR
D = lambda *p: os.path.join(HERE, "data", *p)

from fractalsig.robustness.p4_task5_tabular_features import build_feature_row  # noqa: E402
from fractalsig.pipeline.build_p1_feature_store_gpu import load_models  # noqa: E402


def load_rescore_filtered(scorer, wanted_ids):
    """Stream the rescore file and keep only rows this shard actually needs.
    A sharded process only uses ~1/num_shards of a 120,005-row rescore file
    (~2.3GB loaded whole) -- with 16 processes each loading BOTH files in
    full, peak memory blew past the 32GiB memory cap
    (/sys/fs/cgroup/memory.max), causing repeated silent SIGKILLs with no
    traceback (found 2026-08-28, after two rounds of shards dying with no
    error). Filtering while streaming cuts each process's rescore memory to
    roughly 1/num_shards of the unfiltered size."""
    out = {}
    for l in open(D(f"p5_raid_rescore_{scorer}.jsonl")):
        r = json.loads(l)
        if r["id"] in wanted_ids:
            out[r["id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    args = ap.parse_args()

    print("[p5-feat] loading embedder...", flush=True)
    _, device, _, _, _, embedder = load_models(scorers=())
    print(f"[p5-feat] embedder loaded on device={device}", flush=True)

    sample = [json.loads(l) for l in open(D("p5_raid_sample.jsonl"))]
    if args.limit:
        sample = sample[:args.limit]
    out_path = D("p5_raid_features.jsonl")
    if args.num_shards:
        sample = sample[args.shard_index::args.num_shards]
        out_path = D(f"p5_raid_features.shard{args.shard_index}.jsonl")
    print(f"[p5-feat] {len(sample)} rows to featurize -> {out_path}", flush=True)

    wanted_ids = {row["id"] for row in sample}
    print(f"[p5-feat] loading rescore files (filtered to {len(wanted_ids)} ids)...", flush=True)
    gemma_rescore = load_rescore_filtered("gemma", wanted_ids)
    mistral_rescore = load_rescore_filtered("mistral", wanted_ids)
    print(f"[p5-feat] {len(gemma_rescore)} gemma, {len(mistral_rescore)} mistral rescores loaded", flush=True)

    out = open(out_path, "w")
    n_ok, n_missing_rescore, n_err = 0, 0, 0
    for i, row in enumerate(sample):
        grec = gemma_rescore.get(row["id"])
        mrec = mistral_rescore.get(row["id"])
        if grec is None or mrec is None or grec.get("skipped_below_min_tokens") or mrec.get("skipped_below_min_tokens"):
            n_missing_rescore += 1
            continue
        try:
            feats = build_feature_row(row["text"], grec, mrec, embedder, device)
            if any(v != v for v in feats.values() if isinstance(v, float)):  # any NaN
                # keep the row but let the exit gate / downstream consumer decide;
                # NaN in an individual feature (e.g. skew needing n>2) is expected
                # for short docs, not the same as a missing rescore entirely.
                pass
            out.write(json.dumps({"id": row["id"], "Label": row["Label"], "domain": row["domain"],
                                    "attack": row["attack"], **feats}) + "\n")
            n_ok += 1
        except Exception as e:
            n_err += 1
            if n_err <= 5:
                print(f"[p5-feat] ERROR row {row['id']}: {e}", flush=True)
        if (i + 1) % 2000 == 0:
            out.flush()
            print(f"[p5-feat]   {i+1}/{len(sample)}  ok={n_ok} missing_rescore={n_missing_rescore} err={n_err}", flush=True)

    out.close()
    print(f"[p5-feat] DONE  ok={n_ok} missing_rescore={n_missing_rescore} err={n_err}  -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
