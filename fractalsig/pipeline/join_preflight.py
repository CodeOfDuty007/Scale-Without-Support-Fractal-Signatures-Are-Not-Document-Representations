#!/usr/bin/env python3
"""
Cheap pre-flight for build_p1_join_feature_store.py's id-set gate.

WHY THIS EXISTS
  The join already refuses to write a partial feature store: it asserts
  set(cpu_ids) == set(gpu_ids) and aborts with the missing/extra ids if they
  differ. That guards the catastrophic failure mode. What it does NOT do is
  tell you WHY they differ -- and after X-01 they certainly will, because the
  CPU pass was rebuilt under the D-07 dedup key (human 11,696 -> 5,000) while
  the self-scored GPU store still carries the pre-X-01 human population.

  So this script answers "which rows, and whose fault" BEFORE the real join is
  attempted, and specifically ATTRIBUTES the symmetric difference:

    - human rows that X-01 legitimately removed  (expected, benign)
    - human rows X-01 introduced under new ids   (expected, need GPU rescoring)
    - LLM-side differences                        (NOT expected -- a real bug:
                                                   a skipped shard, or a
                                                   MIN_TOKENS drop that fired
                                                   differently between passes)

  Run it before the join, not instead of it.

READS ID COLUMNS ONLY
  Both stores are hundreds of MB. Only `id` and `Label` are parsed per line.

USAGE
  python -m fractalsig.pipeline.join_preflight
  python -m fractalsig.pipeline.join_preflight --gpu data/p1_feature_store_gpu_selfscored.jsonl
"""
from fractalsig import paths as _paths  # noqa: E402
import argparse
import collections
import json
import os
import sys

HERE = _paths.WORKSPACE_STR
DATA = os.path.join(HERE, "data")

DEFAULT_CPU = os.path.join(DATA, "p1_feature_store_cpu.jsonl")
# D-01: the SELF-SCORED store is the production Family B path. The join script
# still hardcodes GPU_PATH = data/p1_feature_store_gpu.jsonl, which is the
# alignment-GATED output (D-06: audit evidence, not production) and does not
# exist on this box. Defaulting here to the self-scored file so the pre-flight
# reflects what will actually be joined.
DEFAULT_GPU = os.path.join(DATA, "p1_feature_store_gpu_selfscored.jsonl")
PRE_X01_CPU = os.path.join(DATA, "pre_x01_2026-08-23", "p1_feature_store_cpu.jsonl")


def load_ids(path, label_of=None):
    """-> (set of ids, {id: Label}). Parses two fields per line, not full rows."""
    ids, labels = set(), {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            i = d["id"]
            ids.add(i)
            labels[i] = d.get("Label")
    return ids, labels


def summarize(name, ids, labels):
    c = collections.Counter(labels[i] for i in ids)
    print(f"  {name:<34} {len(ids):>8,}   " +
          "  ".join(f"{k}={v:,}" for k, v in sorted(c.items())))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", default=DEFAULT_CPU)
    ap.add_argument("--gpu", default=DEFAULT_GPU)
    ap.add_argument("--pre-x01", default=PRE_X01_CPU,
                    help="pre-X-01 CPU store, used to attribute the difference")
    ap.add_argument("--show", type=int, default=5, help="sample ids to print")
    args = ap.parse_args()

    for p in (args.cpu, args.gpu):
        if not os.path.exists(p):
            print(f"[abort] not found: {p}")
            return 2

    print("=" * 78)
    print("JOIN PRE-FLIGHT -- id-set reconciliation")
    print("=" * 78)
    print(f"  {'store':<34} {'rows':>8}   composition")
    cpu_ids, cpu_lab = load_ids(args.cpu)
    summarize("CPU (Family A+C, post-X-01)", cpu_ids, cpu_lab)
    gpu_ids, gpu_lab = load_ids(args.gpu)
    summarize("GPU (Family B, self-scored)", gpu_ids, gpu_lab)

    only_cpu = cpu_ids - gpu_ids
    only_gpu = gpu_ids - cpu_ids
    print(f"\n  intersection            : {len(cpu_ids & gpu_ids):,}")
    print(f"  CPU-only (missing Family B): {len(only_cpu):,}")
    print(f"  GPU-only (orphaned)        : {len(only_gpu):,}")
    print(f"  symmetric difference       : {len(only_cpu) + len(only_gpu):,}")

    if not only_cpu and not only_gpu:
        print("\n[PASS] id sets match. The join's own assert will pass.")
        return 0

    # ---- attribution -------------------------------------------------------
    print("\n" + "-" * 78)
    print("ATTRIBUTION")
    print("-" * 78)

    cl = collections.Counter(cpu_lab[i] for i in only_cpu)
    gl = collections.Counter(gpu_lab[i] for i in only_gpu)
    print(f"  CPU-only by Label: {dict(cl) or '{}'}")
    print(f"  GPU-only by Label: {dict(gl) or '{}'}")

    cpu_llm_only = {i for i in only_cpu if cpu_lab[i] != "Human"}
    gpu_llm_only = {i for i in only_gpu if gpu_lab[i] != "Human"}

    if os.path.exists(args.pre_x01):
        pre_ids, pre_lab = load_ids(args.pre_x01)
        pre_human = {i for i in pre_ids if pre_lab[i] == "Human"}
        now_human = {i for i in cpu_ids if cpu_lab[i] == "Human"}
        print(f"\n  pre-X-01 human ids : {len(pre_human):,}")
        print(f"  post-X-01 human ids: {len(now_human):,}")
        print(f"    retired by X-01 (in pre, not post): {len(pre_human - now_human):,}")
        print(f"    new under X-01   (in post, not pre): {len(now_human - pre_human):,}")
        print(f"    carried through  (in both)         : {len(pre_human & now_human):,}")
        explained = len({i for i in only_gpu if gpu_lab[i] == "Human"} & (pre_human - now_human))
        print(f"\n  GPU-only human ids explained by X-01 retirement: "
              f"{explained:,} / {gl.get('Human', 0):,}")
    else:
        print(f"\n  [warn] pre-X-01 store not at {args.pre_x01}; cannot attribute.")

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    rc = 0
    if cpu_llm_only or gpu_llm_only:
        print(f"  [FAIL] {len(cpu_llm_only):,} CPU-only + {len(gpu_llm_only):,} GPU-only "
              "rows are NOT human.")
        print("         X-01 touches only the human side, so an LLM-side difference is a")
        print("         REAL defect -- a skipped shard, or MIN_TOKENS firing differently")
        print("         between the two passes. Investigate before joining.")
        for i in list(cpu_llm_only)[:args.show]:
            print(f"           cpu-only: {i}  ({cpu_lab[i]})")
        for i in list(gpu_llm_only)[:args.show]:
            print(f"           gpu-only: {i}  ({gpu_lab[i]})")
        rc = 1
    else:
        print("  [OK] every differing id is human-side -- consistent with X-01 and")
        print("       nothing else. The LLM populations agree exactly.")

    n_need = len({i for i in only_cpu if cpu_lab[i] == "Human"})
    if n_need:
        print(f"\n  ACTION: {n_need:,} post-X-01 human rows have no Family B features yet.")
        print("          Re-score just those through the self-scored GPU path")
        print("          (gpu_manifest_shards/human_dedup_x01.parquet) and merge, then")
        print("          re-run this pre-flight. Do NOT re-run the full 720-shard pass.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
