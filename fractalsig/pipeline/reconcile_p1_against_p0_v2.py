#!/usr/bin/env python3
"""
P1 -> P0 reconciliation, v2 (2026-08-23). Supersedes
fractalsig/pipeline/reconcile_p1_against_p0.py for post-X-01 / post-F-4 data.

WHY A V2 RATHER THAN AN EDIT
  The original re-derived P0's document selection inline, with the human key
      human_id = f"{split}_{domain}_{dataset_id}_human"
  That is the PRE-X-01 key. D-07 removed `split` from the human identity, and
  the original's copy was never updated -- its docstring still claims the key
  is "copied verbatim" from build_p0_local.py, which stopped being true the
  moment X-01 landed. Running it unmodified would pool ~11,696 human docs
  against a 5,000-doc P0 ground truth and report a mismatch that looks like an
  extraction bug but is really a stale copy. The original is left in place,
  unmodified, as the record of the old procedure (D-10).

WHAT THIS VERSION TESTS (and why it is not circular)
  Re-deriving P0's selection here would only ever compare P0 to itself. So the
  population is taken from **P1's own output** -- the ids in
  p1_feature_store_cpu.jsonl -- and those documents' raw trajectories are
  pooled with P0's method (calc_S list-mode / pooled increments for H), then
  diffed against P0's stored numbers.

  That makes the test: DOES P1'S DOCUMENT POPULATION REPRODUCE P0'S
  CORPUS-LEVEL NUMBERS? X-01 was applied at two independent sites
  (build_p0_local.py, job 73; build_p1_feature_store_cpu.py, job 84). If those
  two sites ever selected different documents, nothing else in the pipeline
  would notice -- the join only checks CPU-vs-GPU, never CPU-vs-P0. This does.

  Raw arrays come from gpu_manifest_shards/ (written BY the P1 CPU pass, so
  they are P1's population by construction) rather than from a fresh zip read,
  which would reintroduce a selection step and cost ~10 GB of I/O.

TWO PROCEDURES FOR TWO STATISTICS -- deliberate, inherited from v1
  H: reconciled BOTH ways -- pooled increments (P0's method) and the average of
     P1's own per-doc Gemma_H. Per-doc H is trustworthy, so the average is a
     genuinely independent check on P1's own H column.
  S: pooled trajectories ONLY. Per-doc S is severely biased
     (Holder_PerDoc_Unreliable), so averaging per-doc S is a DIFFERENT
     estimator than P0's pooled calc_S and would show a large expected gap
     having nothing to do with extraction correctness.

Gemma-2b only: P0 never scored mistral-7b, so there is no ground truth for it.

USAGE
    python -m fractalsig.pipeline.reconcile_p1_against_p0_v2
"""
from fractalsig import paths as _paths  # noqa: E402
import collections
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

HERE = _paths.WORKSPACE_STR
from fractalsig.features.fractal_math import generate_trajectory, calc_H, calc_S  # noqa: E402

DATA = os.path.join(HERE, "data")
SHARDS = os.environ.get("GAGLE_MANIFEST_DIR", _paths.MANIFEST_SHARD_DIR_STR)
CPU_FS = os.path.join(DATA, "p1_feature_store_cpu.jsonl")
P0_SH = os.path.join(DATA, "p0_corpus_level_SH.json")
OUT = os.path.join(DATA, "p1_p0_reconciliation_v2.json")
MIN_DOCS = 10


def p1_population():
    """(domain, Label) -> set of ids, plus P1's own per-doc Gemma_H by group.
    Only rows where the gemma scorer actually produced Family A."""
    ids = collections.defaultdict(set)
    hsum = collections.defaultdict(list)
    with open(CPU_FS) as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            if not (d.get("Gemma_available") is True or d.get("Gemma_available") == 1):
                continue
            k = (d["domain"], d["Label"])
            ids[k].add(d["id"])
            h = d.get("Gemma_H")
            if h is not None and h == h:
                hsum[k].append(float(h))
    return ids, hsum


def pool_from_shards(wanted):
    """id -> gemma sliced score array, for every id P1 kept."""
    flat = {i for s in wanted.values() for i in s}
    got = {}
    files = sorted(glob.glob(os.path.join(SHARDS, "*.parquet")))
    print(f"[info] scanning {len(files)} shards for {len(flat):,} ids "
          f"(no zip re-read)...", flush=True)
    for n, f in enumerate(files, 1):
        t = pq.read_table(f, columns=["id", "gagle_gemma_sliced_scores"])
        for r in t.to_pylist():
            if r["id"] in flat and r["gagle_gemma_sliced_scores"]:
                got[r["id"]] = r["gagle_gemma_sliced_scores"]
        if n % 200 == 0:
            print(f"[info]   {n}/{len(files)} shards, {len(got):,} arrays", flush=True)
    return got


def main():
    for p in (CPU_FS, P0_SH):
        if not os.path.exists(p):
            sys.exit(f"[abort] missing {p}")
    p0 = json.load(open(P0_SH))
    ids, hsum = p1_population()
    print(f"[info] P1 population: {sum(len(v) for v in ids.values()):,} gemma-available "
          f"rows across {len(ids)} (domain,label) cells")
    arrays = pool_from_shards(ids)
    print(f"[info] located {len(arrays):,} score arrays\n")

    rows = []
    for k in sorted(ids):
        domain, label = k
        got = [np.asarray(arrays[i], dtype=np.float64) for i in ids[k] if i in arrays]
        if len(got) < MIN_DOCS:
            continue
        trajs, incrs = [], []
        for a in got:
            t, inc = generate_trajectory(a)
            trajs.append(t)
            incrs.append(inc)
        S, S_r2 = calc_S(trajs)
        H, H_r2 = calc_H(np.concatenate(incrs))
        key = f"{domain}|{label}"
        ref = p0.get(key, {})
        rows.append({
            "cell": key, "n_p1": len(got), "n_p0": ref.get("n_docs"),
            "S_p1": float(S), "S_p0": ref.get("S"),
            "H_p1": float(H), "H_p0": ref.get("H"),
            "H_avg_perdoc_p1": float(np.mean(hsum[k])) if hsum[k] else None,
            "S_R2_p1": float(S_r2), "H_R2_p1": float(H_r2),
        })

    print("=" * 100)
    print("P1 -> P0 RECONCILIATION  (gemma-2b; S/H pooled by P0's method over P1's population)")
    print("=" * 100)
    print(f"{'cell':<30}{'n P1':>8}{'n P0':>8}   "
          f"{'S P1':>8}{'S P0':>8}{'dS':>9}   {'H P1':>8}{'H P0':>8}{'dH':>9}   {'H avg/doc':>10}")
    print("-" * 100)
    worst_s = worst_h = 0.0
    for r in rows:
        ds = (r["S_p1"] - r["S_p0"]) if r["S_p0"] is not None else float("nan")
        dh = (r["H_p1"] - r["H_p0"]) if r["H_p0"] is not None else float("nan")
        if ds == ds:
            worst_s = max(worst_s, abs(ds))
        if dh == dh:
            worst_h = max(worst_h, abs(dh))
        nm = "" if r["n_p0"] == r["n_p1"] else "  <-- n MISMATCH"
        print(f"{r['cell']:<30}{r['n_p1']:>8,}{(r['n_p0'] or 0):>8,}   "
              f"{r['S_p1']:>8.4f}{(r['S_p0'] or float('nan')):>8.4f}{ds:>+9.4f}   "
              f"{r['H_p1']:>8.4f}{(r['H_p0'] or float('nan')):>8.4f}{dh:>+9.4f}   "
              f"{(r['H_avg_perdoc_p1'] or float('nan')):>10.4f}{nm}")
    print("-" * 100)
    print(f"  max |dS| = {worst_s:.5f}     max |dH| = {worst_h:.5f}")
    mism = [r["cell"] for r in rows if r["n_p0"] != r["n_p1"]]
    print(f"  cells where P1's doc count != P0's: {len(mism)}"
          + (f"  -> {mism}" if mism else "  (populations agree exactly)"))
    print("\n  Note: 'H avg/doc' is the mean of P1's OWN per-doc Gemma_H column -- an")
    print("  independent check on P1's H computation. It is NOT expected to equal the")
    print("  pooled H: averaging per-doc estimates is a different estimator.")
    print("  By design, deltas are reported without a hard pass bar.")

    json.dump({"cells": rows, "max_abs_dS": worst_s, "max_abs_dH": worst_h,
               "n_count_mismatches": len(mism)}, open(OUT, "w"), indent=2)
    print(f"\n[written] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
