#!/usr/bin/env python3
"""
Driver for T0 / T1: runs ONE split construction over the full GAGLE corpus
(721 shards, ~223,834 documents x 2 scorers) and writes:
  - outputs/stats/t0_t1_<construction>_features.parquet
      long-format per-document-per-scorer-per-feature half values, with
      stratum keys and n_tokens (T0.1 spec).
  - outputs/stats/t0_t1_<construction>_stats.json
      pooled + within-stratum r/rho/spearman per feature (T0.2 Route 1 for
      every construction; T0.3-equivalent outputs).

Uses multiprocessing (fork) so worker processes share the corpus arrays
loaded once in the parent via copy-on-write, rather than re-reading parquet
or re-pickling ~1.5GB of score arrays per worker.

Usage:
  python run_construction.py interleave
  python run_construction.py block_8
  python run_construction.py matched_sweep
  python -m experiments.run_construction contiguous --workers 8
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fractalsig import paths as _paths  # noqa: E402
from fractalsig.runtime import default_workers  # noqa: E402
from experiments.reliability_lib import (  # noqa: E402
    load_corpus, half_features, CONSTRUCTIONS, SCORERS, WINDOW_SCALE,
    pooled_and_within_stats, fisher_z_ci, cluster_robust_se_pearson_r,
)

OUT_DIR = _paths.STATS_DIR_STR
LOG_DIR = _paths.LOG_DIR_STR

# Populated in the parent process before Pool creation; inherited by forked
# workers via copy-on-write (NOT re-sent per task).
_G = {}


def _worker_init():
    pass  # globals already inherited via fork; nothing to do


def _process_range(args):
    start, end, construction, seed, window_scale, skip_windowed = args
    splitter = CONSTRUCTIONS["interleave"] if construction == "matched_sweep" else CONSTRUCTIONS[construction]
    needs_rng = construction in ("permutation", "block_bootstrap")
    rng = np.random.default_rng(seed) if needs_rng else None

    recs = _G["recs"]
    rows = []  # (id, scorer, Label, domain, temp, prompt_type, n_tokens, fa_dict, fb_dict)
    for i in range(start, end):
        for sname, _ in SCORERS:
            arr = recs[sname][i]
            if arr is None or len(arr) < 8:
                continue
            arr = np.asarray(arr, dtype=np.float64)
            try:
                a, b = splitter(arr, rng) if needs_rng else splitter(arr)
            except Exception:
                continue
            if len(a) < 8 or len(b) < 8:
                continue
            try:
                fa = half_features(a, window_scale=window_scale, skip_windowed=skip_windowed)
                fb = half_features(b, window_scale=window_scale, skip_windowed=skip_windowed)
            except Exception:
                continue
            rows.append((recs["id"][i], sname, recs["Label"][i], recs["domain"][i],
                         recs["temp"][i], recs["prompt_type"][i], len(arr), fa, fb))
    return rows


def run(construction, workers, out_prefix, seed=20260908, window_scale=None,
        skip_windowed=False, save_long_parquet=True):
    t0 = time.time()
    print(f"[run] construction={construction} workers={workers} "
          f"window_scale={window_scale} skip_windowed={skip_windowed}", flush=True)

    if "recs" not in _G:
        _G["recs"] = load_corpus()
    n_docs = len(_G["recs"]["id"])

    chunk = 2000
    ranges = [(s, min(s + chunk, n_docs)) for s in range(0, n_docs, chunk)]
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(len(ranges))
    tasks = [(s, e, construction, int(cs.generate_state(1)[0]), window_scale, skip_windowed)
             for (s, e), cs in zip(ranges, child_seeds)]

    all_rows = []
    with mp.get_context("fork").Pool(processes=workers, initializer=_worker_init) as pool:
        for i, rows in enumerate(pool.imap_unordered(_process_range, tasks), 1):
            all_rows.extend(rows)
            if i % 20 == 0:
                print(f"[run]   {i}/{len(tasks)} chunks, {len(all_rows):,} rows so far, "
                      f"{time.time()-t0:.0f}s", flush=True)
    print(f"[run] feature computation done: {len(all_rows):,} (doc,scorer) rows in "
          f"{time.time()-t0:.0f}s", flush=True)

    # ---- reduce into per-feature arrays ----
    feat_names = sorted(all_rows[0][7].keys()) if all_rows else []
    per_feat = {f"{sname}_{feat}": {"a": [], "b": [], "strata": [], "ids": [], "n_tokens": []}
                for sname in ("gemma", "mistral") for feat in feat_names}
    long_records = {"id": [], "scorer": [], "feature": [], "half_a": [], "half_b": [],
                     "Label": [], "domain": [], "temp": [], "prompt_type": [], "n_tokens": []}

    for (doc_id, sname, label, domain, temp, ptype, ntok, fa, fb) in all_rows:
        stratum = (label, domain, temp)
        for feat in feat_names:
            key = f"{sname}_{feat}"
            va, vb = fa[feat], fb[feat]
            if va == va and vb == vb:  # not NaN
                per_feat[key]["a"].append(va)
                per_feat[key]["b"].append(vb)
                per_feat[key]["strata"].append(stratum)
                per_feat[key]["ids"].append(doc_id)
                per_feat[key]["n_tokens"].append(ntok)
                if save_long_parquet:
                    long_records["id"].append(doc_id)
                    long_records["scorer"].append(sname)
                    long_records["feature"].append(feat)
                    long_records["half_a"].append(va)
                    long_records["half_b"].append(vb)
                    long_records["Label"].append(label)
                    long_records["domain"].append(domain)
                    long_records["temp"].append(temp)
                    long_records["prompt_type"].append(ptype)
                    long_records["n_tokens"].append(ntok)

    # ---- stats per feature ----
    stats_out = {}
    for key, d in per_feat.items():
        s = pooled_and_within_stats(d["a"], d["b"], d["strata"])
        if s is None:
            continue
        ci_lo, ci_hi = fisher_z_ci(s["rho_pooled"], s["n"])
        s["fisher_z_ci_pooled"] = [ci_lo, ci_hi]
        stats_out[key] = s

    os.makedirs(OUT_DIR, exist_ok=True)
    stats_path = os.path.join(OUT_DIR, f"{out_prefix}_stats.json")
    with open(stats_path, "w") as f:
        json.dump({"construction": construction, "n_docs_loaded": n_docs,
                    "n_doc_scorer_pairs": len(all_rows), "features": stats_out,
                    "window_scale_used": window_scale or "WINDOW_SCALE[:4] (default)",
                    "runtime_sec": time.time() - t0}, f, indent=2)
    print(f"[written] {stats_path}", flush=True)

    if save_long_parquet and long_records["id"]:
        table = pa.table(long_records)
        pq_path = os.path.join(OUT_DIR, f"{out_prefix}_features.parquet")
        pq.write_table(table, pq_path)
        print(f"[written] {pq_path} ({table.num_rows:,} rows)", flush=True)

    print(f"[run] TOTAL {time.time()-t0:.0f}s", flush=True)
    return stats_out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("construction", choices=list(CONSTRUCTIONS.keys()) + ["matched_sweep"])
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--no-long-parquet", action="store_true")
    args = ap.parse_args()

    window_scale = WINDOW_SCALE if args.construction == "matched_sweep" else None
    out_prefix = args.out_prefix or f"t0_t1_{args.construction}"
    run(args.construction, args.workers, out_prefix, seed=args.seed,
        window_scale=window_scale, save_long_parquet=not args.no_long_parquet)
