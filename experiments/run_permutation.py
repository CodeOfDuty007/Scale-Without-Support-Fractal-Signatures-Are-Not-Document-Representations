#!/usr/bin/env python3
"""
T1 `permutation` construction: random balanced split-halves.

Each replicate assigns token positions to the two halves at random but keeps
chronological order WITHIN each half (indices are sorted before slicing). R/S
and the displacement-probability estimator both operate on an ordered
trajectory, so an unsorted half would be white noise to the estimator and
drive reliability to ~0 regardless of the underlying signal.

The paper reports 20 replicates (the Pronk et al. standard is >= 500); the
reason for the reduction is written into the output JSON (`reduction_reason`).
The output is a DISTRIBUTION of rho_pooled / rho_within per quantity.

Point features only (S, H, S_R2, H_R2; skip_windowed=True): the windowed
sub-features already fail the reliability bar under every deterministic
construction (see t0_t1_*_stats.json). Per-document arrays are not saved per
replicate; only per-replicate summary statistics are kept.

Usage:
    python -m experiments.run_permutation [N_REPS=20] [WORKERS]
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

from fractalsig import paths as _paths  # noqa: E402
from fractalsig.runtime import default_workers  # noqa: E402
from experiments.reliability_lib import (  # noqa: E402
    load_corpus, half_features, split_permutation, SCORERS, pooled_and_within_stats,
)

OUT_DIR = _paths.STATS_DIR_STR
_G = {}


def _process_range(args):
    start, end, seed = args
    rng = np.random.default_rng(seed)
    recs = _G["recs"]
    rows = []
    for i in range(start, end):
        for sname, _ in SCORERS:
            arr = recs[sname][i]
            if arr is None or len(arr) < 8:
                continue
            arr = np.asarray(arr, dtype=np.float64)
            a, b = split_permutation(arr, rng)
            if len(a) < 8 or len(b) < 8:
                continue
            try:
                fa = half_features(a, skip_windowed=True)
                fb = half_features(b, skip_windowed=True)
            except Exception:
                continue
            rows.append((sname, recs["Label"][i], recs["domain"][i], recs["temp"][i], fa, fb))
    return rows


def run_one_rep(pool, rep_idx, base_seed, n_docs, chunk=3000):
    ranges = [(s, min(s + chunk, n_docs)) for s in range(0, n_docs, chunk)]
    ss = np.random.SeedSequence([base_seed, rep_idx])
    child_seeds = ss.spawn(len(ranges))
    tasks = [(s, e, int(cs.generate_state(1)[0])) for (s, e), cs in zip(ranges, child_seeds)]

    all_rows = []
    for rows in pool.imap_unordered(_process_range, tasks):
        all_rows.extend(rows)

    per_feat = {}
    for (sname, label, domain, temp, fa, fb) in all_rows:
        for feat in fa:
            key = f"{sname}_{feat}"
            va, vb = fa[feat], fb[feat]
            if va == va and vb == vb:
                d = per_feat.setdefault(key, {"a": [], "b": [], "strata": []})
                d["a"].append(va)
                d["b"].append(vb)
                d["strata"].append((label, domain, temp))

    rep_stats = {}
    for key, d in per_feat.items():
        s = pooled_and_within_stats(d["a"], d["b"], d["strata"])
        if s is not None:
            rep_stats[key] = s
    return rep_stats


def main():
    n_reps_planned = 500  # Pronk et al. standard: >= 500 random balanced splits
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else default_workers()
    base_seed = 20260908
    reduction_reason = (
        f"Reduced from the planned {n_reps_planned} replicates to {n_reps} for compute cost "
        "(~1.7 min per replicate on 8 cores). Each replicate's rho is computed over "
        "~200,000 documents, so within-replicate sampling error is negligible; the "
        "replicates characterize variability across permutation draws only, for which a "
        f"5th/95th percentile from {n_reps} draws is adequate."
    ) if n_reps < n_reps_planned else None

    t0 = time.time()
    _G["recs"] = load_corpus()
    n_docs = len(_G["recs"]["id"])

    all_reps = []
    with mp.get_context("fork").Pool(processes=workers) as pool:
        for rep in range(n_reps):
            rs = run_one_rep(pool, rep, base_seed, n_docs)
            all_reps.append(rs)
            if (rep + 1) % 10 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (rep + 1) * (n_reps - rep - 1)
                print(f"[permutation] rep {rep+1}/{n_reps} done, {elapsed:.0f}s elapsed, "
                      f"ETA {eta/60:.1f} min", flush=True)
                # checkpoint intermediate results every 10 reps, in case of interruption
                json.dump({"n_reps_so_far": rep + 1, "n_docs": n_docs, "reps": all_reps},
                           open(os.path.join(OUT_DIR, "t1_permutation_CHECKPOINT.json"), "w"))

    keys = sorted({k for rs in all_reps for k in rs})
    summary = {}
    for key in keys:
        rho_pooled = [rs[key]["rho_pooled"] for rs in all_reps if key in rs]
        rho_within = [rs[key]["rho_within"] for rs in all_reps if key in rs and rs[key]["rho_within"] == rs[key]["rho_within"]]
        ns = [rs[key]["n"] for rs in all_reps if key in rs]
        summary[key] = {
            "n_reps": len(rho_pooled),
            "rho_pooled_median": float(np.median(rho_pooled)),
            "rho_pooled_p5": float(np.percentile(rho_pooled, 5)),
            "rho_pooled_p95": float(np.percentile(rho_pooled, 95)),
            "rho_within_median": float(np.median(rho_within)) if rho_within else float("nan"),
            "rho_within_p5": float(np.percentile(rho_within, 5)) if rho_within else float("nan"),
            "rho_within_p95": float(np.percentile(rho_within, 95)) if rho_within else float("nan"),
            "n_docs_median": int(np.median(ns)),
        }
        print(f"{key}: rho_pooled median={summary[key]['rho_pooled_median']:.4f} "
              f"[{summary[key]['rho_pooled_p5']:.4f},{summary[key]['rho_pooled_p95']:.4f}]  "
              f"rho_within median={summary[key]['rho_within_median']:.4f}", flush=True)

    out_path = os.path.join(OUT_DIR, "t1_permutation_stats.json")
    json.dump({"n_reps_planned": n_reps_planned, "n_reps_run": n_reps, "n_reps": n_reps,
               "reduction_reason": reduction_reason, "n_docs": n_docs, "summary": summary,
               "runtime_sec": time.time() - t0}, open(out_path, "w"), indent=2)
    print(f"\n[written] {out_path}", flush=True)
    print(f"[permutation] TOTAL {time.time()-t0:.0f}s", flush=True)

    ckpt = os.path.join(OUT_DIR, "t1_permutation_CHECKPOINT.json")
    if os.path.exists(ckpt):
        os.remove(ckpt)


if __name__ == "__main__":
    main()
