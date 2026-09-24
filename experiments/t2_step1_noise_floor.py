#!/usr/bin/env python3
"""
T2 Step 1 -- noise floor against ground truth (per-document, single-
trajectory regime -- the regime that matches how Gemma_S/Gemma_H/Mistral_S/
Mistral_H are actually computed per document in the feature store, NOT the
pooled/corpus-level "list of trajectories" mode used by
compute_corpus_level_SH).

Generates synthetic fGn at known H, lengths N in {100,200,400,800,1600,3200,
6400,12800}, and records bias/SD/RMSE of Hhat/Shat using the project's own
unmodified fractal_math.calc_S/calc_H (default module scale list, matching
family_a_for_scorer's real point-estimate call signature).

H grid: {0.60, 0.65, 0.70, 0.72} -- GAGLE's own observed corpus-level H runs
0.607-0.714 across all 15 (domain,label) cells (see the internal verification notes2026-07-26
corpus table), so the floor is evaluated where the corpus actually lives, per
the plan's explicit warning against an arbitrary H grid.

CAVEAT (documented, not hidden): this uses the `fbm` PyPI package's
Davies-Harte fGn generator, since the exact generator used in
the internal verification notes was not preserved in this repository.
t2_step0_validate_fgn.py cross-checked this generator + the project's own
calc_S/calc_H against that report's single-document bias table: the
QUALITATIVE pattern reproduces (S severely negatively biased, worsening with
H; H only mildly biased, much less than S, with the report's noted small
low-H-specific bias) but exact numeric values differ, attributed to the
undocumented/unavailable original generator rather than a defect in this
analysis. Treat this floor as internally consistent (safe to compare bias
change vs N) rather than as a byte-exact reproduction of the earlier report.
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
from fbm import FBM

from fractalsig import paths as _paths  # noqa: E402
from fractalsig.runtime import default_workers  # noqa: E402
from fractalsig.features.fractal_math import generate_trajectory, calc_H, calc_S  # noqa: E402

OUT_DIR = _paths.STATS_DIR_STR
N_GRID = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
H_GRID_CALIBRATED = [0.60, 0.65, 0.70, 0.72]
H_GRID_SANITY = [0.30, 0.50, 0.80]  # extra points for the reported figure, not the N* crossing
N_TRIALS = 40
SEED = 20260908


def make_fgn(h, n, rng_seed):
    np.random.seed(rng_seed % (2**32 - 1))
    f = FBM(n=n, hurst=h, length=1, method="daviesharte")
    return f.fgn()


def _one_trial(args):
    h_true, n, trial_seed = args
    fgn = make_fgn(h_true, n, trial_seed)
    traj, incr = generate_trajectory(fgn)
    s_hat, s_r2 = calc_S(traj)
    h_hat, h_r2 = calc_H(incr)
    return h_true, n, s_hat, s_r2, h_hat, h_r2


def main():
    t0 = time.time()
    h_values = sorted(set(H_GRID_CALIBRATED + H_GRID_SANITY))
    tasks = []
    for h in h_values:
        for n in N_GRID:
            for t in range(N_TRIALS):
                tasks.append((h, n, SEED * 100000 + int(h * 1000) * 1000 + n + t))
    print(f"[t2-step1] {len(tasks)} (H,N,trial) tasks, H_grid={h_values}, N_grid={N_GRID}, "
          f"n_trials={N_TRIALS}", flush=True)

    results = {}
    with mp.get_context("fork").Pool(processes=default_workers()) as pool:
        for i, (h, n, s_hat, s_r2, h_hat, h_r2) in enumerate(pool.imap_unordered(_one_trial, tasks, chunksize=20), 1):
            results.setdefault((h, n), []).append((s_hat, s_r2, h_hat, h_r2))
            if i % 2000 == 0:
                print(f"[t2-step1] {i}/{len(tasks)} trials done, {time.time()-t0:.0f}s", flush=True)

    out = {"n_trials": N_TRIALS, "N_grid": N_GRID, "H_grid_calibrated": H_GRID_CALIBRATED,
           "H_grid_sanity": H_GRID_SANITY, "generator": "fbm package, Davies-Harte, seeded",
           "results": []}
    for (h, n), vals in sorted(results.items()):
        vals = np.array(vals)  # columns: s_hat, s_r2, h_hat, h_r2
        s_hat, s_r2, h_hat, h_r2 = vals[:, 0], vals[:, 1], vals[:, 2], vals[:, 3]
        s_valid = s_hat[~np.isnan(s_hat)]
        row = {
            "H_true": h, "N": n, "n_valid_S": int(len(s_valid)), "n_valid_H": int(np.sum(~np.isnan(h_hat))),
            "S_hat_mean": float(np.nanmean(s_hat)), "S_bias": float(np.nanmean(s_hat) - h),
            "S_hat_sd": float(np.nanstd(s_hat)),
            "S_rmse": float(np.sqrt(np.nanmean((s_hat - h) ** 2))),
            "S_r2_mean": float(np.nanmean(s_r2)),
            "H_hat_mean": float(np.nanmean(h_hat)), "H_bias": float(np.nanmean(h_hat) - h),
            "H_hat_sd": float(np.nanstd(h_hat)),
            "H_rmse": float(np.sqrt(np.nanmean((h_hat - h) ** 2))),
            "H_r2_mean": float(np.nanmean(h_r2)),
        }
        out["results"].append(row)
        print(f"H={h:.2f} N={n:5d}  S: bias={row['S_bias']:+.4f} sd={row['S_hat_sd']:.4f}  "
              f"H: bias={row['H_bias']:+.4f} sd={row['H_hat_sd']:.4f}", flush=True)

    out_path = os.path.join(OUT_DIR, "t2_step1_noise_floor.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n[written] {out_path}", flush=True)
    print(f"[t2-step1] TOTAL {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
