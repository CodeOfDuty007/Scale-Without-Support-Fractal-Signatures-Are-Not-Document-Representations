#!/usr/bin/env python3
"""
T2 Step 0 (not in the plan's numbering, added as a check before trusting
Step 1's new N grid): reproduce the internal verification notes's own
known single-document fBm validation numbers using OUR fGn generator (the
`fbm` PyPI package, Davies-Harte / circulant embedding method) and the
project's unmodified fractal_math.calc_S/calc_H, before running the new
N-grid sweep in t2_step1. If this does not reproduce, the new N grid is not
trustworthy and t2_step1 should not be trusted either.

Known reference point (report table, "production mode" = single document,
non-pooled, exhaustive n_boot=2000 since 400-token n_starts < n_boot):
  H_true=0.30 -> S bias approx -0.072
  H_true=0.50 -> S bias approx -0.160
  H_true=0.65 -> S bias approx -0.231
  H_true=0.80 -> S bias approx -0.310
calc_H should show near-zero bias throughout (report: "no equivalent
per-document pathology").
"""
import os
import sys

import numpy as np
from fbm import FBM

from fractalsig.features.fractal_math import generate_trajectory, calc_H, calc_S  # noqa: E402

N_TRIALS = 30
N_TOKENS = 400
SEED = 20260908


def make_fgn(h, n, rng_seed):
    np.random.seed(rng_seed % (2**32 - 1))
    f = FBM(n=n, hurst=h, length=1, method="daviesharte")
    return f.fgn()  # length n, the increment process -- treated as "RawScores"


def main():
    print(f"[t2-step0] validating fGn generator + calc_S/calc_H against "
          f"the internal verification notes's known table "
          f"(N={N_TOKENS}, {N_TRIALS} trials)\n", flush=True)
    for h_true in (0.30, 0.50, 0.65, 0.80):
        s_hats, h_hats = [], []
        for t in range(N_TRIALS):
            fgn = make_fgn(h_true, N_TOKENS, SEED * 1000 + t)
            traj, incr = generate_trajectory(fgn)
            s_hat, s_r2 = calc_S(traj)
            h_hat, h_r2 = calc_H(incr)
            s_hats.append(s_hat)
            h_hats.append(h_hat)
        s_hats, h_hats = np.array(s_hats), np.array(h_hats)
        print(f"H_true={h_true:.2f}  S_hat_mean={s_hats.mean():.4f}  S_bias={s_hats.mean()-h_true:+.4f}  "
              f"S_std={s_hats.std():.4f}  ||  H_hat_mean={h_hats.mean():.4f}  "
              f"H_bias={h_hats.mean()-h_true:+.4f}  H_std={h_hats.std():.4f}", flush=True)
    print("\n[t2-step0] compare S_bias row to reference: -0.072 / -0.160 / -0.231 / -0.310 "
          "at H=0.30/0.50/0.65/0.80. H_bias should be small throughout.", flush=True)


if __name__ == "__main__":
    main()
