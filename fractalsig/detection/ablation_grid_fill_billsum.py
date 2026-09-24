#!/usr/bin/env python3
"""Fills a real gap in the P3 ablation grid, found while answering the
the question of whether further ablations were needed (2026-08-26).

`results/p3_tables.md` only ran `all`, `all_minus_fractal`, `fractal_only`,
and `stat_only` across all 5 confound configs. `style_only`, `fractal+stat`,
`fractal+style`, and `all_minus_length` were only ever run under config 1
(`full`) -- never checked against the short-window/missingness/clean-both
confound routes the way `all`/`all_minus_fractal` were.

This matters now specifically because today's Holm-Bonferroni correction
(internal review) found the `all`-vs-`all_minus_fractal` significant effect
collapses from "8 of 40 folds" to "3 of 40, all on billsum" once corrected
-- so billsum is now the one fold where confound-route sensitivity is
actually established as real, not just possible. This fills the missing
arm x config cells for billsum ONLY (the fold where it matters), reusing
p3_train_eval.py's exact fit_predict()/config_mask()/arm_features() -- same
learner (lgbm), same seed, no new methodology.
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
import sys
import time

import numpy as np

HERE = _paths.WORKSPACE_STR

from fractalsig.detection.p3_train_eval import (load_matrix, build_folds, config_mask, arm_features,
                            assert_disjoint, fit_predict)  # noqa: E402
from fractalsig.detection.metrics import auroc, tpr_at_1pct_fpr, tpr_at_5pct_fpr  # noqa: E402

D = lambda *p: os.path.join(HERE, *p)
OUT_PATH = D("data", "ablation_grid_fill_billsum.json")

CONFIGS = ["no_short_window", "no_nan_signal", "complete_case", "clean_both"]
ARMS = ["style_only", "fractal+stat", "fractal+style", "all_minus_length"]
FOLD = "billsum"
SCHEME = "lodo"


def main():
    t0 = time.time()
    df, groups = load_matrix()
    feats_all = groups["fractal"] + groups["stat"] + groups["style"]
    splits = json.load(open(D("data", "splits_v1.json")))
    folds = {(s, f): (tr, va, te) for s, f, tr, va, te in build_folds(df, splits)}
    yall = df["y"].to_numpy()

    tr0, va0, te0 = folds[(SCHEME, FOLD)]

    results = {}
    for cfg in CONFIGS:
        cmask, cnote = config_mask(df, cfg, feats_all)
        tr, va, te = tr0 & cmask, va0 & cmask, te0 & cmask
        assert_disjoint(df, tr, va, te, f"{cfg}/{SCHEME}/{FOLD}")
        print(f"[fill] {cfg}/{FOLD}: n_train={tr.sum():,} n_val={va.sum():,} n_test={te.sum():,} -- {cnote}", flush=True)
        results[cfg] = {}
        for arm in ARMS:
            names = arm_features(arm, groups)
            Xa = df[names].to_numpy(float)
            Xtr, Xva, Xte = Xa[tr], Xa[va], Xa[te]
            ytr, yva, yte = yall[tr], yall[va], yall[te]
            t1 = time.time()
            p, base = fit_predict("lgbm", Xtr, ytr, Xva, yva, Xte, names)
            dt = time.time() - t1
            a = auroc(yte, p)
            t1v = tpr_at_1pct_fpr(yte, p)
            t5 = tpr_at_5pct_fpr(yte, p)
            print(f"[fill] {cfg}/{FOLD}/{arm}: AUROC={a:.4f} TPR1={t1v:.4f} TPR5={t5:.4f} [{dt:.1f}s]", flush=True)
            results[cfg][arm] = {"auroc": a, "tpr_at_1pct_fpr": t1v, "tpr_at_5pct_fpr": t5,
                                  "n_test": int(te.sum())}

    # config 1 (full) reference values, already in results/p3_tables.md, for comparison
    ref = {
        "style_only": {"auroc": 0.8722, "tpr_at_1pct_fpr": 0.0491, "tpr_at_5pct_fpr": 0.3110},
        "fractal+stat": {"auroc": 0.8511, "tpr_at_1pct_fpr": 0.3219, "tpr_at_5pct_fpr": 0.5146},
        "fractal+style": {"auroc": 0.9099, "tpr_at_1pct_fpr": 0.2864, "tpr_at_5pct_fpr": 0.5184},
        "all_minus_length": {"auroc": 0.9658, "tpr_at_1pct_fpr": 0.6997, "tpr_at_5pct_fpr": 0.8761},
    }
    print("\n=== full (config 1) reference, billsum, for comparison ===", flush=True)
    for arm, v in ref.items():
        print(f"  {arm}: AUROC={v['auroc']:.4f} TPR1={v['tpr_at_1pct_fpr']:.4f} TPR5={v['tpr_at_5pct_fpr']:.4f}", flush=True)

    with open(OUT_PATH, "w") as f:
        json.dump({"config1_full_reference": ref, "filled": results}, f, indent=2, default=float)
    print(f"\n[fill] done in {time.time()-t0:.1f}s, written to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
