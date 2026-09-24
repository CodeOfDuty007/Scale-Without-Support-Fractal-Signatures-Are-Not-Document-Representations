#!/usr/bin/env python3
"""P0-P5 audit, step 6 (2026-08-28): the 9-step Extension-1 brief's Step 8
names a minimum 4-way ablation: fractal-only / fractal+log-PPL /
fractal+stylistic / all-features. The first, third, and fourth were already
run across the full grid; `fractal+log-PPL` specifically was never
isolated -- `STAT` bundles log-PPL moments with entropy and GLTR rank-bucket
features (17 columns total), so `fractal+stat` (already run) is a strictly
broader combination than the spec's own "fractal + log-PPL" phrase asks for.

New `LOGPPL` feature group (`p3_train_eval.py`) isolates just the
log-PPL-derived columns -- {Gemma,Mistral}_LogPPL_{Mean,Var,Skew,Kurtosis}
plus `Cross_PPL_Ratio` (a ratio of the two scorers' own log-PPL, included
deliberately since it is log-PPL-derived and nothing else) -- explicitly
EXCLUDING Entropy_Mean and the three GLTR_top{10,100,1000}_frac columns,
which is what makes `fractal+stat` a superset rather than this combo.

New `"fractal+logppl"` arm added to `arm_features` (purely additive). Reuses
`load_matrix`/`build_folds`/`splits_v1.json`/`fit_predict("lgbm",...)`
directly -- same D-11 pipeline, same lgbm hyperparameters, all 8 scheme/fold
combinations, so results are population-matched against the already-run
`stat_only` and `fractal+stat` cells in `data/p3_t1_results.json` (config=
full, NaN-native LightGBM, no complete-case drop -- verified by n_test match
before any comparison is quoted, the same check that caught an error in
step 5's MLP comparison).
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
import sys

import numpy as np

HERE = _paths.WORKSPACE_STR
D = lambda *p: os.path.join(HERE, "data", *p)


def main():
    from fractalsig.detection.p3_train_eval import load_matrix, build_folds, arm_features, fit_predict
    from fractalsig.detection.metrics import auroc, tpr_at_1pct_fpr, tpr_at_5pct_fpr, bootstrap_ci

    df, groups = load_matrix()
    feats = arm_features("fractal+logppl", groups)
    print(f"[fractal_logppl] {len(feats)} features (arm=fractal+logppl)", flush=True)

    splits = json.load(open(D("splits_v1.json")))
    folds = list(build_folds(df, splits))
    X = df[feats].to_numpy(dtype=float)
    y_full = (df["Label"] != "Human").astype(int).to_numpy()

    # config=full behavior, matched exactly to p3_train_eval.py's own grid
    # loop (`Xtr, Xva, Xte = Xa[tr], Xa[va], Xa[te]`, no row-dropping under
    # config=full -- LightGBM routes NaN internally at each split). NOT
    # complete-case: an earlier draft of this script (and step 5's MLP
    # comparison, caught the same way) dropped any row with a NaN feature,
    # which silently shrinks the test population below config=full's
    # stat_only/fractal+stat/fractal_only comparator cells and makes any
    # n_test comparison invalid. Verified below that n_test matches exactly.
    def build(mask):
        return X[mask], y_full[mask]

    results = {}
    for scheme, fold, tr, va, te in folds:
        key = f"{scheme}/{fold}"
        Xtr, ytr = build(tr)
        Xva, yva = build(va)
        Xte, yte = build(te)
        print(f"[fractal_logppl] {key}: train={len(ytr)} val={len(yva)} test={len(yte)}", flush=True)

        p_te, base = fit_predict("lgbm", Xtr, ytr, Xva, yva, Xte, feats)

        a = auroc(yte, p_te)
        t1 = tpr_at_1pct_fpr(yte, p_te)
        t5 = tpr_at_5pct_fpr(yte, p_te)
        a_pt, a_lo, a_hi, a_k = bootstrap_ci(auroc, yte, p_te, stratified=True)
        t1_pt, t1_lo, t1_hi, t1_k = bootstrap_ci(tpr_at_1pct_fpr, yte, p_te, stratified=True)
        t5_pt, t5_lo, t5_hi, t5_k = bootstrap_ci(tpr_at_5pct_fpr, yte, p_te, stratified=True)
        print(f"[fractal_logppl] {key}: AUROC={a:.4f} [{a_lo:.4f},{a_hi:.4f}] "
              f"TPR@1%FPR={t1:.4f} [{t1_lo:.4f},{t1_hi:.4f}] "
              f"TPR@5%FPR={t5:.4f} [{t5_lo:.4f},{t5_hi:.4f}]  n_test={len(yte)}", flush=True)

        results[key] = {
            "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)),
            "auroc": float(a), "auroc_ci": [a_lo, a_hi], "auroc_ci_n_valid_folds": a_k,
            "tpr_1pct": float(t1), "tpr_1pct_ci": [t1_lo, t1_hi], "tpr_1pct_ci_n_valid_folds": t1_k,
            "tpr_5pct": float(t5), "tpr_5pct_ci": [t5_lo, t5_hi], "tpr_5pct_ci_n_valid_folds": t5_k,
        }

    with open(D("p3_fractal_logppl_arm.json"), "w") as f:
        json.dump({"learner": "lgbm", "arm": "fractal+logppl", "config": "full",
                    "population": "full (NaN-native lgbm, no complete-case drop)",
                    "features": feats, "results": results}, f, indent=2)
    print("\n[fractal_logppl] written to data/p3_fractal_logppl_arm.json")


if __name__ == "__main__":
    main()
