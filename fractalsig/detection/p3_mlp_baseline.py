#!/usr/bin/env python3
"""P0-P5 audit, step 5 (2026-08-28): the 9-step Extension-1 brief's Step 6
names logistic regression, LightGBM, AND a shallow MLP as the three model
classes to start hybrid-model training with. LightGBM + logreg were run
across the full P3 grid; the MLP was deferred (D-05) to P5 in the phase
plan and never landed there either -- this closes that gap.

Reuses P3's exact reusable pieces directly, no new training code beyond the
new "mlp" branch added to `p3_train_eval.fit_predict` itself (a shallow
one-hidden-layer sklearn.MLPClassifier, imputed+scaled, balanced via
sample_weight since MLPClassifier has no native class_weight): `load_matrix`
(85-feature matrix), `build_folds` + the frozen `splits_v1.json` (all 8
scheme/fold combinations the lgbm/logreg grid already covers -- random,
5x LODO, 2x LOGO), `arm_features("all", groups)` (the headline 85-feature
arm, matching config=full's `all` arm in `results/p3_tables.md`).

Same D-11 pipeline every other P3/P4/P5 binary detector in this project has
used (class weight -> calibrate on val at true prevalence -> report),
confirmed via `fit_predict`'s own dispatch, not reimplemented.
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
    feats_all = arm_features("all", groups)
    print(f"[mlp_baseline] {len(feats_all)} features (arm=all)", flush=True)

    splits = json.load(open(D("splits_v1.json")))
    folds = list(build_folds(df, splits))

    X = df[feats_all].to_numpy(dtype=float)

    results = {}
    for scheme, fold, tr, va, te in folds:
        key = f"{scheme}/{fold}"
        ytr_full = (df["Label"] != "Human").astype(int).to_numpy()

        # No manual complete-case NaN drop here -- fit_predict's "mlp" branch
        # already wraps SimpleImputer(strategy="median") in its Pipeline
        # (same as "logreg"), and the main grid runner (p3_train_eval.py's
        # config loop) passes NaN-containing rows straight through for both.
        # An earlier version of this script pre-dropped NaN rows, which
        # silently shrank n_test (43,413 -> 15,167 on random/random) below
        # what "config": "full" means for lgbm/logreg -- found via code
        # review (2026-08-29), fixed by matching the grid runner's behavior.
        Xtr, ytr = X[tr], ytr_full[tr]
        Xva, yva = X[va], ytr_full[va]
        Xte, yte = X[te], ytr_full[te]
        print(f"[mlp_baseline] {key}: train={len(ytr)} val={len(yva)} test={len(yte)}", flush=True)

        p_te, base = fit_predict("mlp", Xtr, ytr, Xva, yva, Xte, feats_all)

        a = auroc(yte, p_te)
        t1 = tpr_at_1pct_fpr(yte, p_te)
        t5 = tpr_at_5pct_fpr(yte, p_te)
        a_pt, a_lo, a_hi, a_k = bootstrap_ci(auroc, yte, p_te, stratified=True)
        t1_pt, t1_lo, t1_hi, t1_k = bootstrap_ci(tpr_at_1pct_fpr, yte, p_te, stratified=True)
        t5_pt, t5_lo, t5_hi, t5_k = bootstrap_ci(tpr_at_5pct_fpr, yte, p_te, stratified=True)
        print(f"[mlp_baseline] {key}: AUROC={a:.4f} [{a_lo:.4f},{a_hi:.4f}] "
              f"TPR@1%FPR={t1:.4f} [{t1_lo:.4f},{t1_hi:.4f}] "
              f"TPR@5%FPR={t5:.4f} [{t5_lo:.4f},{t5_hi:.4f}]  n_test={len(yte)}", flush=True)

        results[key] = {
            "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)),
            "auroc": float(a), "auroc_ci": [a_lo, a_hi], "auroc_ci_n_valid_folds": a_k,
            "tpr_1pct": float(t1), "tpr_1pct_ci": [t1_lo, t1_hi], "tpr_1pct_ci_n_valid_folds": t1_k,
            "tpr_5pct": float(t5), "tpr_5pct_ci": [t5_lo, t5_hi], "tpr_5pct_ci_n_valid_folds": t5_k,
        }

    with open(D("p3_mlp_baseline.json"), "w") as f:
        json.dump({"learner": "mlp", "arm": "all", "config": "full",
                    "architecture": "sklearn.MLPClassifier, 1 hidden layer (64 units, relu), "
                                     "imputed(median)+scaled inputs, balanced via sample_weight",
                    "results": results}, f, indent=2)
    print("\n[mlp_baseline] written to data/p3_mlp_baseline.json")


if __name__ == "__main__":
    main()
