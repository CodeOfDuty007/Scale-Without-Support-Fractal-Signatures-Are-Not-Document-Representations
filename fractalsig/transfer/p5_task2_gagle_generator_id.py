#!/usr/bin/env python3
"""P5 task 2 (plan doc's own T2, distinct from task 6 -- this is GAGLE-only,
2-class): binary generator-family ID, gemma-2b vs mistral-7b, on GAGLE's
LLM rows (Human excluded -- not one of T2's classes).

Reuses P3's exact D-11 pipeline (weight -> calibrate on val at true
prevalence -> report), the same `fit_predict` every P3/P4 binary detector
in this project has used, confirmed general-purpose for any binary y in
{0,1} by reading its signature -- no new training code. Same frozen
`random/random` split (`splits_v1.json`) task 3 already reused.

Reference note: the plan doc cites "the paper's ~97.8% RF reference" for
this task. That number is a different model class (Random Forest) on a
different corpus construction -- reported here for context only, not
chased as a target. LightGBM is used for consistency with every other
classifier this project has trained.
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
import sys

import numpy as np

HERE = _paths.WORKSPACE_STR
D = lambda *p: os.path.join(HERE, "data", *p)


def main():
    from fractalsig.detection.p3_train_eval import load_matrix, build_folds, fit_predict
    from fractalsig.detection.metrics import auroc, tpr_at_1pct_fpr, tpr_at_5pct_fpr, bootstrap_ci

    df, groups = load_matrix()
    feats_all = groups["fractal"] + groups["stat"] + groups["style"]
    print(f"[task2] {len(feats_all)} features", flush=True)

    splits = json.load(open(D("splits_v1.json")))
    folds = {f"{s}/{f}": (tr, va, te) for s, f, tr, va, te in build_folds(df, splits)}
    tr, va, te = folds["random/random"]

    gen_mask = df["Label"].isin(["gemma_2b", "mistral_7b"]).to_numpy()
    tr, va, te = tr & gen_mask, va & gen_mask, te & gen_mask
    print(f"[task2] gemma_2b/mistral_7b only, random/random fold: "
          f"train={tr.sum()} val={va.sum()} test={te.sum()}", flush=True)

    y = (df["Label"] == "mistral_7b").astype(int).to_numpy()  # gemma_2b=0, mistral_7b=1
    X = df[feats_all].to_numpy(dtype=float)

    def build(mask):
        Xs, ys = X[mask], y[mask]
        m = ~np.isnan(Xs).any(axis=1)
        return Xs[m], ys[m], int(m.sum())

    Xtr, ytr, n_tr = build(tr)
    Xva, yva, n_va = build(va)
    Xte, yte, n_te = build(te)
    print(f"[task2] after NaN drop (complete-case): train={n_tr} val={n_va} test={n_te}", flush=True)
    print(f"[task2] class balance -- train mistral_7b frac={ytr.mean():.4f}, "
          f"test mistral_7b frac={yte.mean():.4f}", flush=True)

    p_te, base = fit_predict("lgbm", Xtr, ytr, Xva, yva, Xte, feats_all)

    a = auroc(yte, p_te)
    t1 = tpr_at_1pct_fpr(yte, p_te)
    t5 = tpr_at_5pct_fpr(yte, p_te)
    print(f"\n[task2] TEST (gemma_2b vs mistral_7b, positive=mistral_7b): "
          f"AUROC={a:.4f} TPR@1%FPR={t1:.4f} TPR@5%FPR={t5:.4f}  n_test={n_te}", flush=True)
    print(f"[task2] REFERENCE ONLY -- paper's RF baseline ~97.8% (different model class, "
          f"different corpus construction, not a strict target)", flush=True)

    # D-3/D-12 CI convention: binary task -> stratified bootstrap (preserves
    # class balance per fold, matching fractalsig/detection/metrics.py's own rationale).
    a_pt, a_lo, a_hi, a_k = bootstrap_ci(auroc, yte, p_te, stratified=True)
    t1_pt, t1_lo, t1_hi, t1_k = bootstrap_ci(tpr_at_1pct_fpr, yte, p_te, stratified=True)
    t5_pt, t5_lo, t5_hi, t5_k = bootstrap_ci(tpr_at_5pct_fpr, yte, p_te, stratified=True)
    print(f"[task2] 95% CI (stratified bootstrap, n_boot=2000): "
          f"AUROC=[{a_lo:.4f},{a_hi:.4f}] (k={a_k}) "
          f"TPR@1%FPR=[{t1_lo:.4f},{t1_hi:.4f}] (k={t1_k}) "
          f"TPR@5%FPR=[{t5_lo:.4f},{t5_hi:.4f}] (k={t5_k})", flush=True)

    with open(D("p5_task2_gagle_generator_id.json"), "w") as f:
        json.dump({
            "n_train": n_tr, "n_val": n_va, "n_test": n_te,
            "positive_class": "mistral_7b", "negative_class": "gemma_2b",
            "auroc": float(a), "auroc_ci": [a_lo, a_hi], "auroc_ci_n_valid_folds": a_k,
            "tpr_1pct": float(t1), "tpr_1pct_ci": [t1_lo, t1_hi], "tpr_1pct_ci_n_valid_folds": t1_k,
            "tpr_5pct": float(t5), "tpr_5pct_ci": [t5_lo, t5_hi], "tpr_5pct_ci_n_valid_folds": t5_k,
            "test_mistral_frac": float(yte.mean()),
        }, f, indent=2)
    print("\n[task2] written to data/p5_task2_gagle_generator_id.json")


if __name__ == "__main__":
    main()
