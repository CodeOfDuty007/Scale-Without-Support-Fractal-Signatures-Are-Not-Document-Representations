#!/usr/bin/env python3
"""Task 2 (the 9-task plan, 2026-08-26): tabular-detector seed stability
on the same LODO folds/seed-count as the RoBERTa multi-seed investigation.

If RoBERTa's seed instability (AUROC 0.459-0.981 on big_patent) is going to
be a headline claim, the referee's first question is whether the tabular
detector does the same thing under seed variation. "LightGBM is more stable
by construction" was an assertion in the report; this makes it a measurement.

arm=all, learner=lgbm. Runs BOTH config=full (all rows, matches the
population RoBERTa is actually scored on -- roberta_finetune.py applies no
config_mask) and config=clean_both (matches the paired-significance script's
population). Internal review (2026-08-26) flagged that the first version of
this script used config=clean_both alone and got silently compared against
RoBERTa's config=full numbers in the promoted cross-domain table -- two
different row populations (14,324 vs 45,175 test rows on big_patent) read as
one unlabeled tabular AUROC. Both configs are now computed and labeled so
every table in the report can cite the population-matched number explicitly.

Only LGB_PARAMS' random_state varies across seeds; everything else (data,
folds, features) is identical across runs. CPU-only, ~10 seeds x 2 folds x
2 configs x a few seconds each fit -- the earlier refit did 16 fits in 60s,
this is the same order of magnitude.
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
import sys
import time

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

HERE = _paths.WORKSPACE_STR

from fractalsig.detection.p3_train_eval import (load_matrix, build_folds, config_mask, arm_features,
                            assert_disjoint, LGB_PARAMS)  # noqa: E402
from fractalsig.detection.metrics import auroc, tpr_at_1pct_fpr, tpr_at_5pct_fpr  # noqa: E402

D = lambda *p: os.path.join(HERE, *p)
OUT_PATH = D("data", "tabular_seed_stability.json")

# same 10 seeds as the RoBERTa multi-seed investigation (3 original +
# 7 extra big_patent-only seeds), for a like-for-like seed count
SEEDS = [20260825, 20260826, 20260827, 20260828, 20260829,
         20260830, 20260831, 20260832, 20260833, 20260834]
FOLDS = ["big_patent", "billsum"]  # billsum = the paper's headline collapse fold


def main():
    t0 = time.time()
    df, groups = load_matrix()
    feats_all = groups["fractal"] + groups["stat"] + groups["style"]
    splits = json.load(open(D("data", "splits_v1.json")))
    folds = {(s, f): (tr, va, te) for s, f, tr, va, te in build_folds(df, splits)}
    yall = df["y"].to_numpy()
    names = arm_features("all", groups)
    Xa = df[names].to_numpy(float)

    CONFIGS = ["full", "clean_both"]
    masks = {}
    for cfg in CONFIGS:
        cmask, cnote = config_mask(df, cfg, feats_all)
        masks[cfg] = cmask
        print(f"[tabular-seed] config={cfg}: {int(cmask.sum()):,} rows -- {cnote}", flush=True)

    results = {cfg: {fold: [] for fold in FOLDS} for cfg in CONFIGS}
    for cfg in CONFIGS:
        cmask = masks[cfg]
        for fold in FOLDS:
            tr, va, te = folds[("lodo", fold)]
            tr2, va2, te2 = tr & cmask, va & cmask, te & cmask
            assert_disjoint(df, tr2, va2, te2, f"{cfg}/lodo/{fold}")
            Xtr, Xva, Xte = Xa[tr2], Xa[va2], Xa[te2]
            ytr, yva, yte = yall[tr2], yall[va2], yall[te2]
            print(f"[tabular-seed] {cfg}/lodo/{fold}: n_train={tr2.sum():,} n_val={va2.sum():,} n_test={te2.sum():,}", flush=True)

            for seed in SEEDS:
                params = {**LGB_PARAMS, "random_state": seed}
                base = lgb.LGBMClassifier(**params)
                t1 = time.time()
                base.fit(Xtr, ytr)
                cal = CalibratedClassifierCV(estimator=base, cv="prefit", method="sigmoid")
                cal.fit(Xva, yva)
                pos = int(np.where(cal.classes_ == 1)[0][0])
                p = cal.predict_proba(Xte)[:, pos]
                dt = time.time() - t1

                a = auroc(yte, p)
                t1v = tpr_at_1pct_fpr(yte, p)
                t5 = tpr_at_5pct_fpr(yte, p)
                print(f"[tabular-seed] {cfg}/lodo/{fold} seed={seed}: AUROC={a:.4f} TPR1={t1v:.4f} TPR5={t5:.4f} [{dt:.1f}s]", flush=True)
                results[cfg][fold].append({"seed": seed, "auroc": a, "tpr_at_1pct_fpr": t1v, "tpr_at_5pct_fpr": t5,
                                            "n_test": int(te2.sum())})

    for cfg in CONFIGS:
        for fold in FOLDS:
            aurocs = [r["auroc"] for r in results[cfg][fold]]
            t1s = [r["tpr_at_1pct_fpr"] for r in results[cfg][fold]]
            t5s = [r["tpr_at_5pct_fpr"] for r in results[cfg][fold]]
            print(f"\n[tabular-seed SUMMARY] {cfg}/lodo/{fold} (n={len(aurocs)} seeds, n_test={results[cfg][fold][0]['n_test']:,}):", flush=True)
            for name, xs in [("AUROC", aurocs), ("TPR@1%FPR", t1s), ("TPR@5%FPR", t5s)]:
                print(f"    {name}: mean={np.mean(xs):.4f} min={min(xs):.4f} max={max(xs):.4f} range={max(xs)-min(xs):.4f}", flush=True)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[tabular-seed] done in {time.time()-t0:.1f}s, written to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
