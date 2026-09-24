#!/usr/bin/env python3
"""Review-directed check (2026-08-27): is the tabular detector's sentence_reorder
collapse (95.6% -> 27.8% recall) order-dependence in the FRACTAL features
specifically, or a generic distribution-shift effect that would hit any
feature arm? Refits `fractal_only` and `all_minus_fractal` (config=full,
random/random fold, matching p3_refit_and_persist.py's own TARGET shape,
just two more arms of the same job) on P3's training rows, then scores the
P4 sentence_reorder population through both and compares recall collapse.
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
import sys

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve

HERE = _paths.WORKSPACE_STR
from fractalsig.detection.p3_train_eval import load_matrix, build_folds, config_mask, arm_features, fit_predict, assert_disjoint  # noqa: E402

D = lambda *p: os.path.join(HERE, *p)
ARMS = ["stat_only", "style_only"]


def main():
    df, groups = load_matrix()
    feats_all = groups["fractal"] + groups["stat"] + groups["style"]
    splits = json.load(open(D("data", "splits_v1.json")))
    folds = list(build_folds(df, splits))
    yall = df["y"].to_numpy()
    ids = df["id"].to_numpy()

    cmask, cnote = config_mask(df, "full", feats_all)
    tr = va = te = None
    for scheme, fold, t, v, e in folds:
        if scheme == "random" and fold == "random":
            tr, va, te = t, v, e
            break
    tr2, va2, te2 = tr & cmask, va & cmask, te & cmask
    assert_disjoint(df, tr2, va2, te2, "full/random/random (arm decomposition)")
    ytr, yva, yte = yall[tr2], yall[va2], yall[te2]

    # Build the P4 population + tabular feature rows (already computed and
    # merged into p4_task5_tabular_scores.jsonl -- but that only has the
    # PERSISTED "all" arm's scores. Need the raw feature vectors to score
    # under the new arms, so recompute the feature dict per row the same
    # way p4_task5_tabular_features.py did, but keep it as a matrix here.
    import joblib
    bundle = joblib.load(D("feature_store", "p3_refit_models", "full_random_random_all_lgbm.joblib"))
    all_feature_names = bundle["feature_names"]

    # Re-derive raw feature vectors for sentence_reorder rows by re-running
    # the exact same feature builder used for the full population -- reuse
    # its function directly rather than duplicating logic.
    from fractalsig.robustness.p4_task5_tabular_features import build_feature_row
    from fractalsig.pipeline.build_p1_feature_store_gpu import load_models
    _, device, _, _, _, embedder = load_models(scorers=())

    pop = [r for r in (json.loads(l) for l in open(D("data", "p4_task5_population.jsonl")))
           if r["variant"] == "sentence_reorder"]
    rescore = {}
    for scorer in ("gemma", "mistral"):
        rescore[scorer] = {}
        for batch in (1, 2):
            for l in open(D("data", f"p4_rescore_batch{batch}_{scorer}.jsonl")):
                r = json.loads(l)
                rescore[scorer][(r["id"], r["variant"])] = r

    print(f"[arm-decomp] building raw feature vectors for {len(pop)} sentence_reorder rows", flush=True)
    rows_feats = []
    for i, row in enumerate(pop):
        gemma_rec = rescore["gemma"].get((row["id"], "sentence_reorder"))
        mistral_rec = rescore["mistral"].get((row["id"], "sentence_reorder"))
        if gemma_rec is None or mistral_rec is None:
            continue
        feats = build_feature_row(row["text"], gemma_rec, mistral_rec, embedder, device)
        rows_feats.append(feats)
        if (i + 1) % 3000 == 0:
            print(f"[arm-decomp]   {i+1}/{len(pop)}", flush=True)
    print(f"[arm-decomp] {len(rows_feats)} feature rows built", flush=True)

    results = {}
    for arm in ARMS:
        names = arm_features(arm, groups)
        Xtr, Xva, Xte = df.loc[tr2, names].to_numpy(float), df.loc[va2, names].to_numpy(float), df.loc[te2, names].to_numpy(float)
        p, base = fit_predict("lgbm", Xtr, ytr, Xva, yva, Xte, names)
        cal = CalibratedClassifierCV(estimator=base, cv="prefit", method="sigmoid")
        cal.fit(Xva, yva)
        pos = int(np.where(cal.classes_ == 1)[0][0])

        # pre-attack recall @1%/5% FPR
        fpr, tpr, thresh = roc_curve(yva, cal.predict_proba(Xva)[:, pos])
        idx1 = np.where(fpr <= 0.01)[0]
        idx5 = np.where(fpr <= 0.05)[0]
        t1 = thresh[idx1[-1]] if len(idx1) else None
        t5 = thresh[idx5[-1]] if len(idx5) else None
        p_te = cal.predict_proba(Xte)[:, pos]
        llm_mask_te = yte == 1
        r1_pre = float(np.mean(p_te[llm_mask_te] >= t1))
        r5_pre = float(np.mean(p_te[llm_mask_te] >= t5))

        Xp4 = np.array([[f.get(n, np.nan) for n in names] for f in rows_feats], dtype=float)
        p_p4 = cal.predict_proba(Xp4)[:, pos]
        r1_post = float(np.mean(p_p4 >= t1))
        r5_post = float(np.mean(p_p4 >= t5))

        results[arm] = {"n_features": len(names), "pre_r1": r1_pre, "pre_r5": r5_pre,
                          "post_r1": r1_post, "post_r5": r5_post}
        print(f"[arm-decomp] {arm:20} n_feat={len(names):3}  "
              f"pre@1%={r1_pre:.4f} post@1%={r1_post:.4f}  pre@5%={r5_pre:.4f} post@5%={r5_post:.4f}",
              flush=True)

    with open(D("data", "p4_task5_arm_decomposition_stat_vs_style.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
