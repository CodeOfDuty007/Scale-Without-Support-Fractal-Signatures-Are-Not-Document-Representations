#!/usr/bin/env python3
"""Refit and PERSIST what job 125 threw away.

job 125 (p3_train_eval.py) computed per-row predictions and test-row indices
for every cell, used them for an internal alignment check, then
`results.pop("_pred")` / `results.pop("_idx")` before the only disk write --
so nothing but aggregate metrics survived. Three downstream needs now require
real per-row scores: Part C error analysis, Part D's paired baseline
comparisons (bootstrap_ci_paired against config 5), and F-5(a)'s
short-text scoring. One refit serves all three, keyed by DOCUMENT ID (not
job 125's `_idx`, which only meant anything inside that process) so a
baseline scored independently can join against it.

Two targets:
  - config=clean_both (5), arm=all, learner=lgbm, ALL 8 folds (random + 5
    LODO + 2 LOGO) -- matches the scope RoBERTa is being fit across, so every
    baseline/RoBERTa fold has a config-5 counterpart to pair against.
  - config=full (1), arm=all, learner=lgbm, random fold ONLY -- this is the
    "main test supply" model for F-5(a). Deliberately NOT config 5: config 5's
    row population (clean_both) is by construction 0% missing, so its model
    never saw a real NaN during training and applying it to F-5(a)'s
    all-Family-A/B-missing rows would be pure extrapolation. config=full's
    training population contains real missingness (short-window rows, F-5(b)
    rows), so LightGBM has actually learned split directions for NaN there --
    the model F-5(a) scoring needs, not the paper's headline model.
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
import sys
import time

import joblib
import numpy as np

HERE = _paths.WORKSPACE_STR

from sklearn.calibration import CalibratedClassifierCV

from fractalsig.detection.p3_train_eval import (load_matrix, build_folds, config_mask, arm_features,
                            fit_predict, assert_disjoint, evaluate, log)

D = lambda *p: os.path.join(HERE, *p)
OUT_PRED = D("data", "p3_refit_predictions.jsonl")
OUT_SUMMARY = D("data", "p3_refit_summary.json")
OUT_MODELS_DIR = D("feature_store", "p3_refit_models")

TARGETS = [
    ("clean_both", "all", "lgbm", None),
    ("full", "all", "lgbm", None),
]


def main():
    t0 = time.time()
    df, groups = load_matrix()
    feats_all = groups["fractal"] + groups["stat"] + groups["style"]
    splits = json.load(open(D("data", "splits_v1.json")))
    folds = list(build_folds(df, splits))
    yall = df["y"].to_numpy()
    ids = df["id"].to_numpy()
    log(f"[refit] {len(df):,} rows x {len(feats_all)} features, {len(folds)} folds available")

    os.makedirs(OUT_MODELS_DIR, exist_ok=True)
    fpred = open(OUT_PRED, "w")
    summary_cells = []

    for cfg, arm, learner, scheme_filter in TARGETS:
        cmask, cnote = config_mask(df, cfg, feats_all)
        log(f"[refit] config={cfg}: {int(cmask.sum()):,} rows -- {cnote}")
        for scheme, fold, tr, va, te in folds:
            if scheme_filter is not None and scheme not in scheme_filter:
                continue
            tr2, va2, te2 = tr & cmask, va & cmask, te & cmask
            if tr2.sum() == 0 or va2.sum() == 0 or te2.sum() == 0:
                log(f"  [skip] {cfg}/{scheme}/{fold}: empty partition after config mask")
                continue
            assert_disjoint(df, tr2, va2, te2, f"{cfg}/{scheme}/{fold}")
            ytr, yva, yte = yall[tr2], yall[va2], yall[te2]
            if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2 or len(np.unique(yte)) < 2:
                log(f"  [skip] {cfg}/{scheme}/{fold}: a partition holds one class")
                continue

            names = arm_features(arm, groups)
            Xa = df[names].to_numpy(float)
            Xtr, Xva, Xte = Xa[tr2], Xa[va2], Xa[te2]

            t1 = time.time()
            p, base = fit_predict(learner, Xtr, ytr, Xva, yva, Xte, names)
            dt = time.time() - t1

            # fit_predict returns `base`, the UNCALIBRATED fitted model -- `p`
            # itself comes from the calibrated wrapper built and discarded
            # inside fit_predict. Reconstruct that same calibrator (identical
            # inputs: the same already-fit `base`, the same Xva/yva -> sklearn's
            # sigmoid/Platt fit is deterministic) so what gets PERSISTED is the
            # object that actually produces `p`, not a silently uncalibrated one.
            cal = CalibratedClassifierCV(estimator=base, cv="prefit", method="sigmoid")
            cal.fit(Xva, yva)
            pos = int(np.where(cal.classes_ == 1)[0][0])
            p_check = cal.predict_proba(Xte)[:, pos]
            assert np.allclose(p_check, p, atol=1e-12), (
                f"{cfg}/{scheme}/{fold}: reconstructed calibrator diverges from "
                f"fit_predict's own output -- do not persist a model that can't "
                f"reproduce what was actually evaluated")

            te_ids = ids[te2]
            for doc_id, y_, p_ in zip(te_ids, yte, p):
                fpred.write(json.dumps({
                    "id": str(doc_id), "y": int(y_), "score": float(p_),
                    "config": cfg, "scheme": scheme, "fold": fold,
                    "arm": arm, "learner": learner,
                }) + "\n")
            fpred.flush()

            cell = evaluate(yte, p, with_ci=False)
            pt = cell["point"]
            log(f"  [{cfg}/{scheme}/{fold}] n_test={int(te2.sum()):,} "
                f"AUROC={pt['auroc']:.4f} TPR@5%={pt['tpr_at_5pct_fpr']:.4f} "
                f"TPR@1%={pt['tpr_at_1pct_fpr']:.4f} [{dt:.1f}s]")

            model_path = os.path.join(OUT_MODELS_DIR, f"{cfg}_{scheme}_{fold}_{arm}_{learner}.joblib")
            joblib.dump({"model": cal, "base_model": base, "feature_names": names,
                         "config": cfg, "scheme": scheme, "fold": fold,
                         "arm": arm, "learner": learner}, model_path)

            summary_cells.append({
                "config": cfg, "scheme": scheme, "fold": fold, "arm": arm, "learner": learner,
                "n_train": int(tr2.sum()), "n_val": int(va2.sum()), "n_test": int(te2.sum()),
                "fit_seconds": round(dt, 1), "point": pt,
                "model_path": os.path.relpath(model_path, HERE),
            })

    fpred.close()
    with open(OUT_SUMMARY, "w") as f:
        json.dump({"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "predictions_path": os.path.relpath(OUT_PRED, HERE),
                   "models_dir": os.path.relpath(OUT_MODELS_DIR, HERE),
                   "cells": summary_cells}, f, indent=2, default=float)
    log(f"[refit] done in {time.time()-t0:.1f}s. "
        f"{len(summary_cells)} cells, predictions at {OUT_PRED}, models at {OUT_MODELS_DIR}/")


if __name__ == "__main__":
    main()
