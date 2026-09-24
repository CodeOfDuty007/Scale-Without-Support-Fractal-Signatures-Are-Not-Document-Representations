#!/usr/bin/env python3
"""P5 task 4 (planned sequence: task 4 -> task 6 -> task 3): train on
GAGLE, test on RAID -- "the strongest external validity check" per the plan
doc (Extension1_PhaseWise_Execution_Plan.md SS P5). Reuses the ALREADY-
PERSISTED GAGLE model directly (feature_store/p3_refit_models/
full_random_random_all_lgbm.joblib, config=full/random/random/all/lgbm) --
no retraining, per this project's own reuse-not-restate discipline. Feature
schema verified byte-exact against the RAID feature store before scoring
(85/85 columns match, both built through the same build_feature_row).

Reports AUROC and recall@1%/5%FPR on RAID, using the SAME fixed thresholds
this project has used throughout (derived once on GAGLE's own val fold --
already persisted in p3_refit_summary.json / p4_task5_tabular_thresholds.json
-- reused here rather than re-derived, since re-deriving on RAID would
answer a different question: "what threshold would RAID itself calibrate
to" rather than "how much does the GAGLE-calibrated detector degrade
out-of-distribution"), reported against the GAGLE in-distribution baseline
as the headline transfer-drop number.
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
import sys

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from fractalsig.detection.metrics import auroc as _auroc_fn, tpr_at_1pct_fpr, tpr_at_5pct_fpr, bootstrap_ci

HERE = _paths.WORKSPACE_STR
D = lambda *p: os.path.join(HERE, "data", *p)


def main():
    bundle = joblib.load(os.path.join(HERE, "feature_store", "p3_refit_models",
                                        "full_random_random_all_lgbm.joblib"))
    cal_model, feature_names = bundle["model"], bundle["feature_names"]
    pos = int(np.where(cal_model.classes_ == 1)[0][0])

    # --- GAGLE in-distribution baseline, already persisted, not re-derived ---
    p3_summary = json.load(open(D("p3_refit_summary.json")))
    gagle_cell = next(c for c in p3_summary["cells"]
                       if c["config"] == "full" and c["scheme"] == "random"
                       and c["fold"] == "random" and c["arm"] == "all" and c["learner"] == "lgbm")
    gagle_auroc = gagle_cell["point"]["auroc"]
    gagle_tpr1 = gagle_cell["point"]["tpr_at_1pct_fpr"]
    gagle_tpr5 = gagle_cell["point"]["tpr_at_5pct_fpr"]
    print(f"[task4] GAGLE in-distribution (config=full/random/random/all/lgbm): "
          f"AUROC={gagle_auroc:.4f} TPR@1%={gagle_tpr1:.4f} TPR@5%={gagle_tpr5:.4f}", flush=True)

    # fixed thresholds from GAGLE's own val fold, already persisted for this exact model
    tab_thr = json.load(open(D("p4_task5_tabular_thresholds.json")))
    t1, t5 = tab_thr["t1_fpr1pct"], tab_thr["t5_fpr5pct"]
    print(f"[task4] reusing GAGLE-calibrated fixed thresholds: t1={t1:.4f} t5={t5:.4f}", flush=True)

    # --- score RAID ---
    rows = [json.loads(l) for l in open(D("p5_raid_features.jsonl"))]
    print(f"[task4] {len(rows)} RAID rows to score", flush=True)
    y = np.array([0 if r["Label"] == "human" else 1 for r in rows])
    X = np.array([[r[n] for n in feature_names] for r in rows], dtype=float)
    nan_mask = np.isnan(X).any(axis=1)
    print(f"[task4] {nan_mask.sum()} rows have a NaN feature, excluding (complete-case, F-5(a) precedent)", flush=True)
    X, y = X[~nan_mask], y[~nan_mask]
    rows_clean = [r for r, m in zip(rows, nan_mask) if not m]

    p = cal_model.predict_proba(X)[:, pos]
    raid_auroc = roc_auc_score(y, p)
    raid_tpr1 = float(np.mean(p[y == 1] >= t1))
    raid_tpr5 = float(np.mean(p[y == 1] >= t5))
    print(f"\n[task4] RAID transfer, GAGLE-threshold (n={len(y)}, {int((y==1).sum())} LLM / {int((y==0).sum())} human): "
          f"AUROC={raid_auroc:.4f} TPR@1%={raid_tpr1:.4f} TPR@5%={raid_tpr5:.4f}", flush=True)
    print(f"[task4] DROP vs GAGLE in-distribution: "
          f"AUROC {gagle_auroc-raid_auroc:+.4f}  TPR@1% {gagle_tpr1-raid_tpr1:+.4f}  "
          f"TPR@5% {gagle_tpr5-raid_tpr5:+.4f}", flush=True)

    # CRITICAL self-check (found 2026-08-28): this project's fixed-threshold
    # convention (derive once on the original corpus, apply unchanged to the
    # target population) is valid for every WITHIN-corpus attack test to date
    # because the human population never changes under an attack -- FPR is
    # stable by construction. It does NOT transfer to a CROSS-corpus test
    # like this one, where the human population itself is entirely
    # different. Always check the realized FPR before trusting a
    # cross-corpus "GAGLE-threshold" number.
    fpr1_realized = float(np.mean(p[y == 0] >= t1))
    fpr5_realized = float(np.mean(p[y == 0] >= t5))
    print(f"\n[task4] REALIZED FPR of the GAGLE-calibrated thresholds on RAID's own human population: "
          f"t1 realized FPR={fpr1_realized:.4f} (target 0.01)   t5 realized FPR={fpr5_realized:.4f} (target 0.05)", flush=True)
    if fpr1_realized > 0.02 or fpr5_realized > 0.10:
        print("[task4] *** MISCALIBRATED: the GAGLE-threshold TPR numbers above are NOT a real "
              "1%/5%-FPR comparison on RAID -- use the RAID-native-threshold numbers below as primary ***", flush=True)

    # also re-derive a RAID-native threshold for reference (NOT the headline number,
    # but useful context: is RAID's AUROC itself okay even if the GAGLE threshold
    # transfers badly due to score-distribution shift, a different failure mode
    # than "the ranking itself degrades")
    fpr, tpr, thresh = roc_curve(y, p)
    idx1 = np.where(fpr <= 0.01)[0]
    idx5 = np.where(fpr <= 0.05)[0]
    raid_native_t1 = thresh[idx1[-1]] if len(idx1) else None
    raid_native_t5 = thresh[idx5[-1]] if len(idx5) else None
    raid_native_tpr1 = float(np.mean(p[y == 1] >= raid_native_t1)) if raid_native_t1 is not None else None
    raid_native_tpr5 = float(np.mean(p[y == 1] >= raid_native_t5)) if raid_native_t5 is not None else None
    print(f"\n[task4] REFERENCE ONLY -- RAID-native threshold (re-derived on RAID itself, "
          f"NOT the headline transfer number): TPR@1%={raid_native_tpr1:.4f} TPR@5%={raid_native_tpr5:.4f} "
          f"(t1={raid_native_t1:.4f}, t5={raid_native_t5:.4f})", flush=True)
    print(f"[task4] gap between GAGLE-threshold-TPR and RAID-native-threshold-TPR isolates "
          f"score-distribution shift from ranking-quality degradation", flush=True)

    # CIs on the PRIMARY (RAID-native-threshold) numbers only, per D-3/D-12
    # convention -- stratified bootstrap (binary task, human vs LLM), same
    # metric functions used everywhere else in this project. tpr_at_1pct_fpr/
    # tpr_at_5pct_fpr re-derive their own 1%/5%-FPR operating point on each
    # bootstrap fold's own ROC curve, so this reproduces "RAID-native
    # threshold" behavior per fold, not the single fixed threshold above.
    auroc_pt, auroc_lo, auroc_hi, auroc_k = bootstrap_ci(_auroc_fn, y, p, stratified=True)
    ntpr1_pt, ntpr1_lo, ntpr1_hi, ntpr1_k = bootstrap_ci(tpr_at_1pct_fpr, y, p, stratified=True)
    ntpr5_pt, ntpr5_lo, ntpr5_hi, ntpr5_k = bootstrap_ci(tpr_at_5pct_fpr, y, p, stratified=True)
    print(f"\n[task4] 95% CI on RAID-native-threshold PRIMARY numbers (stratified bootstrap, n_boot=2000): "
          f"AUROC=[{auroc_lo:.4f},{auroc_hi:.4f}] (k={auroc_k}) "
          f"TPR@1%FPR=[{ntpr1_lo:.4f},{ntpr1_hi:.4f}] (k={ntpr1_k}) "
          f"TPR@5%FPR=[{ntpr5_lo:.4f},{ntpr5_hi:.4f}] (k={ntpr5_k})", flush=True)

    # sanity check: bootstrap_ci's internal point (metric_fn(y, p), re-deriving
    # its own 1%/5%-FPR operating point on the full sample) must equal the
    # fixed-raid_native_t1/t5 point computed above -- both take fpr<=target and
    # index [-1] on the same (y, p), so they should be bit-identical. If not,
    # the reported CI isn't centered on the number in the headline text.
    assert ntpr1_pt == raid_native_tpr1, f"TPR@1% point mismatch: bootstrap={ntpr1_pt} vs fixed-threshold={raid_native_tpr1}"
    assert ntpr5_pt == raid_native_tpr5, f"TPR@5% point mismatch: bootstrap={ntpr5_pt} vs fixed-threshold={raid_native_tpr5}"
    print(f"[task4] verified: bootstrap_ci's internal point estimate matches the reported "
          f"raid_native_tpr1/tpr5 exactly (CI is centered on the reported number)", flush=True)

    # per-domain and per-generator breakdown
    from collections import defaultdict
    by_domain = defaultdict(lambda: {"n": 0, "y": [], "p": []})
    by_generator = defaultdict(lambda: {"n": 0, "y": [], "p": []})
    for r, yi, pi in zip(rows_clean, y, p):
        by_domain[r["domain"]]["y"].append(yi); by_domain[r["domain"]]["p"].append(pi)
        by_generator[r["Label"]]["y"].append(yi); by_generator[r["Label"]]["p"].append(pi)

    print(f"\n{'domain':14}{'n':>8}{'AUROC':>10}{'recall@1%':>12}{'recall@5%':>12}")
    domain_results = {}
    for dom, d in sorted(by_domain.items()):
        yy, pp = np.array(d["y"]), np.array(d["p"])
        if len(set(yy)) < 2:
            continue
        a = roc_auc_score(yy, pp)
        r1 = float(np.mean(pp[yy == 1] >= t1)) if (yy == 1).any() else float("nan")
        r5 = float(np.mean(pp[yy == 1] >= t5)) if (yy == 1).any() else float("nan")
        print(f"{dom:14}{len(yy):8}{a:10.4f}{r1:12.4f}{r5:12.4f}")
        domain_results[dom] = {"n": len(yy), "auroc": a, "recall_1pct": r1, "recall_5pct": r5}

    print(f"\n{'generator':16}{'n':>8}{'recall@1%':>12}{'recall@5%':>12}")
    generator_results = {}
    for gen, d in sorted(by_generator.items()):
        if gen == "human":
            continue
        yy, pp = np.array(d["y"]), np.array(d["p"])
        r1 = float(np.mean(pp >= t1))
        r5 = float(np.mean(pp >= t5))
        print(f"{gen:16}{len(yy):8}{r1:12.4f}{r5:12.4f}")
        generator_results[gen] = {"n": len(yy), "recall_1pct": r1, "recall_5pct": r5}

    with open(D("p5_task4_gagle_to_raid_transfer.json"), "w") as f:
        json.dump({
            "gagle_in_distribution": {"n": int(gagle_cell["n_test"]), "auroc": gagle_auroc, "tpr_1pct": gagle_tpr1, "tpr_5pct": gagle_tpr5},
            "raid_transfer_gagle_threshold": {"n": int(len(y)), "n_llm": int((y == 1).sum()), "n_human": int((y == 0).sum()),
                               "auroc": float(raid_auroc), "tpr_1pct": raid_tpr1, "tpr_5pct": raid_tpr5,
                               "realized_fpr_1pct": fpr1_realized, "realized_fpr_5pct": fpr5_realized,
                               "note": "MISCALIBRATED on RAID -- realized FPR far exceeds target, see raid_native_threshold for the honest number"},
            "raid_native_threshold_PRIMARY": {"n_test": int(len(y)), "auroc": float(raid_auroc), "auroc_ci": [auroc_lo, auroc_hi], "auroc_ci_n_valid_folds": auroc_k,
                               "tpr_1pct": raid_native_tpr1, "tpr_1pct_ci": [ntpr1_lo, ntpr1_hi], "tpr_1pct_ci_n_valid_folds": ntpr1_k,
                               "tpr_5pct": raid_native_tpr5, "tpr_5pct_ci": [ntpr5_lo, ntpr5_hi], "tpr_5pct_ci_n_valid_folds": ntpr5_k,
                               "note": "headline number -- threshold re-derived on RAID's own human population to actually hit 1%/5% FPR. "
                                       "auroc is IDENTICAL to raid_transfer_gagle_threshold.auroc by construction -- AUROC is a "
                                       "ranking metric, threshold-independent, so it does not change between the two blocks; only "
                                       "the TPR@FPR numbers (which depend on the threshold) differ between GAGLE-threshold and RAID-native."},
            "drop_vs_gagle": {"auroc": float(gagle_auroc - raid_auroc), "tpr_1pct": float(gagle_tpr1 - raid_tpr1),
                               "tpr_5pct": float(gagle_tpr5 - raid_tpr5)},
            "by_domain": domain_results, "by_generator": generator_results,
        }, f, indent=2)
    print("\n[task4] written to data/p5_task4_gagle_to_raid_transfer.json")


if __name__ == "__main__":
    main()
