#!/usr/bin/env python3
"""
T5 -- RAID group-mean replacement/subtraction test (Part 3 SS2.3's "very
clean test of the proposed mechanism").

Uses the ALREADY-PERSISTED, frozen transfer model
(feature_store/p3_refit_models/full_random_random_all_lgbm.joblib) with NO
retraining -- exactly p5_task4_gagle_to_raid_transfer.py's own reuse
discipline -- and the same complete-case RAID population (85-feature schema,
NaN-drop across all features) that produced the frozen AUROC=0.8032,
n=22,041 result. First reproduces that baseline as an acceptance gate.

Then, for the 54 fractal-manifest columns only:
  REPLACEMENT: substitute each document's fractal features with its
    (Label=generator, domain)-conditioned mean, re-score.
    If AUROC recovers most of the +0.0240 all_85 vs all_minus_fractal_31feat
    gap (data/fractalscope_check_raid_intersection.json), population-mean
    information is doing most of the work.
  SUBTRACTION: x' = x - group_mean(x), re-score.
    If the AUROC gain over the non-fractal baseline (0.7792) disappears,
    confirmed from the other side.

Group means are computed on the SAME complete-case RAID population being
scored (no train/test leakage concern here: this is a frozen, already-
trained model used purely as a fixed scoring function at inference time --
not a re-fit -- so there is no leakage channel for a diagnostic transform of
its inputs).
"""
import json
import os
import sys
import time
from collections import defaultdict

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from fractalsig import paths as _paths  # noqa: E402

PAPER = _paths.WORKSPACE_STR
D = lambda *p: os.path.join(PAPER, "data", *p)
OUT_DIR = _paths.STATS_DIR_STR


def main():
    t0 = time.time()
    bundle = joblib.load(os.path.join(PAPER, "feature_store", "p3_refit_models",
                                        "full_random_random_all_lgbm.joblib"))
    cal_model, feature_names = bundle["model"], bundle["feature_names"]
    pos = int(np.where(cal_model.classes_ == 1)[0][0])

    man = json.load(open(os.path.join(PAPER, "feature_store", "p3_feature_manifest.json")))
    fractal_cols = [c for c in man["features"] if c in feature_names]
    fractal_idx = [feature_names.index(c) for c in fractal_cols]
    print(f"[t5] {len(fractal_cols)} fractal columns of {len(feature_names)} total features", flush=True)

    rows = [json.loads(l) for l in open(D("p5_raid_features.jsonl"))]
    y_all = np.array([0 if r["Label"] == "human" else 1 for r in rows])
    X_all = np.array([[r.get(n, np.nan) for n in feature_names] for r in rows], dtype=float)
    nan_mask = np.isnan(X_all).any(axis=1)
    X = X_all[~nan_mask]
    y = y_all[~nan_mask]
    rows_clean = [r for r, m in zip(rows, nan_mask) if not m]
    groups = np.array([(r["Label"], r["domain"]) for r in rows_clean], dtype=object)
    print(f"[t5] complete-case population: n={len(y)} (excluded {nan_mask.sum()} with a NaN feature)", flush=True)

    # ---- acceptance gate: reproduce frozen baseline ----
    p_baseline = cal_model.predict_proba(X)[:, pos]
    auroc_baseline = float(roc_auc_score(y, p_baseline))
    gate = {"n": len(y), "auroc_reproduced": auroc_baseline, "auroc_frozen": 0.8032112857989935,
            "match": bool(abs(auroc_baseline - 0.8032112857989935) < 1e-3) and len(y) == 22041}
    print(f"[gate] {gate}", flush=True)
    if not gate["match"]:
        print("[FATAL] T5 baseline reproduction gate failed -- stopping.", flush=True)
        json.dump({"gate": gate}, open(os.path.join(OUT_DIR, "t5_ABORTED_gate_failure.json"), "w"), indent=2)
        sys.exit(1)

    intersection = json.load(open(D("fractalscope_check_raid_intersection.json")))
    auroc_all85 = intersection["all_85feat"]["auroc"]
    auroc_minus_fractal = intersection["all_minus_fractal_31feat_same_rows"]["auroc"]
    gap = intersection["paired_bootstrap_auroc_diff_85_minus_31"]["mean"]
    print(f"[t5] reference gap (all_85 - all_minus_fractal_31, same rows): {gap:.4f} "
          f"({auroc_all85:.4f} vs {auroc_minus_fractal:.4f})", flush=True)

    # ---- group means on fractal columns, computed on this population ----
    sums = defaultdict(lambda: np.zeros(len(fractal_idx)))
    counts = defaultdict(int)
    for i in range(len(X)):
        g = tuple(groups[i])
        sums[g] += X[i, fractal_idx]
        counts[g] += 1
    group_mean = {g: sums[g] / counts[g] for g in sums}
    global_mean = X[:, fractal_idx].mean(axis=0)
    n_groups = len(group_mean)
    print(f"[t5] {n_groups} (generator,domain) groups", flush=True)

    # ---- REPLACEMENT ----
    X_replaced = X.copy()
    for i in range(len(X)):
        X_replaced[i, fractal_idx] = group_mean.get(tuple(groups[i]), global_mean)
    p_repl = cal_model.predict_proba(X_replaced)[:, pos]
    auroc_repl = float(roc_auc_score(y, p_repl))
    frac_of_gap_recovered = (auroc_repl - auroc_minus_fractal) / gap if gap else float("nan")
    print(f"[t5] REPLACEMENT (fractal cols -> group mean): AUROC={auroc_repl:.4f} "
          f"(baseline={auroc_baseline:.4f}, no-fractal={auroc_minus_fractal:.4f}); "
          f"fraction of +{gap:.4f} gap recovered by group means = {frac_of_gap_recovered:.3f}", flush=True)

    # ---- SUBTRACTION ----
    # x - group_mean + global_mean, NOT x - group_mean: the frozen LightGBM
    # was never trained on inputs centered near zero, so a bare x-group_mean
    # would hand it out-of-distribution values and confound "population
    # signal removed" with "model fed OOD inputs it's never seen". Adding
    # global_mean back keeps every document's fractal columns inside the
    # range the model was actually trained on, while still zeroing out the
    # (generator,domain)-specific offset -- review-identified fix.
    X_subtracted = X.copy()
    for i in range(len(X)):
        X_subtracted[i, fractal_idx] = X[i, fractal_idx] - group_mean.get(tuple(groups[i]), global_mean) + global_mean
    p_sub = cal_model.predict_proba(X_subtracted)[:, pos]
    auroc_sub = float(roc_auc_score(y, p_sub))
    frac_of_gap_remaining = (auroc_sub - auroc_minus_fractal) / gap if gap else float("nan")
    print(f"[t5] SUBTRACTION (fractal cols -> x - group mean): AUROC={auroc_sub:.4f}; "
          f"fraction of +{gap:.4f} gap REMAINING after removing group means = {frac_of_gap_remaining:.3f}", flush=True)

    out = {
        "runtime_sec": time.time() - t0,
        "gate": gate,
        "n_fractal_columns": len(fractal_cols),
        "n_groups_generator_domain": n_groups,
        "reference_gap_all85_minus_all_minus_fractal31": {
            "auroc_all85": auroc_all85, "auroc_minus_fractal": auroc_minus_fractal, "gap": gap,
        },
        "baseline_auroc_this_run": auroc_baseline,
        "replacement": {"auroc": auroc_repl, "fraction_of_gap_recovered": frac_of_gap_recovered,
                          "interpretation": "recovers most of the gap -> population-mean interpretation confirmed"},
        "subtraction": {"auroc": auroc_sub, "fraction_of_gap_remaining": frac_of_gap_remaining,
                          "interpretation": "gap disappears (fraction near 0) -> confirmed from the other side"},
    }
    out_path = os.path.join(OUT_DIR, "t5_raid_group_mean.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n[written] {out_path}", flush=True)
    print(f"[T5] TOTAL {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
