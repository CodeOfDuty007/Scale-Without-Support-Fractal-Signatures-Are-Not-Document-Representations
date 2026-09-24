#!/usr/bin/env python3
"""
T3 -- P3 (prompt-type ID) under population control.

Runs, in order, the four sub-tasks of the population-control analysis:
  T3.1  Population-only control arm: predict prompt_type from
        (generator=Label, domain, temp) alone. Ceiling attributable to
        population structure alone.
  T3.2  Balance table: cross-tab surviving complete-case TEST rows by
        prompt_type x generator x domain x temp; chi2 + max cell-proportion
        deviation from the factorial expectation.
  T3.3  Group-mean removal: subtract the (generator, domain, temp) cell mean
        (computed on TRAIN rows only, applied to train/val/test -- a
        deliberate anti-leakage choice not specified explicitly by the plan)
        from each of the 54 fractal columns; re-run both arms (all-features,
        fractal-only).
  T3.4  Within-stratum repeats: re-run the 8-class task INSIDE each of the
        30 (generator, domain, temp) cells; report the macro-F1 DISTRIBUTION
        across cells for both arms, never a point.

Reuses p3_train_eval.load_matrix/build_folds/arm_features and
eval.metrics.bootstrap_ci unmodified -- the same code path
p5_task3_prompt_type_id.py already uses -- rather than re-deriving the
feature matrix or the frozen random/random split.

First reproduces the frozen p5_task3_prompt_type_id.json numbers
(0.5816 all-features, 0.1981 fractal-only) as an acceptance gate before
running anything new: if the reproduction fails, the harness has diverged
from the frozen result and nothing below it is trustworthy.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import f1_score
import lightgbm as lgb

from fractalsig import paths as _paths  # noqa: E402

PAPER = _paths.WORKSPACE_STR
from fractalsig.detection.p3_train_eval import load_matrix, build_folds, arm_features  # noqa: E402
from fractalsig.detection.metrics import bootstrap_ci  # noqa: E402

OUT_DIR = _paths.STATS_DIR_STR
SEED = 20260828


def macro_f1_multiclass(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


def fit_eval(Xtr, ytr, Xva, yva, Xte, yte, n_classes, seed=SEED):
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=n_classes, n_estimators=300,
                              learning_rate=0.05, max_depth=-1, num_leaves=63,
                              random_state=seed, verbosity=-1, class_weight="balanced")
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(30, verbose=False)])
    pred = clf.predict(Xte)
    return float(f1_score(yte, pred, average="macro")), pred, clf


def build_xy(X, y, mask):
    Xs, ys = X[mask], y[mask]
    m = ~np.isnan(Xs).any(axis=1)
    return Xs[m], ys[m], m


def main():
    t0 = time.time()
    df, groups = load_matrix()
    feats_all = groups["fractal"] + groups["stat"] + groups["style"]
    fractal_names = arm_features("fractal_only", groups)
    raw = pd.read_parquet(os.path.join(PAPER, "feature_store", "v1_gagle.parquet"))[["id", "prompt_type", "temp"]]
    df = df.merge(raw, on="id", how="left")

    splits = json.load(open(os.path.join(PAPER, "data", "splits_v1.json")))
    folds = {f"{s}/{f}": (tr, va, te) for s, f, tr, va, te in build_folds(df, splits)}
    tr, va, te = folds["random/random"]
    llm_mask = (df["Label"] != "Human").to_numpy()
    tr, va, te = tr & llm_mask, va & llm_mask, te & llm_mask

    labels = sorted(df.loc[llm_mask, "prompt_type"].dropna().unique())
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y = df["prompt_type"].map(label_to_idx).to_numpy()
    X_all = df[feats_all].to_numpy(dtype=float)
    X_frac = df[fractal_names].to_numpy(dtype=float)

    # ---------------------------------------------------------------
    # ACCEPTANCE GATE: reproduce the frozen p5_task3 numbers first.
    # ---------------------------------------------------------------
    Xtr, ytr, _ = build_xy(X_all, y, tr)
    Xva, yva, _ = build_xy(X_all, y, va)
    Xte, yte, mte_all = build_xy(X_all, y, te)
    f1_all, pred_all, _ = fit_eval(Xtr, ytr, Xva, yva, Xte, yte, len(labels))
    Xtr_f, ytr_f, _ = build_xy(X_frac, y, tr)
    Xva_f, yva_f, _ = build_xy(X_frac, y, va)
    Xte_f, yte_f, mte_f = build_xy(X_frac, y, te)
    f1_frac, pred_frac, _ = fit_eval(Xtr_f, ytr_f, Xva_f, yva_f, Xte_f, yte_f, len(labels))

    frozen = json.load(open(os.path.join(PAPER, "data", "p5_task3_prompt_type_id.json")))
    # Acceptance is against the plan's own documented CI (LightGBM's multi-
    # threaded histogram construction is not bit-deterministic across
    # versions/thread counts even in the ORIGINAL pipeline -- no determinism
    # flags are set there either -- so a tight point-tolerance gate would
    # reject a legitimate reproduction. Falling inside the frozen 95% CI is
    # the plan's own stated bar ("reproduces 0.5816 [0.5747, 0.5885]").
    frozen_all_ci = frozen["macro_f1_all_features_ci"]
    frozen_frac_ci = frozen["macro_f1_fractal_only_ci"]
    gate = {
        "all_features_reproduced": f1_all, "all_features_frozen": frozen["macro_f1_test_all_features"],
        "all_features_frozen_ci": frozen_all_ci,
        "fractal_only_reproduced": f1_frac, "fractal_only_frozen": frozen["macro_f1_test_fractal_only"],
        "fractal_only_frozen_ci": frozen_frac_ci,
        "all_features_match": bool(frozen_all_ci[0] <= f1_all <= frozen_all_ci[1]),
        "fractal_only_match": bool(frozen_frac_ci[0] <= f1_frac <= frozen_frac_ci[1]),
    }
    print("[gate]", json.dumps(gate, indent=2), flush=True)
    if not (gate["all_features_match"] and gate["fractal_only_match"]):
        print("[FATAL] reproduction gate failed -- stopping before T3.1-T3.4.", flush=True)
        json.dump({"gate": gate}, open(os.path.join(OUT_DIR, "t3_ABORTED_gate_failure.json"), "w"), indent=2)
        sys.exit(1)

    f1_all_pt, f1_all_lo, f1_all_hi, f1_all_k = bootstrap_ci(macro_f1_multiclass, yte, pred_all, stratified=False)
    f1_frac_pt, f1_frac_lo, f1_frac_hi, f1_frac_k = bootstrap_ci(macro_f1_multiclass, yte_f, pred_frac, stratified=False)

    # ---------------------------------------------------------------
    # T3.1 -- population-only control arm
    # ---------------------------------------------------------------
    print("\n[T3.1] population-only control arm", flush=True)
    pop_cat = pd.DataFrame({
        "Label": df["Label"].astype("category").cat.codes,
        "domain": df["domain"].astype("category").cat.codes,
        "temp": df["temp"].astype("category").cat.codes,
    }).to_numpy(dtype=float)
    Xtr_p, ytr_p, _ = build_xy(pop_cat, y, tr)
    Xva_p, yva_p, _ = build_xy(pop_cat, y, va)
    Xte_p, yte_p, _ = build_xy(pop_cat, y, te)
    f1_pop, pred_pop, _ = fit_eval(Xtr_p, ytr_p, Xva_p, yva_p, Xte_p, yte_p, len(labels))
    f1_pop_pt, f1_pop_lo, f1_pop_hi, f1_pop_k = bootstrap_ci(macro_f1_multiclass, yte_p, pred_pop, stratified=False)
    print(f"[T3.1] macro-F1={f1_pop:.4f}  95% CI=[{f1_pop_lo:.4f},{f1_pop_hi:.4f}]  "
          f"n_test={len(yte_p)}  chance={1/len(labels):.4f}", flush=True)

    # ---------------------------------------------------------------
    # T3.2 -- balance table on surviving complete-case TEST rows
    # ---------------------------------------------------------------
    print("\n[T3.2] balance table", flush=True)
    te_idx = np.flatnonzero(te)
    te_idx_survive = te_idx[mte_all]  # complete-case (all-features arm) survivors
    bt = df.iloc[te_idx_survive][["prompt_type", "Label", "domain", "temp"]].copy()
    ct = pd.crosstab([bt["Label"], bt["domain"], bt["temp"]], bt["prompt_type"])
    n_total = ct.values.sum()
    n_cells, n_types = ct.shape
    expected_uniform = n_total / (n_cells * n_types)
    chi2, chi2_p, dof, expected = chi2_contingency(ct)
    obs_frac = ct.values / n_total
    exp_frac = expected / n_total
    max_dev = float(np.max(np.abs(obs_frac - exp_frac)))
    print(f"[T3.2] {n_cells} observed (generator,domain,temp) cells x {n_types} prompt types, "
          f"n={n_total}, chi2={chi2:.2f} (dof={dof}, p={chi2_p:.3g}), "
          f"max cell-proportion deviation from factorial expectation={max_dev:.5f}", flush=True)

    # ---------------------------------------------------------------
    # T3.3 -- group-mean removal (54 fractal columns), TRAIN-only means
    # ---------------------------------------------------------------
    print("\n[T3.3] group-mean removal", flush=True)
    strata = list(zip(df["Label"], df["domain"], df["temp"]))
    strata_arr = np.array(strata, dtype=object)
    X_frac_demeaned = X_frac.copy()
    tr_idx = np.flatnonzero(tr)
    from collections import defaultdict
    sums = defaultdict(lambda: np.zeros(X_frac.shape[1]))
    counts = defaultdict(int)
    for i in tr_idx:
        s = strata[i]
        row = X_frac[i]
        if not np.isnan(row).any():
            sums[s] += row
            counts[s] += 1
    means = {s: sums[s] / counts[s] for s in sums if counts[s] > 0}
    global_mean = np.nanmean(X_frac[tr_idx], axis=0)
    for i in range(len(df)):
        s = strata[i]
        X_frac_demeaned[i] = X_frac[i] - means.get(s, global_mean)

    Xtr_fd, ytr_fd, _ = build_xy(X_frac_demeaned, y, tr)
    Xva_fd, yva_fd, _ = build_xy(X_frac_demeaned, y, va)
    Xte_fd, yte_fd, _ = build_xy(X_frac_demeaned, y, te)
    f1_frac_demeaned, pred_frac_d, _ = fit_eval(Xtr_fd, ytr_fd, Xva_fd, yva_fd, Xte_fd, yte_fd, len(labels))
    f1_fd_pt, f1_fd_lo, f1_fd_hi, f1_fd_k = bootstrap_ci(macro_f1_multiclass, yte_fd, pred_frac_d, stratified=False)
    print(f"[T3.3] fractal_only AFTER group-mean removal: macro-F1={f1_frac_demeaned:.4f} "
          f"(vs {f1_frac:.4f} un-demeaned)  95% CI=[{f1_fd_lo:.4f},{f1_fd_hi:.4f}]", flush=True)

    # all-features arm with fractal columns demeaned (rest unchanged), for completeness
    frac_idx = [feats_all.index(c) for c in fractal_names]
    X_all_demeaned = X_all.copy()
    X_all_demeaned[:, frac_idx] = X_frac_demeaned
    Xtr_ad, ytr_ad, _ = build_xy(X_all_demeaned, y, tr)
    Xva_ad, yva_ad, _ = build_xy(X_all_demeaned, y, va)
    Xte_ad, yte_ad, _ = build_xy(X_all_demeaned, y, te)
    f1_all_demeaned, pred_all_d, _ = fit_eval(Xtr_ad, ytr_ad, Xva_ad, yva_ad, Xte_ad, yte_ad, len(labels))
    f1_ad_pt, f1_ad_lo, f1_ad_hi, f1_ad_k = bootstrap_ci(macro_f1_multiclass, yte_ad, pred_all_d, stratified=False)
    print(f"[T3.3] all-features AFTER group-mean removal (fractal cols only): "
          f"macro-F1={f1_all_demeaned:.4f} (vs {f1_all:.4f} un-demeaned)", flush=True)

    # ---------------------------------------------------------------
    # T3.4 -- within-stratum repeats over the 30 (generator,domain,temp) cells
    # ---------------------------------------------------------------
    print("\n[T3.4] within-stratum repeats", flush=True)
    cells = sorted(set(strata[i] for i in np.flatnonzero(llm_mask)))
    print(f"[T3.4] {len(cells)} cells", flush=True)
    per_cell_results = []
    for cell in cells:
        in_cell = np.array([s == cell for s in strata])
        tr_c, va_c, te_c = tr & in_cell, va & in_cell, te & in_cell
        row = {"cell": {"generator": cell[0], "domain": cell[1], "temp": cell[2]},
               "n_train": int(tr_c.sum()), "n_val": int(va_c.sum()), "n_test": int(te_c.sum())}
        for arm_name, Xarm in (("all", X_all), ("fractal_only", X_frac)):
            Xtr_c, ytr_c, _ = build_xy(Xarm, y, tr_c)
            Xva_c, yva_c, _ = build_xy(Xarm, y, va_c)
            Xte_c, yte_c, _ = build_xy(Xarm, y, te_c)
            n_classes_c = len(np.unique(ytr_c)) if len(ytr_c) else 0
            if len(Xtr_c) < 20 or len(Xte_c) < 5 or n_classes_c < 2 or len(Xva_c) < 5:
                row[f"macro_f1_{arm_name}"] = None
                row[f"note_{arm_name}"] = "insufficient rows/classes in this cell"
                continue
            try:
                f1_c, _, _ = fit_eval(Xtr_c, ytr_c, Xva_c, yva_c, Xte_c, yte_c, len(labels), seed=SEED)
                row[f"macro_f1_{arm_name}"] = float(f1_c)
            except Exception as e:
                row[f"macro_f1_{arm_name}"] = None
                row[f"note_{arm_name}"] = f"failed: {e}"
        per_cell_results.append(row)
        print(f"  {cell}: n_test={row['n_test']} all={row.get('macro_f1_all')} "
              f"fractal_only={row.get('macro_f1_fractal_only')}", flush=True)

    def dist(key):
        vals = [r[key] for r in per_cell_results if r.get(key) is not None]
        if not vals:
            return {}
        vals = np.array(vals)
        return {"n_cells_valid": int(len(vals)), "mean": float(vals.mean()), "std": float(vals.std()),
                "min": float(vals.min()), "p25": float(np.percentile(vals, 25)),
                "median": float(np.median(vals)), "p75": float(np.percentile(vals, 75)),
                "max": float(vals.max())}

    dist_all = dist("macro_f1_all")
    dist_frac = dist("macro_f1_fractal_only")
    print(f"\n[T3.4] distribution (all-features arm): {dist_all}", flush=True)
    print(f"[T3.4] distribution (fractal_only arm): {dist_frac}", flush=True)

    # ---------------------------------------------------------------
    # write output
    # ---------------------------------------------------------------
    out = {
        "runtime_sec": time.time() - t0,
        "gate": gate,
        "headline": {
            "all_features": {"macro_f1": f1_all, "ci": [f1_all_lo, f1_all_hi], "n_test": len(yte)},
            "fractal_only": {"macro_f1": f1_frac, "ci": [f1_frac_lo, f1_frac_hi], "n_test": len(yte_f)},
        },
        "t3_1_population_only_control": {
            "macro_f1": f1_pop, "ci": [f1_pop_lo, f1_pop_hi], "n_test": len(yte_p),
            "n_train": len(ytr_p), "chance_level": 1 / len(labels), "n_classes": len(labels),
            "features_used": ["Label(generator)", "domain", "temp"],
        },
        "t3_2_balance_table": {
            "n_observed_cells": int(n_cells), "n_prompt_types": int(n_types), "n_total_test_rows": int(n_total),
            "chi2": float(chi2), "dof": int(dof), "p_value": float(chi2_p),
            "max_cell_proportion_deviation": max_dev,
            "expected_uniform_fraction_per_cell_type": float(expected_uniform / n_total),
        },
        "t3_3_group_mean_removal": {
            "fractal_only_before": f1_frac, "fractal_only_after": f1_frac_demeaned,
            "fractal_only_after_ci": [f1_fd_lo, f1_fd_hi],
            "all_features_before": f1_all, "all_features_after_fractal_demeaned": f1_all_demeaned,
            "all_features_after_ci": [f1_ad_lo, f1_ad_hi],
            "means_computed_from": "TRAIN split only, applied to train/val/test",
            "interpretation": ("fell toward chance -> residual was population structure (paper strengthens); "
                                "held near un-demeaned value -> fractal family carries genuine "
                                "document-level prompt-type signal (P3 claim must soften further)"),
        },
        "t3_4_within_stratum": {
            "n_cells": len(cells), "per_cell": per_cell_results,
            "distribution_all_features": dist_all, "distribution_fractal_only": dist_frac,
        },
        "direction_of_risk_for_rebuttal": (
            "Residual population structure in the un-demeaned fractal_only=0.1981 result means "
            "that number is PARTLY leakage -- true Var(delta) is smaller than 0.1981 implies, which "
            "STRENGTHENS the paper's conclusion. The number genuinely at risk is the 0.5816 "
            "all-features reference and the 'about a sixth of the distance' framing built on it."
        ),
    }
    out_path = os.path.join(OUT_DIR, "t3_population_control.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n[written] {out_path}", flush=True)
    print(f"[T3] TOTAL {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
