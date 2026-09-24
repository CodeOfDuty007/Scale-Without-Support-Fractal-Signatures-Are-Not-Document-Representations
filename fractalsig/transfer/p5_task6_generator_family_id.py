#!/usr/bin/env python3
"""P5 task 6 (new task, added 2026-08-27 -- the plan doc's
own T2 is GAGLE-only 2-class, gemma-2b vs mistral-7b; this is RAID's own
11-class generator-family ID, added because the stratification
directive implied it and no existing task covered it): multiclass
classification over RAID's 11 LLM generators (human excluded -- not one of
this task's classes), on the RAID feature store (`p5_raid_features.jsonl`,
68,212-row complete-case population, all 85 features from build_feature_row).

GROUP-DISJOINT SPLIT, not naive random -- found via a targeted lookup
against RAID's raw train.csv (not kept by ingest_raid.py, added back for
this task specifically): 13,144 unique source_ids across the 68,212-row
population, avg ~5.2 generator/attack variants sharing the SAME underlying
source article. Splitting a source's rows across train/test would let a
classifier memorize source-article content as a shortcut instead of
learning genuine generator style -- exactly the leakage class GAGLE's own
splits.py was built to prevent for the binary task. Same discipline applied
here for the multiclass one.
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import lightgbm as lgb

HERE = _paths.WORKSPACE_STR
D = lambda *p: os.path.join(HERE, "data", *p)
SEED = 20260828


def _macro_f1_multiclass(y_true, y_pred):
    """metric_fn(y_true, y_score) shape for bootstrap_ci -- 'y_score' here is
    already the discrete class prediction, not a continuous score (multiclass
    macro-F1 has no single-column score to threshold)."""
    return f1_score(y_true, y_pred, average="macro")


def main():
    rows = [json.loads(l) for l in open(D("p5_raid_features.jsonl"))]
    df = pd.DataFrame(rows)
    df = df[df["Label"] != "human"].reset_index(drop=True)
    print(f"[task6] {len(df)} LLM-only rows, {df['Label'].nunique()} generators", flush=True)
    print(f"[task6] class counts: {dict(df['Label'].value_counts())}", flush=True)

    source_ids = json.load(open(D("p5_raid_source_ids.json")))
    df["source_id"] = df["id"].apply(lambda i: source_ids.get(i[len("raid_"):], {}).get("source_id"))
    n_missing_source = df["source_id"].isna().sum()
    print(f"[task6] {n_missing_source} rows missing a source_id lookup (dropped)", flush=True)
    df = df.dropna(subset=["source_id"]).reset_index(drop=True)

    exclude = {"Label", "domain", "attack", "id", "source_id"}
    # Also exclude any boolean-dtype column (provenance/metadata flags, e.g.
    # ZWSP_Present_In_Source_Text added 2026-08-29) -- df[feature_names]
    # .to_numpy(dtype=float) below would otherwise silently cast a flag to
    # 0.0/1.0 and train on it as if it were a real feature, with no error.
    feature_names = [c for c in df.columns
                     if c not in exclude and not pd.api.types.is_bool_dtype(df[c])]
    print(f"[task6] {len(feature_names)} features", flush=True)

    # --- group-disjoint split on source_id ---
    rng = np.random.RandomState(SEED)
    groups = df["source_id"].unique()
    rng.shuffle(groups)
    n = len(groups)
    n_tr, n_va = int(n * 0.7), int(n * 0.15)
    tr_groups = set(groups[:n_tr])
    va_groups = set(groups[n_tr:n_tr + n_va])
    te_groups = set(groups[n_tr + n_va:])

    tr_mask = df["source_id"].isin(tr_groups)
    va_mask = df["source_id"].isin(va_groups)
    te_mask = df["source_id"].isin(te_groups)
    assert not (tr_groups & va_groups) and not (tr_groups & te_groups) and not (va_groups & te_groups), \
        "group split is not disjoint"
    print(f"[task6] split: train={tr_mask.sum()} ({len(tr_groups)} groups)  "
          f"val={va_mask.sum()} ({len(va_groups)} groups)  test={te_mask.sum()} ({len(te_groups)} groups)", flush=True)

    labels = sorted(df["Label"].unique())
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y = df["Label"].map(label_to_idx).to_numpy()
    X = df[feature_names].to_numpy(dtype=float)

    Xtr, ytr = X[tr_mask.to_numpy()], y[tr_mask.to_numpy()]
    Xva, yva = X[va_mask.to_numpy()], y[va_mask.to_numpy()]
    Xte, yte = X[te_mask.to_numpy()], y[te_mask.to_numpy()]

    # complete-case within the split (a handful of rows may have residual NaN
    # in a specific feature even after the F-5(a) rescore-level complete-case
    # filter, e.g. skew/kurtosis undefined for very short docs -- same
    # treatment as every other detector in this project, drop don't impute)
    def clean(Xs, ys):
        m = ~np.isnan(Xs).any(axis=1)
        return Xs[m], ys[m], m.sum()
    Xtr, ytr, n_tr_clean = clean(Xtr, ytr)
    Xva, yva, n_va_clean = clean(Xva, yva)
    Xte, yte, n_te_clean = clean(Xte, yte)
    print(f"[task6] after NaN drop: train={n_tr_clean} val={n_va_clean} test={n_te_clean}", flush=True)

    clf = lgb.LGBMClassifier(objective="multiclass", num_class=len(labels), n_estimators=300,
                              learning_rate=0.05, max_depth=-1, num_leaves=63,
                              random_state=SEED, verbosity=-1, class_weight="balanced")
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(30, verbose=False)])

    pred_te = clf.predict(Xte)
    macro_f1 = f1_score(yte, pred_te, average="macro")
    print(f"\n[task6] TEST macro-F1 = {macro_f1:.4f}  (n_test={len(yte)})", flush=True)

    # D-3/D-12 CI convention: multiclass task -> UNSTRATIFIED bootstrap
    # (bootstrap_ci's stratified=True path is binary-specific, y_true==1 vs
    # !=1 -- doesn't generalize to 11 classes). With n_test=1,406 across 11
    # classes, the smallest class (gpt3) has ~34 test rows, so report
    # n_valid_folds alongside the interval -- an unstratified resample can
    # occasionally drop a class entirely, and bootstrap_ci skips those folds
    # rather than counting them, per its own docstring.
    from fractalsig.detection.metrics import bootstrap_ci
    f1_pt, f1_lo, f1_hi, f1_k = bootstrap_ci(_macro_f1_multiclass, yte, pred_te, stratified=False)
    print(f"[task6] 95% CI (unstratified bootstrap, n_boot=2000): "
          f"macro-F1=[{f1_lo:.4f},{f1_hi:.4f}]  n_valid_folds={f1_k}/2000", flush=True)
    print(f"[task6] reference: paper's own GAGLE 2-class generator-ID RF baseline ~97.8% "
          f"(NOT directly comparable -- 11-class vs 2-class, RAID vs GAGLE, different task shape)", flush=True)

    report = classification_report(yte, pred_te, target_names=labels, output_dict=True, zero_division=0)
    print(f"\n{'generator':16}{'precision':>11}{'recall':>9}{'f1':>9}{'n':>8}")
    for l in labels:
        r = report[l]
        print(f"{l:16}{r['precision']:11.4f}{r['recall']:9.4f}{r['f1-score']:9.4f}{int(r['support']):8}")

    cm = confusion_matrix(yte, pred_te, labels=list(range(len(labels))))
    print(f"\nconfusion matrix (rows=true, cols=predicted), labels={labels}")
    print(cm)

    # which pairs are most confused, since that's the informative part of an 11-class result
    confusions = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], labels[i], labels[j]))
    confusions.sort(reverse=True)
    print(f"\ntop confused pairs (true -> predicted, count):")
    for count, true_l, pred_l in confusions[:15]:
        print(f"  {true_l:16} -> {pred_l:16}  {count}")

    feat_imp = sorted(zip(feature_names, clf.feature_importances_), key=lambda x: -x[1])
    print(f"\ntop 15 features by importance:")
    for name, imp in feat_imp[:15]:
        print(f"  {name:30}{imp}")

    with open(D("p5_task6_generator_family_id.json"), "w") as f:
        json.dump({
            "n_train": int(n_tr_clean), "n_val": int(n_va_clean), "n_test": int(n_te_clean),
            "labels": labels, "macro_f1_test": float(macro_f1),
            "macro_f1_ci": [f1_lo, f1_hi], "macro_f1_ci_n_valid_folds": f1_k,
            "per_class": {l: {"precision": report[l]["precision"], "recall": report[l]["recall"],
                                "f1": report[l]["f1-score"], "n": int(report[l]["support"])} for l in labels},
            "confusion_matrix": cm.tolist(),
            "top_confusions": [{"true": t, "pred": p, "count": int(c)} for c, t, p in confusions[:15]],
            "top_features": [{"name": n, "importance": int(i)} for n, i in feat_imp[:15]],
        }, f, indent=2)
    print("\n[task6] written to data/p5_task6_generator_family_id.json")


if __name__ == "__main__":
    main()
