#!/usr/bin/env python3
"""P5 task 3 (plan doc's own T3, planned sequence: task 4 -> task 6
-> task 3): multiclass over GAGLE's 8 prompt types (continue_it, continue_pt,
cot, excerpt, keywords, short_keywords, summary, summary_keywords) -- tests
whether fractal/stat/style structure encodes prompting regime, on LLM rows
only (human text has no prompt_type in this sense).

Reuses P3's existing 85-feature set (`load_matrix`) and its frozen
`random/random` group-disjoint split (`splits_v1.json`/`build_folds`)
directly -- no new split needed, unlike task 6's RAID split which had no
existing frozen split to reuse.

Pre-training safety check (run and recorded in the ledger before this
script existed, not assumed): the project notes documents that 4 of 8 prompt types
were EXPECTED to mostly/entirely fail the Family B alignment gate, which
would make this task partly solvable via missingness alone rather than real
fractal signal -- a severe, near-definitional leak for a task that predicts
prompt type itself, a different risk than P3's binary task ever had. Checked
empirically: any-NaN rate by prompt_type clusters tightly at 62.1%-67.6%,
not the near-0/near-100% pattern that would signal a leak -- safe to proceed
with the same complete-case policy this project uses everywhere else.
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
    already the discrete class prediction (multiclass macro-F1 has no single
    continuous score to threshold)."""
    return f1_score(y_true, y_pred, average="macro")


def main():
    from fractalsig.detection.p3_train_eval import load_matrix, build_folds

    df, groups = load_matrix()
    feats_all = groups["fractal"] + groups["stat"] + groups["style"]
    print(f"[task3] {len(feats_all)} features", flush=True)

    raw = pd.read_parquet(os.path.join(HERE, "feature_store", "v1_gagle.parquet"))[["id", "prompt_type"]]
    df = df.merge(raw, on="id", how="left")

    splits = json.load(open(D("splits_v1.json")))
    folds = {f"{s}/{f}": (tr, va, te) for s, f, tr, va, te in build_folds(df, splits)}
    tr, va, te = folds["random/random"]

    llm_mask = (df["Label"] != "Human").to_numpy()
    tr, va, te = tr & llm_mask, va & llm_mask, te & llm_mask
    print(f"[task3] LLM-only, random/random fold: train={tr.sum()} val={va.sum()} test={te.sum()}", flush=True)

    labels = sorted(df.loc[llm_mask, "prompt_type"].dropna().unique())
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y = df["prompt_type"].map(label_to_idx).to_numpy()
    X = df[feats_all].to_numpy(dtype=float)

    def build(mask):
        Xs, ys = X[mask], y[mask]
        m = ~np.isnan(Xs).any(axis=1)
        return Xs[m], ys[m], int(m.sum())

    Xtr, ytr, n_tr = build(tr)
    Xva, yva, n_va = build(va)
    Xte, yte, n_te = build(te)
    print(f"[task3] after NaN drop (complete-case): train={n_tr} val={n_va} test={n_te}", flush=True)

    clf = lgb.LGBMClassifier(objective="multiclass", num_class=len(labels), n_estimators=300,
                              learning_rate=0.05, max_depth=-1, num_leaves=63,
                              random_state=SEED, verbosity=-1, class_weight="balanced")
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(30, verbose=False)])

    pred_te = clf.predict(Xte)
    macro_f1 = f1_score(yte, pred_te, average="macro")
    print(f"\n[task3] TEST macro-F1 = {macro_f1:.4f}  (n_test={len(yte)}, 8-class chance ~12.5%)", flush=True)

    from fractalsig.detection.metrics import bootstrap_ci
    f1_pt, f1_lo, f1_hi, f1_k = bootstrap_ci(_macro_f1_multiclass, yte, pred_te, stratified=False)
    print(f"[task3] 95% CI (unstratified bootstrap, n_boot=2000): "
          f"macro-F1=[{f1_lo:.4f},{f1_hi:.4f}]  n_valid_folds={f1_k}/2000", flush=True)

    report = classification_report(yte, pred_te, target_names=labels, output_dict=True, zero_division=0)
    print(f"\n{'prompt_type':20}{'precision':>11}{'recall':>9}{'f1':>9}{'n':>8}")
    for l in labels:
        r = report[l]
        print(f"{l:20}{r['precision']:11.4f}{r['recall']:9.4f}{r['f1-score']:9.4f}{int(r['support']):8}")

    cm = confusion_matrix(yte, pred_te, labels=list(range(len(labels))))
    print(f"\nconfusion matrix (rows=true, cols=predicted), labels={labels}")
    print(cm)

    confusions = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], labels[i], labels[j]))
    confusions.sort(reverse=True)
    print(f"\ntop confused pairs (true -> predicted, count):")
    for count, true_l, pred_l in confusions[:15]:
        print(f"  {true_l:20} -> {pred_l:20}  {count}")

    feat_imp = sorted(zip(feats_all, clf.feature_importances_), key=lambda x: -x[1])
    print(f"\ntop 15 features by importance:")
    for name, imp in feat_imp[:15]:
        print(f"  {name:30}{imp}")

    # fractal-only arm, for the same reuse-arm_features pattern this project
    # uses everywhere -- does fractal structure ALONE encode prompting regime?
    from fractalsig.detection.p3_train_eval import arm_features
    fractal_names = arm_features("fractal_only", groups)
    Xf = df[fractal_names].to_numpy(dtype=float)
    def build_f(mask):
        Xs, ys = Xf[mask], y[mask]
        m = ~np.isnan(Xs).any(axis=1)
        return Xs[m], ys[m]
    Xtr_f, ytr_f = build_f(tr)
    Xva_f, yva_f = build_f(va)
    Xte_f, yte_f = build_f(te)
    n_te_f = len(yte_f)
    clf_f = lgb.LGBMClassifier(objective="multiclass", num_class=len(labels), n_estimators=300,
                                learning_rate=0.05, max_depth=-1, num_leaves=63,
                                random_state=SEED, verbosity=-1, class_weight="balanced")
    clf_f.fit(Xtr_f, ytr_f, eval_set=[(Xva_f, yva_f)], callbacks=[lgb.early_stopping(30, verbose=False)])
    pred_te_f = clf_f.predict(Xte_f)
    macro_f1_fractal = f1_score(yte_f, pred_te_f, average="macro")
    print(f"\n[task3] fractal_only arm ({len(fractal_names)} features): "
          f"TEST macro-F1 = {macro_f1_fractal:.4f}  (n_test={len(yte_f)})", flush=True)
    f1f_pt, f1f_lo, f1f_hi, f1f_k = bootstrap_ci(_macro_f1_multiclass, yte_f, pred_te_f, stratified=False)
    print(f"[task3] fractal_only 95% CI (unstratified bootstrap, n_boot=2000): "
          f"macro-F1=[{f1f_lo:.4f},{f1f_hi:.4f}]  n_valid_folds={f1f_k}/2000", flush=True)

    with open(D("p5_task3_prompt_type_id.json"), "w") as f:
        json.dump({
            "n_train": n_tr, "n_val": n_va, "n_test": n_te,
            "labels": labels, "macro_f1_test_all_features": float(macro_f1),
            "macro_f1_all_features_ci": [f1_lo, f1_hi], "macro_f1_all_features_ci_n_valid_folds": f1_k,
            "macro_f1_test_fractal_only": float(macro_f1_fractal),
            "macro_f1_fractal_only_ci": [f1f_lo, f1f_hi], "macro_f1_fractal_only_ci_n_valid_folds": f1f_k,
            "n_test_fractal_only": int(n_te_f),
            "per_class": {l: {"precision": report[l]["precision"], "recall": report[l]["recall"],
                                "f1": report[l]["f1-score"], "n": int(report[l]["support"])} for l in labels},
            "confusion_matrix": cm.tolist(),
            "top_confusions": [{"true": t, "pred": p, "count": int(c)} for c, t, p in confusions[:15]],
            "top_features": [{"name": n, "importance": int(i)} for n, i in feat_imp[:15]],
        }, f, indent=2)
    print("\n[task3] written to data/p5_task3_prompt_type_id.json")


if __name__ == "__main__":
    main()
