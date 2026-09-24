#!/usr/bin/env python3
"""Task 5 gap-fill (internal review, 2026-08-26): perplexity-threshold's
TPR@5%FPR was never tabulated anywhere in the report (only AUROC/TPR@1%FPR
were). Reuses part_d_paired_significance.py's exact sign-selection method
(sign chosen once globally on the random/val fold) so these numbers are
directly comparable to the existing AUROC/TPR@1% table -- not a new,
differently-derived baseline. Also recomputes AUROC/TPR@1% as a
cross-check against the report's existing numbers before trusting TPR@5%.
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
import sys

import numpy as np

HERE = _paths.WORKSPACE_STR

from fractalsig.detection.p3_train_eval import load_matrix, build_folds  # noqa: E402
from fractalsig.detection.metrics import auroc, tpr_at_1pct_fpr, tpr_at_5pct_fpr  # noqa: E402

D = lambda *p: os.path.join(HERE, *p)


def load_perplexity_scores(df, col_name, folds):
    tr, va, te = folds["random/random"]
    x = df[col_name].to_numpy(float)
    y = df["y"].to_numpy()
    notnan = ~np.isnan(x)
    va2 = va & notnan
    auc_pos = auroc(y[va2], x[va2])
    auc_neg = auroc(y[va2], -x[va2])
    sign = 1.0 if auc_pos >= auc_neg else -1.0
    return sign * x, notnan, sign


def main():
    df, groups = load_matrix()
    splits = json.load(open(D("data", "splits_v1.json")))
    folds = {f"{s}/{f}": (tr, va, te) for s, f, tr, va, te in build_folds(df, splits)}
    y = df["y"].to_numpy()

    FOLD_KEYS = ["random/random", "lodo/big_patent", "lodo/billsum", "lodo/newsroom",
                 "lodo/scientific_papers", "lodo/wikipedia", "logo/gemma_2b", "logo/mistral_7b"]

    results = {}
    for col in ["Gemma_LogPPL_Mean", "Mistral_LogPPL_Mean"]:
        scores, notnan, sign = load_perplexity_scores(df, col, folds)
        print(f"[perplexity-tpr5] {col}: sign={sign:+.0f}")
        results[col] = {}
        for fk in FOLD_KEYS:
            tr, va, te = folds[fk]
            te2 = te & notnan
            a = auroc(y[te2], scores[te2])
            t1 = tpr_at_1pct_fpr(y[te2], scores[te2])
            t5 = tpr_at_5pct_fpr(y[te2], scores[te2])
            print(f"[perplexity-tpr5] {col} {fk}: n_test={te2.sum():,} AUROC={a:.4f} TPR1={t1:.4f} TPR5={t5:.4f}")
            results[col][fk] = {"n_test": int(te2.sum()), "auroc": a, "tpr_at_1pct_fpr": t1, "tpr_at_5pct_fpr": t5}

    with open(D("data", "perplexity_tpr5_fill.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("written to data/perplexity_tpr5_fill.json")


if __name__ == "__main__":
    main()
