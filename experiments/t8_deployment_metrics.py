#!/usr/bin/env python3
"""
T8 -- deployment metrics, entirely from stored predictions (no retraining,
no rescoring). Uses the primary GAGLE in-distribution detector's test-fold
predictions (config=full/random/random/all/lgbm, data/p3_refit_predictions.jsonl,
n=43,413 -- the same config p5_task4_gagle_to_raid_transfer.py's own
"GAGLE in-distribution baseline" is read from).

GAGLE's actual human prevalence is 2.234% corpus-wide (5,000 Human /
223,834 total) -- the test-fold prevalence in this particular held-out fold
need not match that exactly (group-disjoint random/random split), so both
the FOLD's own prevalence and a re-weighted 2.234%-prevalence PPV are
reported; raw AUROC on a fold this skewed is hard to read operationally,
which is the whole reason for this task.

Reuses eval/metrics.py's own auroc/tpr_at_1pct_fpr/tpr_at_5pct_fpr/
expected_calibration_error/bootstrap_ci unmodified.
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from fractalsig import paths as _paths  # noqa: E402

PAPER = _paths.WORKSPACE_STR
from fractalsig.detection.metrics import (tpr_at_1pct_fpr, tpr_at_5pct_fpr, tpr_at_fpr,  # noqa: E402
                           expected_calibration_error, bootstrap_ci)

OUT_DIR = _paths.STATS_DIR_STR
CONFIG = ("full", "random", "random", "all", "lgbm")
GAGLE_CORPUS_HUMAN_PREVALENCE = 5000 / 223834  # 0.022337...


def load_predictions():
    y, p = [], []
    with open(os.path.join(PAPER, "data", "p3_refit_predictions.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            key = (r["config"], r["scheme"], r["fold"], r["arm"], r["learner"])
            if key == CONFIG:
                y.append(r["y"])
                p.append(r["score"])
    return np.array(y), np.array(p)


def partial_auroc(y, p, max_fpr):
    """Partial AUROC restricted to FPR<=max_fpr, normalized to [0.5,1.0]
    the way sklearn's roc_auc_score(max_fpr=...) does (McClish-corrected)."""
    return float(roc_auc_score(y, p, max_fpr=max_fpr))


def ppv_at_threshold(y, p, thr, llm_prevalence=None):
    """PPV = TP/(TP+FP) at a given score threshold, for the POSITIVE class
    (y=1 = LLM-generated, per p3_refit_predictions.jsonl's own labeling).
    If `llm_prevalence` is given, reweight TP/FP counts to that assumed
    real-world LLM share of traffic (holding the fold's own TPR/FPR at this
    threshold fixed) rather than using the fold's own class balance -- this
    answers "what would PPV be if deployed where X% of traffic is actually
    LLM-generated" for a chosen X, rather than at this held-out fold's own
    (very different, GAGLE-construction-driven) LLM/human mix.

    NOTE: GAGLE's own 2.234% figure is a HUMAN prevalence (humans are the
    rare class in this research corpus, by construction -- GAGLE oversamples
    machine generations). A real content-moderation deployment is the
    opposite shape: LLM content is typically the MINORITY of traffic. So the
    deployment scenarios swept here are LLM-prevalence values (1%/5%/10%/
    25%/50%), representing plausible real-world AI-content shares, plus
    GAGLE's own corpus-wide LLM share (1 - 0.022338 = 0.977662) as a
    reference point for "if deployed on a GAGLE-shaped population instead"."""
    pred_pos = p >= thr
    tpr = float(np.mean(pred_pos[y == 1])) if (y == 1).any() else float("nan")
    fpr = float(np.mean(pred_pos[y == 0])) if (y == 0).any() else float("nan")
    if llm_prevalence is None:
        llm_prevalence = float(np.mean(y))
    tp = tpr * llm_prevalence
    fp = fpr * (1 - llm_prevalence)
    return float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")


def main():
    y, p = load_predictions()
    n = len(y)
    fold_prevalence = float(np.mean(y))
    print(f"[t8] config={CONFIG} n={n} fold_prevalence(LLM)={fold_prevalence:.4f} "
          f"(GAGLE corpus-wide human prevalence={GAGLE_CORPUS_HUMAN_PREVALENCE:.5f})", flush=True)

    auroc_full = float(roc_auc_score(y, p))
    auroc_fpr1 = partial_auroc(y, p, 0.01)
    auroc_fpr5 = partial_auroc(y, p, 0.05)

    fpr_arr, tpr_arr, thr_arr = roc_curve(y, p)
    tpr1 = float(tpr_at_1pct_fpr(y, p))
    tpr5 = float(tpr_at_5pct_fpr(y, p))
    tpr01, _lo, _hi, k01 = bootstrap_ci(lambda yt, ys: tpr_at_fpr(yt, ys, 0.001), y, p, stratified=True)

    # threshold achieving each target FPR on the human (y=0) class, then the ACHIEVED fpr/CI on it
    def threshold_for_fpr(target):
        idx = np.searchsorted(fpr_arr, target, side="right") - 1
        idx = max(0, min(idx, len(thr_arr) - 1))
        return float(thr_arr[idx]), float(fpr_arr[idx])

    thr1, achieved_fpr1 = threshold_for_fpr(0.01)
    thr5, achieved_fpr5 = threshold_for_fpr(0.05)

    def _fpr_metric(yt, ys, thr):
        pred_pos = ys >= thr
        neg = yt == 0
        return float(np.mean(pred_pos[neg])) if neg.any() else float("nan")

    fpr1_pt, fpr1_lo, fpr1_hi, fpr1_k = bootstrap_ci(lambda yt, ys: _fpr_metric(yt, ys, thr1), y, p, stratified=True)
    fpr5_pt, fpr5_lo, fpr5_hi, fpr5_k = bootstrap_ci(lambda yt, ys: _fpr_metric(yt, ys, thr5), y, p, stratified=True)

    ece = float(expected_calibration_error(y, p, n_bins=15))

    gagle_corpus_llm_prevalence = 1 - GAGLE_CORPUS_HUMAN_PREVALENCE
    deployment_llm_prevalences = [0.01, 0.05, 0.10, 0.25, 0.50, gagle_corpus_llm_prevalence]
    ppv_table = {}
    for prev in deployment_llm_prevalences:
        label = "gagle_corpus_composition" if abs(prev - gagle_corpus_llm_prevalence) < 1e-9 else f"llm_share_{prev:.2f}"
        ppv_table[label] = {
            "assumed_llm_prevalence": prev,
            "ppv_at_1pct_fpr_threshold": ppv_at_threshold(y, p, thr1, llm_prevalence=prev),
            "ppv_at_5pct_fpr_threshold": ppv_at_threshold(y, p, thr5, llm_prevalence=prev),
        }

    out = {
        "config": CONFIG, "n": n, "fold_prevalence_llm": fold_prevalence,
        "gagle_corpus_human_prevalence": GAGLE_CORPUS_HUMAN_PREVALENCE,
        "auroc_full": auroc_full, "partial_auroc_fpr_le_1pct": auroc_fpr1, "partial_auroc_fpr_le_5pct": auroc_fpr5,
        "tpr_at_1pct_fpr": tpr1, "tpr_at_5pct_fpr": tpr5,
        "tpr_at_0.1pct_fpr": {"point": float(tpr01), "n_valid_folds": k01},
        "achieved_fpr_at_1pct_threshold": {"point": fpr1_pt, "ci": [fpr1_lo, fpr1_hi], "n_valid_folds": fpr1_k,
                                             "target": 0.01},
        "achieved_fpr_at_5pct_threshold": {"point": fpr5_pt, "ci": [fpr5_lo, fpr5_hi], "n_valid_folds": fpr5_k,
                                             "target": 0.05},
        "expected_calibration_error_15bins": ece,
        "ppv_by_deployment_llm_prevalence": ppv_table,
        "note_on_prevalence_direction": (
            "y=1 is the LLM (positive/flagged) class. GAGLE's 2.234% figure is a HUMAN "
            "prevalence (humans are the rare class in this research corpus by construction); "
            "a real content-moderation deployment is typically the opposite shape (LLM content "
            "is the minority of traffic), so the PPV sweep above is over assumed LLM "
            "prevalence, not human prevalence -- with GAGLE's own corpus-wide LLM share "
            "(1 - 0.022338 = 0.977662) included as a reference point, not as 'the' deployment case."
        ),
    }
    print(json.dumps(out, indent=2), flush=True)
    out_path = os.path.join(OUT_DIR, "t8_deployment_metrics.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n[written] {out_path}", flush=True)


if __name__ == "__main__":
    main()
