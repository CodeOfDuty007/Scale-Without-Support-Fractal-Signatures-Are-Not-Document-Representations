#!/usr/bin/env python3
"""P4 task 5, perplexity-threshold detector. CPU-only -- log-PPL is
mean(scores) from the already-computed rescore arrays (job206), no new
scoring needed.

Threshold derivation matches perplexity_tpr5_fill.py's own method exactly
(sign chosen once on random/val, same fold): pick the fixed score cutoff
that achieves the target FPR on the VAL fold of the ORIGINAL (clean,
unattacked) corpus, then apply that SAME fixed cutoff to the P4 attacked
population and report recall (TPR) -- this is what "robustness" means here,
per the F-5(a) precedent (recall_intact vs recall_masked framing).
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_curve

HERE = _paths.WORKSPACE_STR
from fractalsig.detection.p3_train_eval import load_matrix, build_folds  # noqa: E402

D = lambda *p: os.path.join(HERE, *p)
SCORER_COL = {"gemma": "Gemma_LogPPL_Mean", "mistral": "Mistral_LogPPL_Mean"}


def pick_threshold(y_val, score_val, target_fpr):
    fpr, tpr, thresh = roc_curve(y_val, score_val)
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return None
    return float(thresh[idx[-1]])


def main():
    df, groups = load_matrix()
    splits = json.load(open(D("data", "splits_v1.json")))
    folds = {f"{s}/{f}": (tr, va, te) for s, f, tr, va, te in build_folds(df, splits)}
    tr, va, te = folds["random/random"]
    y = df["y"].to_numpy()

    thresholds = {}
    for scorer, col in SCORER_COL.items():
        x = df[col].to_numpy(float)
        notnan = ~np.isnan(x)
        va2 = va & notnan
        from sklearn.metrics import roc_auc_score
        auc_p = roc_auc_score(y[va2], x[va2])
        auc_n = roc_auc_score(y[va2], -x[va2])
        sign = 1.0 if auc_p >= auc_n else -1.0
        s = sign * x
        t1 = pick_threshold(y[va2], s[va2], 0.01)
        t5 = pick_threshold(y[va2], s[va2], 0.05)
        thresholds[scorer] = {"sign": sign, "t1_fpr1pct": t1, "t5_fpr5pct": t5}
        print(f"[perplexity] {scorer}: sign={sign:+.0f}  threshold@1%FPR={t1:.4f}  threshold@5%FPR={t5:.4f}")

    pop = [json.loads(l) for l in open(D("data", "p4_task5_population.jsonl"))]
    rescore = {}
    for scorer in ("gemma", "mistral"):
        rescore[scorer] = {}
        for batch in (1, 2):
            for l in open(D("data", f"p4_rescore_batch{batch}_{scorer}.jsonl")):
                r = json.loads(l)
                rescore[scorer][(r["id"], r["variant"])] = r

    from collections import defaultdict
    by_type_scorer = defaultdict(lambda: {"t1_hit": 0, "t5_hit": 0, "n": 0})
    for row in pop:
        for scorer in ("gemma", "mistral"):
            rec = rescore[scorer].get((row["id"], row["variant"]))
            if rec is None or not rec["scores"]:
                continue
            logppl = float(np.mean(rec["scores"]))
            sign = thresholds[scorer]["sign"]
            s = sign * logppl
            key = (row["variant"], scorer)
            by_type_scorer[key]["n"] += 1
            if thresholds[scorer]["t1_fpr1pct"] is not None and s >= thresholds[scorer]["t1_fpr1pct"]:
                by_type_scorer[key]["t1_hit"] += 1
            if thresholds[scorer]["t5_fpr5pct"] is not None and s >= thresholds[scorer]["t5_fpr5pct"]:
                by_type_scorer[key]["t5_hit"] += 1

    print(f"\n{'type':22} {'scorer':8} {'n':>6} {'recall@1%FPR':>13} {'recall@5%FPR':>13}")
    results = {}
    for (vtype, scorer), d in sorted(by_type_scorer.items()):
        r1 = d["t1_hit"] / d["n"] if d["n"] else 0
        r5 = d["t5_hit"] / d["n"] if d["n"] else 0
        print(f"{vtype:22} {scorer:8} {d['n']:6} {r1:13.4f} {r5:13.4f}")
        results[f"{vtype}/{scorer}"] = {"n": d["n"], "recall_at_1pct_fpr": r1, "recall_at_5pct_fpr": r5}

    with open(D("data", "p4_task5_perplexity_threshold.json"), "w") as f:
        json.dump({"thresholds": thresholds, "results": results}, f, indent=2, default=float)
    print("\nwritten to data/p4_task5_perplexity_threshold.json")


if __name__ == "__main__":
    main()
