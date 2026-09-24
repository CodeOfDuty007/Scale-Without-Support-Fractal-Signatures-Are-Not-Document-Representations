#!/usr/bin/env python3
"""P4 task 5, Fast-DetectGPT. Reuses score_documents/score_batch_curvature
from fast_detectgpt_score.py as-is (design decision: no new detector code).
Mandatory pre-attack baseline: re-derive the exact score threshold from the
val fold of the ORIGINAL corpus (same method as every other detector in this
battery), cross-checked against the already-computed data/fastdetectgpt_eval.json
random/random entry (TPR@1%=0.5104, TPR@5%=0.6482) before trusting it, then
apply that fixed threshold to P4's attacked population.
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_curve

HERE = _paths.WORKSPACE_STR
from fractalsig.baselines.fast_detectgpt_score import score_documents  # noqa: E402
from fractalsig.pipeline.build_p1_feature_store_gpu import load_models  # noqa: E402
from fractalsig.detection.p3_train_eval import load_matrix, build_folds  # noqa: E402

D = lambda *p: os.path.join(HERE, "data", *p)


def main(limit=None):
    torch_mod, device, dtype, tokenizers, models, _ = load_models(scorers=("gemma-2b",))
    tok, model = tokenizers["gemma-2b"], models["gemma-2b"]
    print(f"[fdgpt] gemma-2b loaded on {device}", flush=True)

    # --- pre-attack threshold, derived from val fold of the ORIGINAL corpus ---
    existing = {json.loads(l)["id"]: json.loads(l)["score"] for l in open(D("fastdetectgpt_scores.jsonl"))}
    df, groups = load_matrix()
    splits = json.load(open(D("splits_v1.json")))
    folds = {f"{s}/{f}": (tr, va, te) for s, f, tr, va, te in build_folds(df, splits)}
    tr, va, te = folds["random/random"]
    y = df["y"].to_numpy()
    ids = df["id"].to_numpy()
    score_col = np.array([existing.get(i, np.nan) for i in ids])
    notnan = ~np.isnan(score_col)
    va2 = va & notnan
    te2 = te & notnan
    fpr, tpr, thresh = roc_curve(y[va2], score_col[va2])
    idx1 = np.where(fpr <= 0.01)[0]
    idx5 = np.where(fpr <= 0.05)[0]
    t1 = thresh[idx1[-1]] if len(idx1) else None
    t5 = thresh[idx5[-1]] if len(idx5) else None
    llm_mask_te = te2 & (y == 1)
    r1_pre = float(np.mean(score_col[llm_mask_te] >= t1))
    r5_pre = float(np.mean(score_col[llm_mask_te] >= t5))
    print(f"[fdgpt] thresholds: t1={t1:.4f} t5={t5:.4f}", flush=True)
    print(f"[fdgpt] PRE-ATTACK recall (re-derived): n={llm_mask_te.sum()} "
          f"recall@1%FPR={r1_pre:.4f} recall@5%FPR={r5_pre:.4f}", flush=True)
    print(f"[fdgpt] cross-check vs data/fastdetectgpt_eval.json random/random: "
          f"TPR1=0.5104 TPR5=0.6482 (pre-computed on FULL test fold, different n)", flush=True)

    # --- score P4 population ---
    pop = [json.loads(l) for l in open(D("p4_task5_population.jsonl"))]
    if limit:
        pop = pop[:limit]
    rows = [{"idx": i, "id": r["id"], "text": r["text"], "variant": r["variant"],
              "separate_bucket": r["separate_bucket"]} for i, r in enumerate(pop)]
    print(f"[fdgpt] scoring {len(rows)} P4 population rows", flush=True)
    scores = score_documents(torch_mod, model, tok, device, rows)
    print(f"[fdgpt] scored {len(scores)}/{len(rows)} (some may be below MIN_TOKENS)", flush=True)

    from collections import defaultdict
    by_type = defaultdict(lambda: {"n": 0, "r1": 0, "r5": 0})
    out = open(D("p4_task5_fastdetectgpt_scores.jsonl"), "w")
    for idx, row in enumerate(rows):
        s = scores.get(idx)
        if s is None:
            continue
        out.write(json.dumps({"id": row["id"], "variant": row["variant"], "d": s["d"],
                                "n_scored": s["n_scored"]}) + "\n")
        d_val = by_type[row["variant"]]
        d_val["n"] += 1
        if s["d"] >= t1:
            d_val["r1"] += 1
        if s["d"] >= t5:
            d_val["r5"] += 1
    out.close()

    print(f"\n{'type':22} {'n':>6} {'recall@1%FPR':>13} {'recall@5%FPR':>13}")
    results = {"pre_attack": {"recall_1pct": r1_pre, "recall_5pct": r5_pre, "n": int(llm_mask_te.sum())}}
    for t, d in sorted(by_type.items()):
        r1 = d["r1"] / d["n"] if d["n"] else 0
        r5 = d["r5"] / d["n"] if d["n"] else 0
        print(f"{t:22} {d['n']:6} {r1:13.4f} {r5:13.4f}")
        results[t] = {"n": d["n"], "recall_1pct": r1, "recall_5pct": r5}

    with open(D("p4_task5_fastdetectgpt_summary.json"), "w") as f:
        json.dump({"thresholds": {"t1": float(t1), "t5": float(t5)}, "results": results}, f, indent=2)
    print("\nwritten to data/p4_task5_fastdetectgpt_summary.json", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(limit=args.limit)
