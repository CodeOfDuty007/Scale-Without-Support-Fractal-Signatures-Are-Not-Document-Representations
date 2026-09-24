#!/usr/bin/env python3
"""P4 task 5, Binoculars. Reuses load_falcon_pair/score_batch_binoculars/
compute_window from binoculars_score.py as-is. Mandatory pre-attack
baseline: re-derive the fixed threshold from the val fold of the ORIGINAL
corpus using the already-persisted data/binoculars_scores.jsonl (no need to
rescore the whole original corpus), then apply that same threshold to P4's
attacked population, freshly scored here (both Falcon-7b models resident,
~28GB bf16 -- confirmed no other GPU job running before this was submitted,
per this project's own documented OOM-cascade precedent on shared MIG
memory).
"""
from fractalsig import paths as _paths  # noqa: E402
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

HERE = _paths.WORKSPACE_STR
from fractalsig.baselines.binoculars_score import load_falcon_pair, score_batch_binoculars, compute_window  # noqa: E402
from fractalsig.detection.p3_train_eval import load_matrix, build_folds  # noqa: E402

D = lambda *p: os.path.join(HERE, "data", *p)
BATCH_SIZE = 4


def main(limit=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[bino] loading Falcon pair on {device}", flush=True)
    tok, observer, performer = load_falcon_pair(device)
    print("[bino] both Falcon-7b models loaded", flush=True)

    # --- pre-attack threshold from the ORIGINAL corpus's val fold ---
    existing = {json.loads(l)["id"]: json.loads(l)["score"] for l in open(D("binoculars_scores.jsonl"))}
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
    # Binoculars' own convention: LOWER score = more machine-like. Sign so
    # "higher = more positive/LLM" matches every other detector's convention.
    auc_pos = roc_auc_score(y[va2], score_col[va2])
    auc_neg = roc_auc_score(y[va2], -score_col[va2])
    sign = 1.0 if auc_pos >= auc_neg else -1.0
    s = sign * score_col
    fpr, tpr, thresh = roc_curve(y[va2], s[va2])
    idx1 = np.where(fpr <= 0.01)[0]
    idx5 = np.where(fpr <= 0.05)[0]
    t1 = thresh[idx1[-1]] if len(idx1) else None
    t5 = thresh[idx5[-1]] if len(idx5) else None
    llm_mask_te = te2 & (y == 1)
    r1_pre = float(np.mean(s[llm_mask_te] >= t1))
    r5_pre = float(np.mean(s[llm_mask_te] >= t5))
    print(f"[bino] sign={sign:+.0f}  thresholds: t1={t1:.4f} t5={t5:.4f}", flush=True)
    print(f"[bino] PRE-ATTACK recall: n={llm_mask_te.sum()} recall@1%FPR={r1_pre:.4f} "
          f"recall@5%FPR={r5_pre:.4f}", flush=True)

    # --- score P4 population ---
    pop = [json.loads(l) for l in open(D("p4_task5_population.jsonl"))]
    if limit:
        pop = pop[:limit]
    print(f"[bino] scoring {len(pop)} P4 population rows", flush=True)

    usable = []
    for row in pop:
        n_tokens_ids = tok(row["text"])["input_ids"]
        window = compute_window(len(n_tokens_ids))
        if window is None:
            continue
        usable.append((row, window))
    usable.sort(key=lambda x: x[1][1])  # sort by window end (proxy for length), OOM guard

    out = open(D("p4_task5_binoculars_scores.jsonl"), "w")
    from collections import defaultdict
    by_type = defaultdict(lambda: {"n": 0, "r1": 0, "r5": 0})
    pos = 0
    bs = BATCH_SIZE
    n_done = 0
    while pos < len(usable):
        batch = usable[pos:pos + bs]
        texts = [b[0]["text"] for b in batch]
        windows = [b[1] for b in batch]
        try:
            results = score_batch_binoculars(tok, observer, performer, device, texts, windows)
        except RuntimeError as e:
            if "memory" in str(e).lower() and bs > 1:
                bs = max(1, bs // 2)
                torch.cuda.empty_cache()
                continue
            raise
        for (row, window), r in zip(batch, results):
            if r is None:
                continue
            raw_score = r["score"]
            sc = sign * raw_score
            out.write(json.dumps({"id": row["id"], "variant": row["variant"],
                                    "score_raw": raw_score, "score_signed": sc,
                                    "n_scored": r["n_scored"]}) + "\n")
            d = by_type[row["variant"]]
            d["n"] += 1
            if sc >= t1:
                d["r1"] += 1
            if sc >= t5:
                d["r5"] += 1
            n_done += 1
        pos += len(batch)
        if pos % 400 < bs:
            out.flush()
            print(f"[bino]   {pos}/{len(usable)}  done={n_done}", flush=True)
    out.close()

    print(f"\n{'type':22} {'n':>6} {'recall@1%FPR':>13} {'recall@5%FPR':>13}")
    results_summary = {"pre_attack": {"recall_1pct": r1_pre, "recall_5pct": r5_pre, "n": int(llm_mask_te.sum())}}
    for t, d in sorted(by_type.items()):
        r1 = d["r1"] / d["n"] if d["n"] else 0
        r5 = d["r5"] / d["n"] if d["n"] else 0
        print(f"{t:22} {d['n']:6} {r1:13.4f} {r5:13.4f}")
        results_summary[t] = {"n": d["n"], "recall_1pct": r1, "recall_5pct": r5}

    with open(D("p4_task5_binoculars_summary.json"), "w") as f:
        json.dump({"sign": sign, "thresholds": {"t1": float(t1), "t5": float(t5)},
                    "results": results_summary}, f, indent=2)
    print("\nwritten to data/p4_task5_binoculars_summary.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(limit=args.limit)
