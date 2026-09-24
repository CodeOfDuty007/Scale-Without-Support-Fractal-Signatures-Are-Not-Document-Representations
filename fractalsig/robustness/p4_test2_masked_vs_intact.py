#!/usr/bin/env python3
"""Gate Test 2, actually run (internal review caught 2026-08-27: the original gate
write-up substituted a design decision -- exclude NaN-carrying rows, citing
P3's F-5(a) number -- for the literal procedure as specified. This
runs the real procedure on P4 data: score complete-case attacked documents
INTACT, then the SAME documents with Family A (fractal) + Family B (stat)
masked to NaN and Family C (style) intact, compare recall.

Pre-registered here, before running: masked recall - intact recall >= 0.02
(2pp), same direction as P3's own confound (masked HIGHER), constitutes
confirmation that the missingness confound is present in P4 data.
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
import sys

import numpy as np
import joblib

HERE = _paths.WORKSPACE_STR
from fractalsig.detection.p3_train_eval import load_matrix  # noqa: E402
from fractalsig.robustness.p4_task5_tabular_features import build_feature_row  # noqa: E402
from fractalsig.pipeline.build_p1_feature_store_gpu import load_models  # noqa: E402

D = lambda *p: os.path.join(HERE, "data", *p)
N_SAMPLE = 2000  # matches F-5(a)'s own n_sampled_complete_case_llm_test_rows
THRESH = 0.02


def main():
    _, groups = load_matrix()
    fractal_cols = set(groups["fractal"])
    stat_cols = set(groups["stat"])
    mask_cols = fractal_cols | stat_cols  # Family A + B

    bundle = joblib.load(os.path.join(HERE, "feature_store", "p3_refit_models", "full_random_random_all_lgbm.joblib"))
    cal_model = bundle["model"]
    feature_names = bundle["feature_names"]
    pos = int(np.where(cal_model.classes_ == 1)[0][0])

    thr = json.load(open(D("p4_task5_tabular_thresholds.json")))
    t1, t5 = thr["t1_fpr1pct"], thr["t5_fpr5pct"]
    print(f"[test2] thresholds: t1(1%FPR)={t1:.4f}  t5(5%FPR)={t5:.4f}")
    print(f"[test2] pre-registered: masked_recall - intact_recall >= {THRESH} = confirmed confound")

    pop = [json.loads(l) for l in open(D("p4_task5_population.jsonl"))]
    rng = np.random.RandomState(20260827)
    idx = rng.choice(len(pop), size=min(N_SAMPLE, len(pop)), replace=False)
    sample = [pop[i] for i in idx]
    print(f"[test2] sampled {len(sample)} complete-case attacked docs")

    rescore = {}
    for scorer in ("gemma", "mistral"):
        rescore[scorer] = {}
        for batch in (1, 2):
            for l in open(D(f"p4_rescore_batch{batch}_{scorer}.jsonl")):
                r = json.loads(l)
                rescore[scorer][(r["id"], r["variant"])] = r

    _, device, _, _, _, embedder = load_models(scorers=())

    intact_probs, masked_probs = [], []
    for i, row in enumerate(sample):
        gemma_rec = rescore["gemma"].get((row["id"], row["variant"]))
        mistral_rec = rescore["mistral"].get((row["id"], row["variant"]))
        if gemma_rec is None or mistral_rec is None:
            continue
        feats = build_feature_row(row["text"], gemma_rec, mistral_rec, embedder, device)
        x_intact = np.array([[feats.get(n, np.nan) for n in feature_names]], dtype=float)
        x_masked = np.array([[np.nan if n in mask_cols else feats.get(n, np.nan) for n in feature_names]], dtype=float)
        intact_probs.append(cal_model.predict_proba(x_intact)[0, pos])
        masked_probs.append(cal_model.predict_proba(x_masked)[0, pos])
        if (i + 1) % 500 == 0:
            print(f"[test2]   {i+1}/{len(sample)}", flush=True)

    intact_probs = np.array(intact_probs)
    masked_probs = np.array(masked_probs)
    for label, t in [("1%FPR", t1), ("5%FPR", t5)]:
        r_intact = np.mean(intact_probs >= t)
        r_masked = np.mean(masked_probs >= t)
        gap = r_masked - r_intact
        verdict = "CONFIRMED (confound present)" if gap >= THRESH else "not confirmed"
        print(f"[test2] @{label}: recall_intact={r_intact:.4f}  recall_masked={r_masked:.4f}  "
              f"gap={gap:+.4f}  verdict={verdict}")

    with open(D("p4_test2_masked_vs_intact.json"), "w") as f:
        json.dump({
            "n_sample": len(intact_probs),
            "recall_intact_1pct": float(np.mean(intact_probs >= t1)),
            "recall_masked_1pct": float(np.mean(masked_probs >= t1)),
            "recall_intact_5pct": float(np.mean(intact_probs >= t5)),
            "recall_masked_5pct": float(np.mean(masked_probs >= t5)),
        }, f, indent=2)


if __name__ == "__main__":
    main()
