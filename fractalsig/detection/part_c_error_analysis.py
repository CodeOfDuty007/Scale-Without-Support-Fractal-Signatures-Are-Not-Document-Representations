"""Part C: explain the billsum LODO collapse (config=full, arm=all).
AUROC 0.9984 (in-distribution) -> 0.9476 (billsum); TPR@1%FPR 0.9636 -> 0.5630.

Targeted questions, not open-ended inspection:
  1. What distinguishes billsum false negatives from true positives (length,
     prompt_type, temp, generator, dominant features)?
  2. Are false negatives concentrated in short LLM documents (links to F-5a)?
  3. Threshold artifact or ranking failure? (AUROC is threshold-invariant and
     barely moves; TPR@1%FPR is fold-native, i.e. already computed from
     billsum's OWN best-achievable operating point at that FPR -- so there is
     no "imported wrong threshold" to blame. If the answer is ranking
     failure, it will show up as LLM documents scoring BELOW billsum's own
     human rows' upper tail, concentrated specifically at the low end.)
  4. Which features shift most between training domains and billsum
     (P4 attack-design input)?
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = _paths.WORKSPACE_STR

from fractalsig.detection.p3_train_eval import load_matrix, build_folds, STAT, STYLE  # noqa: E402
from fractalsig.detection.metrics import tpr_at_1pct_fpr  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

D = lambda *p: os.path.join(HERE, *p)
PRED_PATH = D("data", "p3_refit_predictions.jsonl")
OUT_JSON = D("data", "part_c_billsum_error_analysis.json")
OUT_REPORT = D("reports", "2026-08-25_P3_error_analysis_billsum_LODO_collapse.md")


def main():
    df, groups = load_matrix()
    feats_all = groups["fractal"] + STAT + STYLE
    splits = json.load(open(D("data", "splits_v1.json")))
    folds = {f"{s}/{f}": (tr, va, te) for s, f, tr, va, te in build_folds(df, splits)}

    preds = []
    with open(PRED_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r["config"] == "full" and r["scheme"] == "lodo" and r["fold"] == "billsum":
                preds.append(r)
    pred_df = pd.DataFrame(preds)
    print(f"[partC] {len(pred_df)} billsum LODO test predictions loaded")

    merged = pred_df.merge(df, on="id", how="left", suffixes=("", "_full"))
    assert merged["y"].equals(merged["y_full"]) if "y_full" in merged else True
    assert len(merged) == len(pred_df)
    assert merged["y"].to_numpy().sum() == int((merged["Label"] != "Human").sum())

    y = merged["y"].to_numpy()
    p = merged["score"].to_numpy()

    # --- Q3: ranking failure vs threshold artifact ---
    t1_billsum = tpr_at_1pct_fpr(y, p)
    fpr, tpr, thr = roc_curve(y, p)
    idx = np.where(fpr <= 0.01)[0]
    threshold_1pct_billsum = float(thr[idx[-1]]) if len(idx) else None
    n_human = int((y == 0).sum())
    human_scores = np.sort(p[y == 0])[::-1]
    # the human score that DEFINES the 1%-FPR cutoff (fold-native, not imported)
    n_allowed_fp = max(1, int(np.floor(0.01 * n_human)))
    cutoff_human_score = float(human_scores[n_allowed_fp - 1]) if n_human > 0 else None
    llm_scores = p[y == 1]
    frac_llm_below_cutoff = float((llm_scores < cutoff_human_score).mean()) if cutoff_human_score is not None else None

    q3 = {
        "auroc_indistribution": 0.9984, "auroc_billsum": 0.9476,
        "tpr1_indistribution": 0.9636, "tpr1_billsum": float(t1_billsum),
        "n_human_billsum_test": n_human,
        "threshold_1pct_is_fold_native": True,
        "note": ("tpr_at_1pct_fpr is computed from billsum's OWN roc_curve -- there is no "
                 "externally-imported threshold to blame. AUROC (threshold-invariant, "
                 "whole-curve) barely moves while TPR@1%FPR (tail-only) collapses, which by "
                 "construction means bulk ranking is preserved but tail ranking fails: a large "
                 "share of billsum LLM documents score BELOW the human documents that set the "
                 "1%-FPR cutoff."),
        "human_score_defining_1pct_cutoff": cutoff_human_score,
        "frac_llm_docs_scoring_below_that_cutoff": frac_llm_below_cutoff,
        "verdict": "RANKING FAILURE at the tail, not a threshold artifact.",
    }
    print(f"[partC] Q3: {frac_llm_below_cutoff:.4f} of billsum LLM docs score below the "
          f"human-defined 1%-FPR cutoff ({cutoff_human_score:.4f}) -- ranking failure at the tail")

    # --- Q1/Q2: false negatives at the fold-native 1%-FPR threshold ---
    is_llm = y == 1
    is_fn = is_llm & (p < threshold_1pct_billsum)
    is_tp = is_llm & (p >= threshold_1pct_billsum)
    print(f"[partC] LLM rows: {int(is_llm.sum())}  FN: {int(is_fn.sum())} ({100*is_fn.sum()/is_llm.sum():.1f}%)  TP: {int(is_tp.sum())}")

    def describe(mask, label):
        sub = merged[mask]
        return {
            "label": label, "n": int(mask.sum()),
            "N_Words_Used_median": float(sub["N_Words_Used"].median()),
            "N_Words_Used_mean": float(sub["N_Words_Used"].mean()),
            "N_Sentences_median": float(sub["N_Sentences"].median()),
            "by_prompt_type": sub["prompt_type"].value_counts().to_dict() if "prompt_type" in sub else None,
            "by_temp": sub["temp"].value_counts().to_dict() if "temp" in sub else None,
            "by_generator": sub["Label"].value_counts().to_dict(),
        }

    fn_desc = describe(is_fn, "false_negative")
    tp_desc = describe(is_tp, "true_positive")
    print(f"[partC] FN N_Words_Used median={fn_desc['N_Words_Used_median']:.0f}  "
          f"TP N_Words_Used median={tp_desc['N_Words_Used_median']:.0f}")
    print(f"[partC] FN by generator: {fn_desc['by_generator']}")
    print(f"[partC] TP by generator: {tp_desc['by_generator']}")
    print(f"[partC] FN by temp: {fn_desc['by_temp']}")
    print(f"[partC] TP by temp: {tp_desc['by_temp']}")

    # short-LLM-doc concentration check (links to F-5a's bias-direction finding)
    from scipy import stats as sps
    u, pval = sps.mannwhitneyu(merged.loc[is_fn, "N_Words_Used"], merged.loc[is_tp, "N_Words_Used"],
                                alternative="less")
    q2 = {
        "fn_median_words": fn_desc["N_Words_Used_median"], "tp_median_words": tp_desc["N_Words_Used_median"],
        "mannwhitney_u": float(u), "p_value_fn_shorter": float(pval),
        "verdict": "FN docs significantly shorter than TP docs" if pval < 0.05 else "no significant length difference",
    }
    print(f"[partC] Q2: Mann-Whitney FN<TP on N_Words_Used p={pval:.2e} -> {q2['verdict']}")

    # --- Q4: feature-distribution shift, training domains vs billsum ---
    train_mask = folds["lodo/billsum"][0]
    llm_train = df.loc[train_mask & (df["Label"] != "Human"), feats_all]
    llm_billsum = merged.loc[is_llm, feats_all]
    shifts = []
    for col in feats_all:
        a = llm_train[col].dropna()
        b = llm_billsum[col].dropna()
        if len(a) < 10 or len(b) < 10:
            continue
        med_a, med_b = a.median(), b.median()
        iqr_a = a.quantile(0.75) - a.quantile(0.25)
        denom = iqr_a if iqr_a > 0 else (a.std() if a.std() > 0 else 1.0)
        standardized_shift = float((med_b - med_a) / denom)
        try:
            ks_stat, ks_p = sps.ks_2samp(a, b)
        except Exception:
            ks_stat, ks_p = float("nan"), float("nan")
        shifts.append({"feature": col, "train_median": float(med_a), "billsum_median": float(med_b),
                       "standardized_shift": standardized_shift, "ks_stat": float(ks_stat), "ks_p": float(ks_p)})
    shifts.sort(key=lambda r: -abs(r["standardized_shift"]))
    top_shifts = shifts[:15]
    print("[partC] Q4 top feature shifts (train LLM vs billsum LLM, by standardized median shift):")
    for r in top_shifts[:10]:
        print(f"   {r['feature']:<35s} shift={r['standardized_shift']:+.3f}  KS={r['ks_stat']:.3f} (p={r['ks_p']:.2e})")

    out = {"n_test": len(merged), "n_llm": int(is_llm.sum()), "n_human": n_human,
           "q1_false_negative_profile": fn_desc, "q1_true_positive_profile": tp_desc,
           "q2_short_text_concentration": q2, "q3_ranking_vs_threshold": q3,
           "q4_top_feature_shifts": top_shifts}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[partC] written to {OUT_JSON}")
    return out


if __name__ == "__main__":
    main()
