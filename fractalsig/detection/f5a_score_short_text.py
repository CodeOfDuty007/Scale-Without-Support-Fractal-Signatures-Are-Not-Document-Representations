"""F-5(a) short-text scoring: does the detector work on the 7,397 dropped rows.

Family A and B are structurally unavailable for this population (confirmed by
the tokenizer probe: only ~10% self-score under either scorer's own
tokenization -- these documents are genuinely short, not truncated by a GAGLE
array artifact). Family C is text-only and fully computable. This script:

  1. Computes Family C for all 7,397 dropped rows (reuses family_c_style,
     compute_word_budget exactly -- no reimplementation).
  2. Builds the 85-column matrix the config=full/random/random model expects,
     with Family A (fractal, 54 cols) and Family B (the 7 GAGLE/Family-B-derived
     STAT columns: LogPPL*4, Entropy_Mean, GLTR*3 per scorer, Cross_PPL_Ratio --
     17 cols total) as NaN, Family C (14 cols) filled from step 1.
  3. Scores through the config=full/random/random model (the ONLY config
     trained with real missingness in its data -- config 5/clean_both was
     trained on zero-missing rows by construction, so applying it here would
     be pure extrapolation, not a defensible test).
  4. Extracts the operating threshold at 5% FPR from that model's OWN test-fold
     ROC curve (the "main test supply" calibration requested) and
     applies it to the F-5(a) scores. Reports RECALL ONLY -- this population
     is 100% LLM (Human=0), so no FPR is computable from it; that is a
     property of the population, not a limitation of the analysis.
  5. THE DISCRIMINATING CHECK: samples retained complete-case test rows from
     the SAME fold, scores them twice -- once intact, once with Family A+B
     masked to NaN (same rows, same model). If masked-retained recall
     collapses to near the F-5(a) recall, the F-5(a) number is measuring the
     model's NaN-branch behavior, not short text, and must be reported as
     such. If masked-retained recall holds up, the F-5(a) number means what
     it's meant to mean.
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
import sys

import joblib
import numpy as np
from sklearn.metrics import roc_curve

HERE = _paths.WORKSPACE_STR

from fractalsig.detection.p3_train_eval import STAT, STYLE, load_matrix  # noqa: E402
from fractalsig.pipeline.build_p1_feature_store_cpu import family_c_style, compute_word_budget  # noqa: E402

D = lambda *p: os.path.join(HERE, *p)
MODEL_PATH = D("feature_store", "p3_refit_models", "full_random_random_all_lgbm.joblib")
DROPPED_PATH = D("data", "f5a_dropped_llm_rows.jsonl")
PRED_PATH = D("data", "p3_refit_predictions.jsonl")
OUT_PATH = D("data", "f5a_short_text_scoring.json")

STYLE_KEYS_FROM_FAMILY_C = STYLE  # the 14-column order the model expects


def build_style_row(text, n_gemma, n_mistral):
    n_ref = n_gemma if n_gemma else n_mistral
    wb = compute_word_budget(text, n_ref)
    feats = family_c_style(text, wb)
    return [feats.get(k, float("nan")) for k in STYLE_KEYS_FROM_FAMILY_C]


def main():
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feat_names = bundle["feature_names"]  # fractal(54) + stat(17) + style(14), in this order
    assert feat_names[:54 + 17] == feat_names[:71], "sanity"
    n_fractal = 54
    n_stat = len(STAT)
    n_style = len(STYLE)
    assert feat_names == feat_names[:n_fractal] + STAT + STYLE, (
        "feature order assumption broke -- rebuild build_style_row's placement")

    # --- step 1+2: build the F-5(a) matrix, Family A+B = NaN, Family C filled ---
    dropped_rows = []
    with open(DROPPED_PATH) as f:
        for line in f:
            dropped_rows.append(json.loads(line))
    print(f"[f5a-score] {len(dropped_rows)} dropped rows loaded")

    X_f5a = np.full((len(dropped_rows), len(feat_names)), np.nan)
    ids_f5a = []
    for i, r in enumerate(dropped_rows):
        lens = r.get("raw_scores_lens", {})
        n_g = lens.get("gemma-2b")
        n_m = lens.get("mistral-7b")
        style_vals = build_style_row(r["text"], n_g, n_m)
        X_f5a[i, n_fractal + n_stat:] = style_vals
        ids_f5a.append(r["id"])
    y_f5a = np.ones(len(dropped_rows), dtype=int)  # all LLM by construction

    p_f5a = model.predict_proba(X_f5a)[:, int(np.where(model.classes_ == 1)[0][0])] \
        if hasattr(model, "classes_") else model.predict_proba(X_f5a)[:, 1]

    # --- step 4: threshold from the model's own test-fold (config=full/random/random) ---
    yte, pte = [], []
    with open(PRED_PATH) as f:
        for line in f:
            rec = json.loads(line)
            if rec["config"] == "full" and rec["scheme"] == "random" and rec["fold"] == "random":
                yte.append(rec["y"])
                pte.append(rec["score"])
    yte, pte = np.array(yte), np.array(pte)
    fpr, tpr, thr = roc_curve(yte, pte)
    idx = np.where(fpr <= 0.05)[0]
    threshold_5pct = float(thr[idx[-1]])
    achieved_fpr = float(fpr[idx[-1]])
    print(f"[f5a-score] threshold at <=5% FPR on config=full/random test fold: "
          f"{threshold_5pct:.6f} (achieved FPR {achieved_fpr:.4f})")

    recall_f5a = float((p_f5a >= threshold_5pct).mean())
    print(f"[f5a-score] F-5(a) recall at that threshold: {recall_f5a:.4f} "
          f"(n={len(dropped_rows)}, all LLM -- no FPR computable from this population)")

    # --- step 5: the discriminating check ---
    df, groups = load_matrix()
    fractal, stat, style = groups["fractal"], groups["stat"], groups["style"]
    assert fractal + stat + style == feat_names
    id_to_row = {i: k for k, i in enumerate(df["id"].values)}
    test_ids_intact = [i for i in id_to_row]  # placeholder, filtered below

    # complete-case rows from the SAME test fold used for calibration/threshold
    test_id_set = set()
    with open(PRED_PATH) as f:
        for line in f:
            rec = json.loads(line)
            if rec["config"] == "full" and rec["scheme"] == "random" and rec["fold"] == "random":
                test_id_set.add(rec["id"])

    feats_all = fractal + stat + style
    complete_mask = df[feats_all].notna().all(axis=1).values
    df_test = df[df["id"].isin(test_id_set) & complete_mask]
    df_test_llm = df_test[df_test["Label"] != "Human"]
    rng = np.random.default_rng(20260825)
    n_sample = min(2000, len(df_test_llm))
    sample = df_test_llm.sample(n=n_sample, random_state=20260825)
    print(f"[f5a-score] discriminating check: {n_sample} complete-case LLM test rows sampled "
          f"(of {len(df_test_llm)} available)")

    X_intact = sample[feats_all].to_numpy(float)
    X_masked = X_intact.copy()
    X_masked[:, :n_fractal + n_stat] = np.nan  # mask Family A + Family B, keep Family C real

    def score(X):
        proba = model.predict_proba(X)
        pos = int(np.where(model.classes_ == 1)[0][0])
        return proba[:, pos]

    p_intact = score(X_intact)
    p_masked = score(X_masked)
    recall_intact = float((p_intact >= threshold_5pct).mean())
    recall_masked = float((p_masked >= threshold_5pct).mean())
    print(f"[f5a-score] discriminating check -- intact recall: {recall_intact:.4f}  "
          f"masked (A+B->NaN) recall: {recall_masked:.4f}  F-5(a) recall: {recall_f5a:.4f}")

    interpretation = (
        "F-5(a) recall tracks the masked-retained recall (both driven by the same "
        "NaN-branch behavior), NOT the intact recall -- the number measures missingness, "
        "not short text. Report with this caveat explicit."
        if abs(recall_masked - recall_f5a) < abs(recall_intact - recall_f5a)
        else
        "F-5(a) recall is closer to intact than to masked-retained recall -- Family C "
        "carries real signal on this population even with A+B absent. The number is "
        "defensible as 'detector performance on short text', not merely a missingness artifact."
    )
    print(f"[f5a-score] INTERPRETATION: {interpretation}")

    out = {
        "n_dropped_llm_scored": len(dropped_rows),
        "threshold_5pct_source": "config=full, scheme=random, fold=random, arm=all, learner=lgbm",
        "threshold_5pct_value": threshold_5pct,
        "achieved_fpr_on_source_test_fold": achieved_fpr,
        "f5a_recall_at_threshold": recall_f5a,
        "discriminating_check": {
            "n_sampled_complete_case_llm_test_rows": n_sample,
            "recall_intact": recall_intact,
            "recall_masked_family_ab_to_nan": recall_masked,
            "f5a_recall": recall_f5a,
        },
        "interpretation": interpretation,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[f5a-score] written to {OUT_PATH}")


if __name__ == "__main__":
    main()
