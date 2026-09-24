#!/usr/bin/env python3
"""P0-P5 audit, step 4 (2026-08-28): the 9-step Extension-1 brief's Step 1
names four classification tasks -- binary detection, generator-family ID,
prompt-type ID, and "edited vs. non-edited LLM text". The first three were
built (P3, P5 T2/task6, P5 task3). The fourth was never built as its own
trained classifier -- P4's robustness battery measures how EXISTING
detectors' recall degrades under attack, a different question from "is this
specific document itself edited or clean."

Built from data already on disk, no new generation: `p4_family_d_v2_full_
per_doc.jsonl` (Family D's post-gate recompute, `human_revision` already
excluded -- 89.2% near-identical to the original, not a real edit -- and any
edit with semantic_cosine < 0.5 already filtered, so the population here is
already quality-gated). 6 surviving attack types: paraphrase_light/heavy,
sentence_reorder, style_transfer, summarize_then_expand, typo_noise.

DESIGN, chosen deliberately over a delta-feature classifier: each row is ONE
SIDE of a matched pair (the clean document's own {H, log-PPL} per scorer, or
the edited document's own {H, log-PPL} per scorer) -- never both sides in
the same row. This is deployable on a single, unpaired document at
inference (the literal "edited vs. non-edited" framing). A delta-feature
classifier (Delta_H, Delta_logppl as inputs) would trivially need BOTH sides
already scored to compute its own features -- a different, oracle-only
question ("do these two texts differ") not what Step 1 asks for. Reported
only as a secondary ceiling, not the primary result.

Group-disjoint split on `id` (the shared source document) -- the same
document appears under multiple attack variants, and its clean row is
therefore reused across those variants (deduplicated to one clean row per
id, since clean_H/clean_logppl are deterministic and byte-identical across
variant rows for the same id -- confirmed empirically, not assumed).
Splitting by variant instead of by id would leak the source document's own
style straight across train/test, the same leakage class task 6's
source_id fix and GAGLE's own splits.py exist to prevent.
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
import sys
import collections

import numpy as np

HERE = _paths.WORKSPACE_STR
D = lambda *p: os.path.join(HERE, "data", *p)
SEED = 20260828


def main():
    from fractalsig.detection.p3_train_eval import fit_predict
    from fractalsig.detection.metrics import auroc, tpr_at_1pct_fpr, tpr_at_5pct_fpr, bootstrap_ci

    # --- load and pivot Family D v2's "full" scope per-doc file ---
    by_key = collections.defaultdict(dict)  # (id, variant) -> {scorer: row}
    with open(D("p4_family_d_v2_full_per_doc.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            by_key[(row["id"], row["variant"])][row["scorer"]] = row

    complete = {k: v for k, v in by_key.items() if "gemma" in v and "mistral" in v}
    print(f"[edited_vs_clean] {len(by_key)} (id,variant) pairs, "
          f"{len(complete)} with both scorers present (complete-case)", flush=True)

    variant_counts = collections.Counter(k[1] for k in complete)
    print(f"[edited_vs_clean] variant counts: {dict(variant_counts)}", flush=True)
    assert "human_revision" not in variant_counts, \
        "human_revision must stay excluded (89.2% near-identity, not a real edit)"

    # --- clean rows: ONE per unique id (byte-identical across variants) ---
    clean_seen = {}
    edited_rows = []
    for (doc_id, variant), scorers in complete.items():
        g, m = scorers["gemma"], scorers["mistral"]
        if doc_id not in clean_seen:
            clean_seen[doc_id] = {
                "id": doc_id, "label": 0, "variant": "clean",
                "domain": g["domain"], "generator": g["generator"],
                "Gemma_H": g["clean_H"], "Gemma_LogPPL": g["clean_logppl"],
                "Mistral_H": m["clean_H"], "Mistral_LogPPL": m["clean_logppl"],
            }
        edited_rows.append({
            "id": doc_id, "label": 1, "variant": variant,
            "domain": g["domain"], "generator": g["generator"],
            "Gemma_H": g["edited_H"], "Gemma_LogPPL": g["edited_logppl"],
            "Mistral_H": m["edited_H"], "Mistral_LogPPL": m["edited_logppl"],
        })

    all_rows = list(clean_seen.values()) + edited_rows
    print(f"[edited_vs_clean] {len(clean_seen)} clean rows (deduplicated by id), "
          f"{len(edited_rows)} edited rows -- class balance {len(edited_rows)/len(all_rows):.4f} edited", flush=True)

    feat_names = ["Gemma_H", "Gemma_LogPPL", "Mistral_H", "Mistral_LogPPL"]
    ids = sorted(clean_seen.keys())
    rng = np.random.RandomState(SEED)
    ids_shuffled = list(ids)
    rng.shuffle(ids_shuffled)
    n = len(ids_shuffled)
    n_tr, n_va = int(n * 0.7), int(n * 0.15)
    tr_ids = set(ids_shuffled[:n_tr])
    va_ids = set(ids_shuffled[n_tr:n_tr + n_va])
    te_ids = set(ids_shuffled[n_tr + n_va:])
    assert not (tr_ids & va_ids) and not (tr_ids & te_ids) and not (va_ids & te_ids), \
        "group split is not disjoint"
    print(f"[edited_vs_clean] group-disjoint split on source id: "
          f"train={len(tr_ids)} val={len(va_ids)} test={len(te_ids)} unique docs", flush=True)

    def build(id_set):
        rows = [r for r in all_rows if r["id"] in id_set]
        X = np.array([[r[f] for f in feat_names] for r in rows], dtype=float)
        y = np.array([r["label"] for r in rows], dtype=int)
        m = ~np.isnan(X).any(axis=1)
        return X[m], y[m], [r for r, keep in zip(rows, m) if keep]

    Xtr, ytr, rows_tr = build(tr_ids)
    Xva, yva, rows_va = build(va_ids)
    Xte, yte, rows_te = build(te_ids)
    print(f"[edited_vs_clean] after NaN drop: train={len(ytr)} val={len(yva)} test={len(yte)}", flush=True)
    print(f"[edited_vs_clean] test edited frac={yte.mean():.4f}", flush=True)

    p_te, base = fit_predict("lgbm", Xtr, ytr, Xva, yva, Xte, feat_names)

    a = auroc(yte, p_te)
    t1 = tpr_at_1pct_fpr(yte, p_te)
    t5 = tpr_at_5pct_fpr(yte, p_te)
    print(f"\n[edited_vs_clean] TEST (positive=edited): AUROC={a:.4f} "
          f"TPR@1%FPR={t1:.4f} TPR@5%FPR={t5:.4f}  n_test={len(yte)}", flush=True)

    a_pt, a_lo, a_hi, a_k = bootstrap_ci(auroc, yte, p_te, stratified=True)
    t1_pt, t1_lo, t1_hi, t1_k = bootstrap_ci(tpr_at_1pct_fpr, yte, p_te, stratified=True)
    t5_pt, t5_lo, t5_hi, t5_k = bootstrap_ci(tpr_at_5pct_fpr, yte, p_te, stratified=True)
    print(f"[edited_vs_clean] 95% CI (stratified bootstrap, n_boot=2000): "
          f"AUROC=[{a_lo:.4f},{a_hi:.4f}] (k={a_k}) "
          f"TPR@1%FPR=[{t1_lo:.4f},{t1_hi:.4f}] (k={t1_k}) "
          f"TPR@5%FPR=[{t5_lo:.4f},{t5_hi:.4f}] (k={t5_k})", flush=True)

    # Sanity check on the 0.5 threshold BEFORE trusting any recall table at
    # it: at 80% positive prevalence with class_weight="balanced", 0.5 need
    # not be a meaningful operating point. Measured: clean-side FPR@0.5 =
    # 50.3% -- HALF of clean documents are called "edited" at that
    # threshold, so a per-variant recall_at_0.5 table would mostly measure
    # "how permissive is 0.5", not genuine per-variant discrimination
    # (internal review caught this: a classifier that always predicts "edited"
    # would also look uniform across variants). Use the TPR@5%FPR operating
    # point instead -- derived from the clean side's own score distribution,
    # matching this project's standard low-FPR reporting convention.
    fpr_at_half = float(np.mean(p_te[yte == 0] >= 0.5))
    print(f"\n[edited_vs_clean] clean-side FPR at 0.5 threshold: {fpr_at_half:.4f} "
          f"-- {'TOO PERMISSIVE, using 5%-FPR operating point instead' if fpr_at_half > 0.20 else 'acceptable'}", flush=True)

    thresh_5pct = float(np.percentile(p_te[yte == 0], 95))
    print(f"[edited_vs_clean] threshold at clean-side 5% FPR: {thresh_5pct:.6f}", flush=True)

    print(f"\n{'variant':24}{'n':>8}{'recall_at_5pct_fpr':>20}", flush=True)
    by_variant_recall = {}
    for v in sorted(variant_counts):
        idx = [i for i, r in enumerate(rows_te) if r["variant"] == v]
        if not idx:
            continue
        yy = yte[idx]
        pp = p_te[idx]
        recall = float(np.mean(pp >= thresh_5pct))
        print(f"{v:24}{len(idx):8}{recall:20.4f}", flush=True)
        by_variant_recall[v] = {"n": len(idx), "recall_at_5pct_fpr": recall}

    with open(D("p4_edited_vs_clean.json"), "w") as f:
        json.dump({
            "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte),
            "positive_class": "edited", "negative_class": "clean",
            "features": feat_names,
            "test_edited_frac": float(yte.mean()),
            "auroc": float(a), "auroc_ci": [a_lo, a_hi], "auroc_ci_n_valid_folds": a_k,
            "tpr_1pct": float(t1), "tpr_1pct_ci": [t1_lo, t1_hi], "tpr_1pct_ci_n_valid_folds": t1_k,
            "tpr_5pct": float(t5), "tpr_5pct_ci": [t5_lo, t5_hi], "tpr_5pct_ci_n_valid_folds": t5_k,
            "clean_side_fpr_at_0.5": fpr_at_half,
            "by_variant_recall_at_5pct_fpr": by_variant_recall,
            "variant_counts": dict(variant_counts),
            "limitation": "Features are single-side {H, log-PPL} values per scorer, deployable on "
                           "one unpaired document at inference. This is NOT a delta/paired-comparison "
                           "classifier -- Delta_H/Delta_logppl would require both sides already scored "
                           "and would answer a different question ('do these two texts differ') than "
                           "'is this one document edited'.",
        }, f, indent=2)
    print("\n[edited_vs_clean] written to data/p4_edited_vs_clean.json")


if __name__ == "__main__":
    main()
