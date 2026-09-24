#!/usr/bin/env python3
"""Paired significance: each baseline vs config 5 (clean_both), on the
INTERSECTION of ids each pair actually shares -- never assumed equal to
either side's own population.

Mechanics (internally reviewed, 2026-08-25):
  - bootstrap_ci_paired / delong_test index y_true, score_a, score_b with ONE
    shared array of positions -- so position i must be the SAME document in
    all three. Built here via an explicit inner-join on id, sorted by id,
    never by zipping two separately-ordered lists.
  - The paired n is reported explicitly next to every interval -- it is
    neither config 5's own test-fold n nor the baseline's own coverage n.
  - DeLong runs alongside the bootstrap for AUROC as a cross-check (§7:
    DeLong and the paired bootstrap agreed 40/40 on the tabular comparisons;
    a disagreement here is a signal something is wrong with the pairing, not
    a finding).
  - Primary metric TPR@5%FPR (D-11/R3.0); TPR@1%FPR reported but flagged
    low-resolution given config 5's ~1,846-row human population.
  - RoBERTa: random/LOGO folds use the single verified run (calibrated
    probability, real score variance). LODO folds use the 2026-08-25
    multi-seed rerun's pre-softmax logit margin (softmax saturates on these
    folds -- see report -- the margin is the trustworthy score), run
    PER SEED rather than pooled or arbitrarily reduced to one seed, because
    big_patent's 3 seeds span worse-than-chance to excellent and a single
    number would misrepresent the fold.
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = _paths.WORKSPACE_STR

from fractalsig.detection.metrics import auroc, tpr_at_1pct_fpr, tpr_at_5pct_fpr, bootstrap_ci_paired, delong_test  # noqa: E402

D = lambda *p: os.path.join(HERE, *p)
OUT_PATH = D("data", "part_d_paired_significance.json")


def load_config5_predictions():
    """{(scheme, fold): DataFrame[id, y, score]} for config=clean_both."""
    rows = []
    with open(D("data", "p3_refit_predictions.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r["config"] == "clean_both":
                rows.append(r)
    df = pd.DataFrame(rows)
    out = {}
    for (scheme, fold), g in df.groupby(["scheme", "fold"]):
        out[(scheme, fold)] = g[["id", "y", "score"]].drop_duplicates("id").set_index("id")
    return out


def load_baseline_scores(name):
    """Returns {id: score} for a baseline. Handles each file's own schema."""
    scores = {}
    if name == "fastdetectgpt":
        path = D("data", "fastdetectgpt_scores.jsonl")
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                scores[r["id"]] = r["score"]
    elif name == "binoculars":
        path = D("data", "binoculars_scores.jsonl")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                # lower Binoculars score => more machine-like; FLIP so higher
                # score => more machine-like, matching every other baseline's
                # and the detector's convention (positive class = LLM), so
                # AUROC/TPR@FPR are computed the same direction throughout
                scores[r["id"]] = -r["score"]
    else:
        raise ValueError(name)
    return scores


def load_roberta_scores_for_fold(scheme, fold):
    """{id: score} for ONE (scheme, fold) only -- RoBERTa's per-row file has
    multiple folds appended, and big_patent specifically has THREE runs
    appended from the 2026-08-25 instability investigation (see report:
    weight-cap tested and refuted, LODO output collapses under domain shift
    regardless of class weight). Dedup to the LAST n_test rows for this
    fold, matching that fold's own summary n_test, so a stale run is never
    silently mixed in."""
    summary_path = D("data", "roberta_summaries", f"{scheme}_{fold}.json")
    n_test = json.load(open(summary_path))["n_test"]
    rows = []
    with open(D("data", "roberta_predictions.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r["scheme"] == scheme and r["fold"] == fold:
                rows.append(r)
    rows = rows[-n_test:]
    return {r["id"]: r["score"] for r in rows}


def load_roberta_margin_scores(scheme, fold, seed):
    """{id: logit_margin} for ONE (scheme, fold, seed) from the multi-seed
    rerun (2026-08-25). Files are keyed by seed only (all 5 LODO folds for
    that seed share one file), so filter by scheme/fold on read."""
    path = D("data", f"roberta_logit_margin_diagnostic_marginfp32_seed{seed}.jsonl")
    scores = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["scheme"] == scheme and r["fold"] == fold:
                scores[r["id"]] = r["logit_margin"]
    return scores


def load_perplexity_scores(col_name):
    """Per-row {id: signed_score} for a LogPPL column, sign chosen once
    globally via the random/val fold (matches perplexity_threshold_baseline.json's
    own methodology) -- avoids re-deriving sign per fold, which would let the
    sign itself be chosen by peeking at each fold's own test labels."""
    pass  # import path handled by the fractalsig package
    from fractalsig.detection.p3_train_eval import load_matrix, build_folds
    df, groups = load_matrix()
    splits = json.load(open(D("data", "splits_v1.json")))
    folds = {f"{s}/{f}": (tr, va, te) for s, f, tr, va, te in build_folds(df, splits)}
    tr, va, te = folds["random/random"]
    x = df[col_name].to_numpy(float)
    y = df["y"].to_numpy()
    notnan = ~np.isnan(x)
    va2 = va & notnan
    auc_pos = auroc(y[va2], x[va2])
    auc_neg = auroc(y[va2], -x[va2])
    sign = 1.0 if auc_pos >= auc_neg else -1.0
    ids = df["id"].to_numpy()
    scores = {i: sign * v for i, v, ok in zip(ids, x, notnan) if ok}
    return scores, sign


def paired_eval(y_true, score_a, score_b, n_boot=2000):
    a_point, a_lo, a_hi, a_k = bootstrap_ci_paired(auroc, y_true, score_a, score_b, n_boot=n_boot)
    t5_point, t5_lo, t5_hi, t5_k = bootstrap_ci_paired(tpr_at_5pct_fpr, y_true, score_a, score_b, n_boot=n_boot)
    t1_point, t1_lo, t1_hi, t1_k = bootstrap_ci_paired(tpr_at_1pct_fpr, y_true, score_a, score_b, n_boot=n_boot)
    auc_a, auc_b, z, p = delong_test(y_true, score_a, score_b)
    bootstrap_auroc_diff = auroc(y_true, score_a) - auroc(y_true, score_b)
    delong_diff = auc_a - auc_b
    agree = abs(bootstrap_auroc_diff - delong_diff) < 1e-6
    return {
        "n": int(len(y_true)), "n_pos": int(y_true.sum()), "n_neg": int((y_true == 0).sum()),
        "auroc_a": float(auroc(y_true, score_a)), "auroc_b": float(auroc(y_true, score_b)),
        "auroc_diff_bootstrap": {"point": a_point, "lo": a_lo, "hi": a_hi, "n_valid": a_k,
                                 "significant": bool(a_lo > 0 or a_hi < 0)},
        "auroc_diff_delong": {"z": z, "p_value": p, "significant": bool(p < 0.05)},
        "delong_vs_bootstrap_agree": agree,
        "tpr5_diff_bootstrap": {"point": t5_point, "lo": t5_lo, "hi": t5_hi, "n_valid": t5_k,
                                "significant": bool(t5_lo > 0 or t5_hi < 0)},
        "tpr1_diff_bootstrap": {"point": t1_point, "lo": t1_lo, "hi": t1_hi, "n_valid": t1_k,
                                "significant": bool(t1_lo > 0 or t1_hi < 0),
                                "note": "low-resolution: config 5 test fold has ~1,846 human rows total"},
    }


def main():
    config5 = load_config5_predictions()
    results = {}

    baselines = {
        "fastdetectgpt": lambda: load_baseline_scores("fastdetectgpt"),
        "perplexity_gemma": lambda: load_perplexity_scores("Gemma_LogPPL_Mean")[0],
        "perplexity_mistral": lambda: load_perplexity_scores("Mistral_LogPPL_Mean")[0],
        # Cross_PPL_Ratio is one of the detector's own 54 Family-B features
        # (exp(Gemma_LogPPL_Mean - Mistral_LogPPL_Mean), see
        # fractalsig/pipeline/build_p1_join_feature_store.py:add_cross_perplexity). Scored
        # and paired-tested identically to the other single-column
        # baselines for comparability, but this is an ABLATION of an
        # existing detector feature, NOT an independent baseline, and must
        # never be reported as Binoculars or as external validation --
        # comparing the full detector against one of its own inputs in
        # isolation is expected to look good and proves nothing about
        # generalization the detector doesn't already claim for itself.
        "cross_ppl_ratio_ABLATION": lambda: load_perplexity_scores("Cross_PPL_Ratio")[0],
    }
    bino_ckpt = D("data", "binoculars_shard_checkpoint.txt")
    bino_done = (os.path.exists(bino_ckpt)
                and sum(1 for _ in open(bino_ckpt)) >= 721)
    if bino_done:
        baselines["binoculars"] = lambda: load_baseline_scores("binoculars")
    else:
        n_done = sum(1 for _ in open(bino_ckpt)) if os.path.exists(bino_ckpt) else 0
        print(f"[part_d_paired] binoculars still running ({n_done}/721 shards) -- "
              f"skipping to avoid a partial-coverage pairing. Re-run once complete.", flush=True)

    for bname, loader in baselines.items():
        bscores = loader()
        results[bname] = {}
        for (scheme, fold), cfg5_df in config5.items():
            # explicit id-keyed inner join, sorted by id -- never zip two lists
            joined = cfg5_df.copy()
            joined["baseline_score"] = joined.index.map(bscores)
            joined = joined.dropna(subset=["baseline_score"])
            joined = joined.sort_index()
            if len(joined) < 200 or joined["y"].nunique() < 2:
                print(f"[part_d_paired] {bname}/{scheme}/{fold}: n={len(joined)}, skipping (too small or one class)")
                continue
            y_true = joined["y"].to_numpy()
            score_detector = joined["score"].to_numpy()
            score_baseline = joined["baseline_score"].to_numpy()
            r = paired_eval(y_true, score_detector, score_baseline)
            r["scheme"], r["fold"] = scheme, fold
            results[bname][f"{scheme}/{fold}"] = r
            flag = "AGREE" if r["delong_vs_bootstrap_agree"] else "MISMATCH -- CHECK PAIRING"
            print(f"[{bname:<20s}] {scheme}/{fold:<18s} n={r['n']:6d} (n_pos={r['n_pos']}, n_neg={r['n_neg']})  "
                  f"detector_AUROC={r['auroc_a']:.4f} baseline_AUROC={r['auroc_b']:.4f}  "
                  f"DeLong_p={r['auroc_diff_delong']['p_value']:.2e}  [{flag}]", flush=True)

    # RoBERTa: ONLY the 3 folds where its score distribution is real
    # (random/random, logo/gemma_2b, logo/mistral_7b -- std 0.14-0.27).
    # All 5 LODO folds are excluded: RoBERTa's output collapses to a
    # near-constant distribution when the test domain is entirely absent
    # from training (std 0.0000-0.019, confirmed not a class-weight or
    # fp16 artifact -- see 2026-08-25_P3_external_baselines_D_progress.md).
    # A paired CI against a saturated near-constant vector would be
    # arithmetic, not evidence, so those folds are not run at all here
    # rather than run-and-flagged.
    results["roberta"] = {}
    for scheme, fold in [("random", "random"), ("logo", "gemma_2b"), ("logo", "mistral_7b")]:
        if (scheme, fold) not in config5:
            continue
        bscores = load_roberta_scores_for_fold(scheme, fold)
        cfg5_df = config5[(scheme, fold)]
        joined = cfg5_df.copy()
        joined["baseline_score"] = joined.index.map(bscores)
        joined = joined.dropna(subset=["baseline_score"])
        joined = joined.sort_index()
        if len(joined) < 200 or joined["y"].nunique() < 2:
            print(f"[part_d_paired] roberta/{scheme}/{fold}: n={len(joined)}, skipping (too small or one class)")
            continue
        y_true = joined["y"].to_numpy()
        score_detector = joined["score"].to_numpy()
        score_baseline = joined["baseline_score"].to_numpy()
        r = paired_eval(y_true, score_detector, score_baseline)
        r["scheme"], r["fold"] = scheme, fold
        results["roberta"][f"{scheme}/{fold}"] = r
        flag = "AGREE" if r["delong_vs_bootstrap_agree"] else "MISMATCH -- CHECK PAIRING"
        print(f"[{'roberta':<20s}] {scheme}/{fold:<18s} n={r['n']:6d} (n_pos={r['n_pos']}, n_neg={r['n_neg']})  "
              f"detector_AUROC={r['auroc_a']:.4f} baseline_AUROC={r['auroc_b']:.4f}  "
              f"DeLong_p={r['auroc_diff_delong']['p_value']:.2e}  [{flag}]", flush=True)
    # RoBERTa LODO folds, PER SEED (2026-08-25 multi-seed rerun landed --
    # margin-based scores are real, but big_patent's 3 seeds span
    # worse-than-chance to excellent, so a single paired comparison would
    # misrepresent the fold. Run all seeds separately per fold and report
    # the range of results, not one arbitrarily-chosen seed. big_patent gets
    # all 10 (Task 1 extended it to 10; internal review 2026-08-26 flagged
    # that leaving paired-sig at 3/10 here while everything else about
    # big_patent is n=10 was an inconsistency worth closing -- cheap CPU
    # re-run, not new GPU work).
    SEEDS_BY_FOLD = {
        "big_patent": [20260825, 20260826, 20260827, 20260828, 20260829,
                       20260830, 20260831, 20260832, 20260833, 20260834],
        "billsum": [20260825, 20260826, 20260827],
        "newsroom": [20260825, 20260826, 20260827],
        "scientific_papers": [20260825, 20260826, 20260827],
        "wikipedia": [20260825, 20260826, 20260827],
    }
    results["roberta_lodo_per_seed"] = {}
    for fold in ["big_patent", "billsum", "newsroom", "scientific_papers", "wikipedia"]:
        scheme = "lodo"
        if (scheme, fold) not in config5:
            continue
        cfg5_df = config5[(scheme, fold)]
        results["roberta_lodo_per_seed"][fold] = {}
        for seed in SEEDS_BY_FOLD[fold]:
            bscores = load_roberta_margin_scores(scheme, fold, seed)
            joined = cfg5_df.copy()
            joined["baseline_score"] = joined.index.map(bscores)
            joined = joined.dropna(subset=["baseline_score"])
            joined = joined.sort_index()
            if len(joined) < 200 or joined["y"].nunique() < 2:
                print(f"[part_d_paired] roberta_lodo/{fold}/seed{seed}: n={len(joined)}, skipping")
                continue
            y_true = joined["y"].to_numpy()
            score_detector = joined["score"].to_numpy()
            score_baseline = joined["baseline_score"].to_numpy()
            r = paired_eval(y_true, score_detector, score_baseline)
            r["scheme"], r["fold"], r["seed"] = scheme, fold, seed
            results["roberta_lodo_per_seed"][fold][str(seed)] = r
            flag = "AGREE" if r["delong_vs_bootstrap_agree"] else "MISMATCH -- CHECK PAIRING"
            sig = "SIG" if r["auroc_diff_delong"]["significant"] else "n.s."
            print(f"[roberta_lodo(margin) ] lodo/{fold:<18s} seed={seed}  n={r['n']:6d}  "
                  f"detector_AUROC={r['auroc_a']:.4f} roberta_AUROC(margin)={r['auroc_b']:.4f}  "
                  f"DeLong_p={r['auroc_diff_delong']['p_value']:.2e} [{sig}]  [{flag}]", flush=True)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"[part_d_paired] written to {OUT_PATH}")


if __name__ == "__main__":
    main()
