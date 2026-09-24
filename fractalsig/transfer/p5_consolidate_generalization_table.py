#!/usr/bin/env python3
"""P5 close-out plan, step 3: pull every generalization result this project
has produced -- P3's LODO/LOGO (config=full, arm=all, lgbm, the model P5
task4 reuses directly), task4 (GAGLE->RAID binary transfer), T2 (GAGLE
gemma_2b-vs-mistral_7b), task6 (RAID 11-class generator ID), task3 (GAGLE
8-class prompt-type ID, both arms) -- into one consolidation table. Pure
aggregation of numbers already computed and ledgered elsewhere: no new
training, no new metrics.

Three different population conventions meet in this one table (flagged by
internal review, 2026-08-29, before this script was written):
  - P3's LODO/LOGO: GAGLE, config=full, native-NaN (LightGBM handles NaN
    without dropping rows) -- NOT complete-case.
  - T2 / task3: GAGLE, complete-case (rows with any NaN feature dropped).
  - task4 / task6: RAID, complete-case, AND (as of the 2026-08-29 ZWSP fix)
    still has an attack-correlated complete-case drop rate -- zero_width_space
    rows are dropped at 84.3% vs a 64.7% baseline for other RAID attacks
    (residual gap is the untouched fractal-scale columns, not a bug -- see
    ledger). Every row below states its population explicitly; do not
    compare `test n` across rows as if they meant the same thing.
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os

HERE = _paths.WORKSPACE_STR
D = lambda *p: os.path.join(HERE, "data", *p)
R = lambda *p: os.path.join(HERE, "results", *p)


def fmt(pt, lo=None, hi=None, k=None):
    if lo is None:
        return f"{pt:.4f}"
    s = f"{pt:.4f} [{lo:.4f}, {hi:.4f}]"
    if k is not None and k < 2000:
        s += f" (†{k})"
    return s


def main():
    lines = []
    lines.append("# P5 multi-task generalization summary\n")
    lines.append(
        "Consolidation of every generalization/transfer result this project has "
        "produced, pulled from already-computed, already-ledgered outputs -- no "
        "new training or metrics in this table. See "
        "`PROJECT_STATE_AND_HPC_MIGRATION.md`'s CHANGE LOG for full provenance "
        "and investigation detail behind each row.\n"
    )
    lines.append(
        "**CIs are 95% percentile bootstrap** (binary metrics: stratified by "
        "positive/negative class; multiclass macro-F1: unstratified, since "
        "`fractalsig/detection/metrics.py`'s stratified path is binary-specific) from the "
        "frozen `fractalsig/detection/metrics.py`, `n_boot=2000`. `†k` marks an interval "
        "resting on fewer than 2,000 usable bootstrap folds -- read the "
        "interval as less precise than the bracket alone suggests, not as "
        "invalid.\n"
    )
    lines.append(
        "**Three different populations appear below -- `population` column "
        "names which one for every row, because `test n` is not comparable "
        "across them:**\n"
        "- **GAGLE native-NaN**: P3's LODO/LOGO rows, LightGBM's own "
        "config=full grid -- NaN features handled natively, no rows dropped.\n"
        "- **GAGLE complete-case**: T2 and task3 -- any row with a NaN "
        "feature is dropped before training/eval.\n"
        "- **RAID complete-case (ZWSP-affected)**: task4 and task6 -- "
        "complete-case on the post-2026-08-29-fix RAID feature store; "
        "`zero_width_space` rows are still dropped at 84.3% (vs 64.7% "
        "baseline for other RAID attacks) due to fractal-scale columns "
        "genuinely needing more real tokens than a ZWSP-inflated document "
        "supplies -- a real consequence of the attack, not a residual bug. "
        "**For the `zero_width_space` rows that DO survive complete-case, "
        "features are a deliberate hybrid, not uniformly attack-blind or "
        "attack-aware**: style/embedding features (sentence/paragraph stats, "
        "TTR, embedding dispersion) are computed on U+200B-stripped text, "
        "while fractal/perplexity features (Family A/B) come from the "
        "earlier GPU rescore pass, which scored the true ZWSP-present token "
        "stream -- so a `zero_width_space` row's style features answer \"how "
        "does this document read once the disguise is removed\" while its "
        "fractal/perplexity features answer \"what did the attack actually "
        "do to the model's view of this document,\" in the same feature "
        "vector. This is why task4's RAID recall numbers should not be read "
        "as a single coherent judgment of ZWSP's effect -- part of the "
        "feature vector already defeats it, part still sees it.\n"
    )

    # --- GAGLE in-distribution + LODO/LOGO (P3, config=full/arm=all/lgbm) ---
    t1 = json.load(open(D("p3_t1_results.json")))
    cells = [c for c in t1["cells"]
             if c.get("config") == "full" and c.get("arm") == "all" and c.get("learner") == "lgbm"]
    order = ["random/random", "lodo/big_patent", "lodo/billsum", "lodo/newsroom",
             "lodo/scientific_papers", "lodo/wikipedia", "logo/gemma_2b", "logo/mistral_7b"]
    cell_by_key = {f"{c['scheme']}/{c['fold']}": c for c in cells}

    lines.append("\n## 1. Binary LLM-vs-human detection: in-distribution and generalization gap\n")
    lines.append(
        "GAGLE, config=full (native-NaN), arm=`all` (85 features), LightGBM, "
        "D-11 pipeline. Positive class: LLM. Source: `data/p3_t1_results.json` "
        "(already-ledgered, unchanged by this table).\n"
    )
    lines.append("| scheme/fold | population | test n | AUROC | TPR@1%FPR | TPR@5%FPR |")
    lines.append("|---|---|---|---|---|---|")
    for key in order:
        c = cell_by_key[key]
        ci = c["ci95"]
        lines.append(
            f"| {key} | GAGLE native-NaN | {c['n']:,} | "
            f"{fmt(ci['auroc']['point'], ci['auroc']['lo'], ci['auroc']['hi'], ci['auroc']['n_valid_folds'])} | "
            f"{fmt(ci['tpr_at_1pct_fpr']['point'], ci['tpr_at_1pct_fpr']['lo'], ci['tpr_at_1pct_fpr']['hi'], ci['tpr_at_1pct_fpr']['n_valid_folds'])} | "
            f"{fmt(ci['tpr_at_5pct_fpr']['point'], ci['tpr_at_5pct_fpr']['lo'], ci['tpr_at_5pct_fpr']['hi'], ci['tpr_at_5pct_fpr']['n_valid_folds'])} |"
        )

    # --- Task 4: GAGLE -> RAID cross-corpus transfer ---
    t4 = json.load(open(D("p5_task4_gagle_to_raid_transfer.json")))
    gagle_id = t4["gagle_in_distribution"]
    prim = t4["raid_native_threshold_PRIMARY"]
    lines.append("\n## 2. Cross-corpus transfer: GAGLE-trained model scored on RAID (task 4)\n")
    lines.append(
        "Same `full/random/random/all/lgbm` model reused directly (no retraining). "
        "RAID-native-threshold PRIMARY numbers: threshold re-derived on RAID's own "
        "human population (the GAGLE-calibrated threshold is miscalibrated on RAID's "
        "different human population -- see ledger 2026-08-28). "
        "Source: `data/p5_task4_gagle_to_raid_transfer.json`. The GAGLE row below "
        "is the same cell as §1's random/random row, repeated here for direct "
        "comparison against the RAID row.\n"
    )
    gagle_cell_r = cell_by_key["random/random"]
    gagle_ci_r = gagle_cell_r["ci95"]
    lines.append("| eval corpus | population | test n | AUROC | TPR@1%FPR | TPR@5%FPR |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(
        f"| GAGLE (in-distribution) | GAGLE native-NaN | {gagle_id['n']:,} | "
        f"{fmt(gagle_ci_r['auroc']['point'], gagle_ci_r['auroc']['lo'], gagle_ci_r['auroc']['hi'], gagle_ci_r['auroc']['n_valid_folds'])} | "
        f"{fmt(gagle_ci_r['tpr_at_1pct_fpr']['point'], gagle_ci_r['tpr_at_1pct_fpr']['lo'], gagle_ci_r['tpr_at_1pct_fpr']['hi'], gagle_ci_r['tpr_at_1pct_fpr']['n_valid_folds'])} | "
        f"{fmt(gagle_ci_r['tpr_at_5pct_fpr']['point'], gagle_ci_r['tpr_at_5pct_fpr']['lo'], gagle_ci_r['tpr_at_5pct_fpr']['hi'], gagle_ci_r['tpr_at_5pct_fpr']['n_valid_folds'])} |"
    )
    lines.append(
        f"| RAID (cross-corpus transfer) | RAID complete-case (ZWSP-affected) | {prim['n_test']:,} | "
        f"{fmt(prim['auroc'], prim['auroc_ci'][0], prim['auroc_ci'][1], prim['auroc_ci_n_valid_folds'])} | "
        f"{fmt(prim['tpr_1pct'], prim['tpr_1pct_ci'][0], prim['tpr_1pct_ci'][1], prim['tpr_1pct_ci_n_valid_folds'])} | "
        f"{fmt(prim['tpr_5pct'], prim['tpr_5pct_ci'][0], prim['tpr_5pct_ci'][1], prim['tpr_5pct_ci_n_valid_folds'])} |"
    )

    # --- T2: GAGLE binary generator ID ---
    t2 = json.load(open(D("p5_task2_gagle_generator_id.json")))
    lines.append("\n## 3. Generator-family identification\n")
    lines.append(
        "Source: `data/p5_task2_gagle_generator_id.json` (T2) and "
        "`data/p5_task6_generator_family_id.json` (task 6). Paper's own RF "
        "baseline for the 2-class case is ~97.8%, a different model class on "
        "a different corpus construction -- reported in the ledger for "
        "context only, not as a target.\n"
    )
    lines.append("| task | population | test n | classes | metric | value |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(
        f"| T2: GAGLE gemma_2b vs mistral_7b | GAGLE complete-case | {t2['n_test']:,} | 2 | AUROC | "
        f"{fmt(t2['auroc'], t2['auroc_ci'][0], t2['auroc_ci'][1], t2['auroc_ci_n_valid_folds'])} |"
    )
    lines.append(
        f"| T2: GAGLE gemma_2b vs mistral_7b | GAGLE complete-case | {t2['n_test']:,} | 2 | TPR@1%FPR | "
        f"{fmt(t2['tpr_1pct'], t2['tpr_1pct_ci'][0], t2['tpr_1pct_ci'][1], t2['tpr_1pct_ci_n_valid_folds'])} |"
    )
    lines.append(
        f"| T2: GAGLE gemma_2b vs mistral_7b | GAGLE complete-case | {t2['n_test']:,} | 2 | TPR@5%FPR | "
        f"{fmt(t2['tpr_5pct'], t2['tpr_5pct_ci'][0], t2['tpr_5pct_ci'][1], t2['tpr_5pct_ci_n_valid_folds'])} |"
    )

    t6 = json.load(open(D("p5_task6_generator_family_id.json")))
    lines.append(
        f"| task 6: RAID 11-class generator ID | RAID complete-case (ZWSP-affected) | {t6['n_test']:,} | 11 | macro-F1 | "
        f"{fmt(t6['macro_f1_test'], t6['macro_f1_ci'][0], t6['macro_f1_ci'][1], t6['macro_f1_ci_n_valid_folds'])} |"
    )

    # --- Task 3: GAGLE 8-class prompt-type ID (full vs fractal_only) ---
    t3 = json.load(open(D("p5_task3_prompt_type_id.json")))
    lines.append("\n## 4. Prompt-type identification (task 3) -- fractal-only arm reinforces the project headline\n")
    lines.append(
        "Same `random/random` frozen split T2 reuses, LLM-only rows, GAGLE. "
        "The `fractal_only` arm's collapse relative to `all_features` is this "
        "project's own headline finding (deep fractal features underperform a "
        "lightweight baseline) reinforced a third independent way. "
        "Source: `data/p5_task3_prompt_type_id.json`.\n"
    )
    lines.append("| arm | population | test n | classes | macro-F1 |")
    lines.append("|---|---|---|---|---|")
    ci_a = t3["macro_f1_all_features_ci"]
    ci_f = t3["macro_f1_fractal_only_ci"]
    lines.append(
        f"| all features (85) | GAGLE complete-case (85-feat) | {t3['n_test']:,} | 8 | "
        f"{fmt(t3['macro_f1_test_all_features'], ci_a[0], ci_a[1], t3['macro_f1_all_features_ci_n_valid_folds'])} |"
    )
    lines.append(
        f"| fractal_only (54) | GAGLE complete-case (54-feat) | {t3['n_test_fractal_only']:,} | 8 | "
        f"{fmt(t3['macro_f1_test_fractal_only'], ci_f[0], ci_f[1], t3['macro_f1_fractal_only_ci_n_valid_folds'])} |"
    )
    lines.append(
        "\n*Note: the two rows above have different `test n` (14,805 vs 14,919) "
        "because complete-case is computed over a different feature count each "
        "time (85 vs 54) -- rows with a NaN only in a stat/style column survive "
        "the narrower `fractal_only` filter. The 114-row difference does not "
        "explain the 0.5847 vs 0.1981 gap.*\n"
    )

    out = "\n".join(lines) + "\n"
    with open(R("p5_multitask_transfer.md"), "w") as f:
        f.write(out)
    print(f"[consolidate] written to results/p5_multitask_transfer.md ({len(lines)} lines)")


if __name__ == "__main__":
    main()
