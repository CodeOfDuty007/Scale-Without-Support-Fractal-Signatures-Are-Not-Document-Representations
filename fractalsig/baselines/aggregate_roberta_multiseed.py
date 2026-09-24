#!/usr/bin/env python3
"""Aggregate the multi-seed LODO reruns into per-fold mean/range tables.
Reads data/roberta_summaries/lodo_<fold>_marginfp32_seed<seed>.json for each
(fold, seed) combination. Reports margin-based AUROC/TPR@FPR (the metric
that's actually trustworthy per the 2026-08-25 retraction) as mean +/- range
across seeds, plus the raw per-seed values so the spread itself is visible,
not just summarized away.

big_patent gets 10 seeds (Task 1, the 9-task plan, 2026-08-26 -- the
original n=3 was an impression, not a measured rate, and this claim is
heading into the paper as a methodological point about single-seed
reporting). The other 4 LODO folds keep the original 3-seed wave.
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os

HERE = _paths.WORKSPACE_STR
D = lambda *p: os.path.join(HERE, *p)

SEEDS_BY_FOLD = {
    "big_patent": [20260825, 20260826, 20260827, 20260828, 20260829,
                   20260830, 20260831, 20260832, 20260833, 20260834],
    "billsum": [20260825, 20260826, 20260827],
    "newsroom": [20260825, 20260826, 20260827],
    "scientific_papers": [20260825, 20260826, 20260827],
    "wikipedia": [20260825, 20260826, 20260827],
}
FOLDS = list(SEEDS_BY_FOLD)


def main():
    print(f"{'fold':<20s} {'metric':<14s} {'mean':>8s} {'min':>8s} {'max':>8s} {'range':>8s}  per-seed values")
    rows = []
    for fold in FOLDS:
        SEEDS = SEEDS_BY_FOLD[fold]
        vals = {"margin_auroc": [], "margin_tpr1": [], "margin_tpr5": [],
                "margin_std": [], "prob_std": []}
        missing = []
        for seed in SEEDS:
            path = D("data", "roberta_summaries", f"lodo_{fold}_marginfp32_seed{seed}.json")
            if not os.path.exists(path):
                missing.append(seed)
                continue
            d = json.load(open(path))
            vals["margin_auroc"].append(d["logit_margin_auroc"])
            vals["margin_tpr1"].append(d["logit_margin_tpr_at_1pct_fpr"])
            vals["margin_tpr5"].append(d["logit_margin_tpr_at_5pct_fpr"])
            vals["margin_std"].append(d["logit_margin_std"])
            vals["prob_std"].append(d["score_std_calibrated"])
        n_total = len(SEEDS)
        if missing:
            print(f"{fold:<20s} -- missing seeds {missing}, {n_total-len(missing)}/{n_total} landed so far")
        for metric, xs in vals.items():
            if not xs:
                continue
            mean = sum(xs) / len(xs)
            lo, hi = min(xs), max(xs)
            print(f"{fold:<20s} {metric:<14s} {mean:8.4f} {lo:8.4f} {hi:8.4f} {hi-lo:8.4f}  {[round(x,4) for x in xs]}")
        if vals["margin_auroc"]:
            n_below_chance = sum(1 for a in vals["margin_auroc"] if a < 0.5)
            n_seeds = len(vals["margin_auroc"])
            print(f"{fold:<20s} -- {n_below_chance}/{n_seeds} seeds landed BELOW-CHANCE AUROC "
                  f"({100*n_below_chance/n_seeds:.0f}%)")
        rows.append({"fold": fold, "n_seeds": len(vals["margin_auroc"]), **{k: v for k, v in vals.items()}})
        print()

    with open(D("data", "roberta_multiseed_aggregate.json"), "w") as f:
        json.dump(rows, f, indent=2, default=float)
    print("written to data/roberta_multiseed_aggregate.json")


if __name__ == "__main__":
    main()
