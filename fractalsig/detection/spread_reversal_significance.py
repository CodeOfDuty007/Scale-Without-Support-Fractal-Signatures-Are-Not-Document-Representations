#!/usr/bin/env python3
"""
Task 1 / Test 6 -- Is the SPREAD reversal statistically significant?

BACKGROUND
  The original paper's Figure 10 claims human writing has a NARROWER spread of
  S than LLM writing. the internal verification notes already discloses
  that this did not replicate here -- the reproduction found the opposite
  direction. What has never existed is a significance test on the SPREAD claim
  itself.

  This is deliberately NOT Test 4. reports/2026-08-05_test4_statistical_
  significance.md tested whether the MEAN gap (LLM minus Human) differs from
  zero. Same corpus, same per-domain/per-scorer breakdown, same bootstrap
  philosophy -- retargeted from the mean to the VARIANCE.

METHOD
  1. Variance ratio  Var(LLM) / Var(Human)  per (domain, scorer, exponent),
     with a percentile bootstrap CI: resample documents with replacement
     WITHIN each group, recompute the ratio per fold.
     A CI excluding 1.0 is the "genuinely more/less varied" result.
  2. Brown-Forsythe (Levene centred on the median, robust to non-normality) as
     an INDEPENDENT second check. Both are reported; neither is relied on alone.
  3. LENGTH CONTROL. Per-doc S is known biased at ~400 tokens, and if the bias
     depends on length while Human and LLM differ in length variability, an
     apparent spread difference could be a length artifact. The test is
     therefore repeated on residuals of S/H regressed on log document length,
     and again within length quintiles.

  B = 2000, not Test 4's 10. Test 4 used 10 outer folds because each fold
  required re-pooling trajectories through calc_S -- expensive. Here we
  resample per-document scalars, which is cheap, and a 10-fold percentile
  interval on a variance RATIO would be meaninglessly coarse.

PREREQUISITES (both now satisfied, and both were genuine blockers)
  - X-01: the human population must be the deduplicated 5,000, not the
    2.34x-inflated 11,696 -- duplicate inflation distorts a variance estimate
    in an unknown direction.
  - F-4: per-doc S must be deterministic. Before 2026-08-23 every S carried
    Monte-Carlo noise with a per-document sd of 0.029-0.080 against a corpus sd
    of 0.28; a variance-ratio test on that data would have been measuring the
    estimator's own noise, and the noise inflates BOTH groups' variance, biasing
    the ratio toward 1.0.

Reads the finished feature store only. Per-column masking, no row dropping
(design decision 2026-08-23), so each cell reports its own n.
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = _paths.WORKSPACE_STR
PARQUET = os.path.join(HERE, "feature_store", "v1_gagle.parquet")
OUT_JSON = os.path.join(HERE, "data", "spread_reversal_significance.json")

B = 2000
SEED = 20260823
HUMAN = "Human"
LENGTH_COL = "N_Words_Used"
EXPONENTS = [("Gemma_S", "S", "gemma-2b"), ("Gemma_H", "H", "gemma-2b"),
             ("Mistral_S", "S", "mistral-7b"), ("Mistral_H", "H", "mistral-7b")]


def var_ratio_ci(llm, hum, rng, B=B):
    """Percentile CI for Var(LLM)/Var(Human). Resamples each group
    independently, which is the correct null structure -- the two groups are
    not paired."""
    if len(llm) < 3 or len(hum) < 3:
        return None
    li = rng.integers(0, len(llm), size=(B, len(llm)))
    hi = rng.integers(0, len(hum), size=(B, len(hum)))
    vl = llm[li].var(axis=1, ddof=1)
    vh = hum[hi].var(axis=1, ddof=1)
    good = vh > 0
    ratios = vl[good] / vh[good]
    if len(ratios) < B // 2:
        return None
    lo, hi_ = np.percentile(ratios, [2.5, 97.5])
    return {"ratio": float(llm.var(ddof=1) / hum.var(ddof=1)),
            "ci_lo": float(lo), "ci_hi": float(hi_),
            "excludes_1": bool(lo > 1.0 or hi_ < 1.0),
            "n_llm": int(len(llm)), "n_human": int(len(hum))}


def residualise(df, col, length):
    """Regress col on log(length) and return residuals. Removes any
    length-proportional component before the variance is measured."""
    m = df[col].notna() & df[length].notna() & (df[length] > 0)
    if m.sum() < 10:
        return pd.Series(np.nan, index=df.index)
    x = np.log(df.loc[m, length].astype(float).to_numpy())
    y = df.loc[m, col].astype(float).to_numpy()
    slope, intercept, *_ = stats.linregress(x, y)
    out = pd.Series(np.nan, index=df.index)
    out.loc[m] = y - (slope * x + intercept)
    return out


def run(df, tag, rng, results):
    rows = []
    for domain in sorted(df["domain"].unique()):
        sub = df[df["domain"] == domain]
        for col, exp, scorer in EXPONENTS:
            if col not in sub.columns:
                continue
            hum = sub.loc[sub["Label"] == HUMAN, col].dropna().to_numpy(float)
            llm = sub.loc[sub["Label"] != HUMAN, col].dropna().to_numpy(float)
            r = var_ratio_ci(llm, hum, rng)
            if r is None:
                continue
            try:
                lev = stats.levene(llm, hum, center="median")
                lev_p = float(lev.pvalue)
            except Exception:
                lev_p = float("nan")
            r.update({"domain": domain, "scorer": scorer, "exponent": exp,
                      "levene_p": lev_p, "variant": tag})
            rows.append(r)
            results.append(r)
    return rows


def show(rows, title):
    print("\n" + "=" * 104)
    print(title)
    print("=" * 104)
    print(f"{'domain':<20}{'scorer':<12}{'exp':<5}{'n LLM':>8}{'n Hum':>7}"
          f"{'Var ratio':>11}{'95% CI':>22}{'excl 1?':>9}{'Levene p':>11}")
    print("-" * 104)
    for r in rows:
        ci = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
        star = "YES" if r["excludes_1"] else "no"
        print(f"{r['domain']:<20}{r['scorer']:<12}{r['exponent']:<5}"
              f"{r['n_llm']:>8,}{r['n_human']:>7,}{r['ratio']:>11.3f}{ci:>22}"
              f"{star:>9}{r['levene_p']:>11.3g}")


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(PARQUET)
    print(f"[info] {len(df):,} rows from {PARQUET}")
    print(f"[info] B={B} bootstrap reps, seed={SEED}")
    results = []

    raw = run(df, "raw", rng, results)
    show(raw, "1. RAW  --  Var(LLM) / Var(Human).  ratio > 1 => LLM MORE varied "
              "(i.e. the paper's Fig-10 direction REVERSED)")

    # --- length control A: residualise on log length -----------------------
    d2 = df.copy()
    for col, _, _ in EXPONENTS:
        if col in d2.columns:
            d2[col] = residualise(d2, col, LENGTH_COL)
    res = run(d2, "length_residualised", rng, results)
    show(res, f"2. LENGTH-RESIDUALISED  --  same test on residuals of "
              f"S/H ~ log({LENGTH_COL})")

    # --- length control B: within length quintiles --------------------------
    print("\n" + "=" * 104)
    print("3. WITHIN LENGTH QUINTILES  --  does the direction survive stratification?")
    print("=" * 104)
    q = pd.qcut(df[LENGTH_COL], 5, labels=False, duplicates="drop")
    strat = []
    for col, exp, scorer in EXPONENTS:
        if col not in df.columns:
            continue
        signs = []
        for qi in sorted(pd.unique(q.dropna())):
            sub = df[q == qi]
            hum = sub.loc[sub["Label"] == HUMAN, col].dropna().to_numpy(float)
            llm = sub.loc[sub["Label"] != HUMAN, col].dropna().to_numpy(float)
            r = var_ratio_ci(llm, hum, rng, B=500)
            if r:
                signs.append(r["ratio"])
        if signs:
            agree = sum(1 for s in signs if s > 1)
            strat.append({"column": col, "exponent": exp, "scorer": scorer,
                          "quintile_ratios": signs,
                          "n_quintiles_ratio_gt_1": agree,
                          "n_quintiles": len(signs)})
            print(f"  {col:<14} ratios by quintile: "
                  + "  ".join(f"{s:.3f}" for s in signs)
                  + f"   ({agree}/{len(signs)} above 1.0)")

    n_excl = sum(1 for r in raw if r["excludes_1"])
    n_gt1 = sum(1 for r in raw if r["ratio"] > 1)
    n_excl_res = sum(1 for r in res if r["excludes_1"])
    print("\n" + "=" * 104)
    print("VERDICT")
    print("=" * 104)
    print(f"  raw            : {n_gt1}/{len(raw)} cells have Var(LLM) > Var(Human); "
          f"{n_excl}/{len(raw)} have a 95% CI excluding 1.0")
    print(f"  length-adjusted: {n_excl_res}/{len(res)} cells still exclude 1.0")
    print(f"  A ratio > 1 means LLM text is MORE varied than human text -- the")
    print(f"  OPPOSITE of the original Figure 10 claim. Cells whose CI excludes")
    print(f"  1.0 are the ones where that reversal is statistically supported.")

    json.dump({"B": B, "seed": SEED, "cells": results, "quintiles": strat},
              open(OUT_JSON, "w"), indent=2)
    print(f"\n[written] {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
