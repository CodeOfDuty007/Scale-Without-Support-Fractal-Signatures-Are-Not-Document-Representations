#!/usr/bin/env python3
"""
T9 -- RAID complete-case sensitivity, cost-ordered items (1) and (2) only
(items (3) IPW-reweighting and (4) reduced-feature-set-for-more-rows are
left for a future session -- explicitly the plan's own next-lowest-value
items in its cost ordering).

(1) Included-vs-excluded population comparison table: how does the
    complete-case population (n=22,041, no NaN in any of the 85 features)
    differ in composition from the excluded rows (n=46,171, at least one
    NaN feature) by Label(generator), domain, and attack type?
(2) Transfer AUROC by attack/domain/generator on the complete-case
    population, using the SAME frozen model as T5/p5_task4 (no retraining),
    reporting only where n permits (>=100 per group, both classes present).
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fractalsig import paths as _paths  # noqa: E402

PAPER = _paths.WORKSPACE_STR
D = lambda *p: os.path.join(PAPER, "data", *p)
OUT_DIR = _paths.STATS_DIR_STR
MIN_N_PER_GROUP = 100


def main():
    bundle = joblib.load(os.path.join(PAPER, "feature_store", "p3_refit_models",
                                        "full_random_random_all_lgbm.joblib"))
    cal_model, feature_names = bundle["model"], bundle["feature_names"]
    pos = int(np.where(cal_model.classes_ == 1)[0][0])

    rows = [json.loads(l) for l in open(D("p5_raid_features.jsonl"))]
    df = pd.DataFrame(rows)
    X_all = df[feature_names].to_numpy(dtype=float)
    nan_mask = np.isnan(X_all).any(axis=1)
    df["included"] = ~nan_mask
    print(f"[t9] n_total={len(df)}  included(complete-case)={(~nan_mask).sum()}  "
          f"excluded(>=1 NaN feature)={nan_mask.sum()}", flush=True)

    # ---- (1) included-vs-excluded composition table ----
    comp = {}
    for col in ("Label", "domain", "attack"):
        ct = pd.crosstab(df[col], df["included"], normalize="columns")
        comp[col] = {
            "included_pct": {str(k): float(v) for k, v in ct[True].to_dict().items()} if True in ct.columns else {},
            "excluded_pct": {str(k): float(v) for k, v in ct[False].to_dict().items()} if False in ct.columns else {},
        }
        print(f"\n[t9] composition by {col} (included % | excluded %):", flush=True)
        for k in sorted(set(df[col])):
            inc = comp[col]["included_pct"].get(str(k), 0.0)
            exc = comp[col]["excluded_pct"].get(str(k), 0.0)
            print(f"  {k:25} included={inc:.3f}  excluded={exc:.3f}  diff={inc-exc:+.3f}", flush=True)

    # ---- (2) transfer AUROC by attack/domain/generator, complete-case only ----
    inc_df = df[~nan_mask].reset_index(drop=True)
    y_inc = (inc_df["Label"] != "human").astype(int).to_numpy()
    X_inc = inc_df[feature_names].to_numpy(dtype=float)
    p_inc = cal_model.predict_proba(X_inc)[:, pos]

    baseline_auroc = float(roc_auc_score(y_inc, p_inc))
    print(f"\n[t9] complete-case baseline AUROC (sanity, should match T5's 0.8032): {baseline_auroc:.4f}", flush=True)

    by_group = {}
    for col in ("attack", "domain"):
        by_group[col] = {}
        for val, sub in inc_df.groupby(col):
            idx = sub.index.to_numpy()
            yv, pv = y_inc[idx], p_inc[idx]
            n = len(yv)
            if n < MIN_N_PER_GROUP or len(set(yv)) < 2:
                by_group[col][str(val)] = {"n": int(n), "note": "insufficient n or single class"}
                continue
            auc = float(roc_auc_score(yv, pv))
            by_group[col][str(val)] = {"n": int(n), "auroc": auc}
        print(f"\n[t9] complete-case AUROC by {col}:", flush=True)
        for k, v in sorted(by_group[col].items(), key=lambda kv: -kv[1].get("n", 0)):
            if "auroc" in v:
                print(f"  {k:25} n={v['n']:6d}  auroc={v['auroc']:.4f}", flush=True)
            else:
                print(f"  {k:25} n={v['n']:6d}  ({v['note']})", flush=True)

    # Label(generator): AUROC needs both classes -- each machine generator's
    # rows vs the SHARED human pool (matching p5_task4's own by_generator
    # breakdown design), not a group-by-Label-alone (which is single-class
    # per group by construction and would trivially fail every time).
    by_group["generator_vs_human"] = {}
    human_idx = inc_df.index[inc_df["Label"] == "human"].to_numpy()
    for gen in sorted(set(inc_df["Label"]) - {"human"}):
        gen_idx = inc_df.index[inc_df["Label"] == gen].to_numpy()
        idx = np.concatenate([human_idx, gen_idx])
        yv, pv = y_inc[idx], p_inc[idx]
        n = len(yv)
        if n < MIN_N_PER_GROUP or len(set(yv)) < 2:
            by_group["generator_vs_human"][gen] = {"n": int(n), "note": "insufficient n or single class"}
            continue
        auc = float(roc_auc_score(yv, pv))
        by_group["generator_vs_human"][gen] = {"n": int(n), "n_generator_rows": int(len(gen_idx)),
                                                  "n_human_rows": int(len(human_idx)), "auroc": auc}
    print("\n[t9] complete-case AUROC by generator (vs shared human pool):", flush=True)
    for k, v in sorted(by_group["generator_vs_human"].items(), key=lambda kv: -kv[1].get("n", 0)):
        if "auroc" in v:
            print(f"  {k:25} n={v['n']:6d}  auroc={v['auroc']:.4f}", flush=True)
        else:
            print(f"  {k:25} n={v['n']:6d}  ({v['note']})", flush=True)

    out = {
        "n_total": len(df), "n_included": int((~nan_mask).sum()), "n_excluded": int(nan_mask.sum()),
        "included_vs_excluded_composition": comp,
        "complete_case_baseline_auroc": baseline_auroc,
        "auroc_by_group_complete_case": by_group,
        "min_n_per_group": MIN_N_PER_GROUP,
        "note": "Items (3) IPW sensitivity and (4) reduced-feature-set-for-more-rows "
                "are deferred to a future session per the plan's own cost ordering.",
    }
    out_path = os.path.join(OUT_DIR, "t9_raid_complete_case.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n[written] {out_path}", flush=True)


if __name__ == "__main__":
    main()
