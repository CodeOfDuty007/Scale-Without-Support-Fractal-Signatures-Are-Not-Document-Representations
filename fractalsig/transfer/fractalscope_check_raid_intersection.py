#!/usr/bin/env python3
"""Fixes the population-mismatch flagged in internal review: the earlier RAID check
scored all_minus_fractal (31-feat) on its OWN complete-case set (66,638
rows), while Task4's persisted 85-feat number is on ITS OWN complete-case
set (22,041 rows) -- different populations, so the +0.036 AUROC gap could be
pure composition, not a fractal-feature effect. This script restricts BOTH
models to the identical row set (complete-case on the 85-feature schema,
matching p5_task4_gagle_to_raid_transfer.py exactly) and scores both there.
"""
from fractalsig import paths as _paths  # noqa: E402
import json, os, sys
import numpy as np
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve

HERE = _paths.WORKSPACE_STR
from fractalsig.detection.p3_train_eval import load_matrix, build_folds, config_mask, arm_features, fit_predict
from fractalsig.detection.metrics import auroc as auroc_fn, tpr_at_1pct_fpr, tpr_at_5pct_fpr, bootstrap_ci
D = lambda *p: os.path.join(HERE, "data", *p)
FS = lambda *p: os.path.join(HERE, "feature_store", *p)

# ---- retrain all_minus_fractal exactly as before (unchanged) ----
df, groups = load_matrix()
feats_all = groups["fractal"] + groups["stat"] + groups["style"]
splits = json.load(open(D("splits_v1.json")))
folds = list(build_folds(df, splits))
yall = df["y"].to_numpy()
cmask, _ = config_mask(df, "full", feats_all)
for scheme, fold, tr, va, te in folds:
    if scheme == "random" and fold == "random":
        tr2, va2, te2 = tr & cmask, va & cmask, te & cmask
        break
names31 = arm_features("all_minus_fractal", groups)
Xa = df[names31].to_numpy(float)
Xtr, Xva = Xa[tr2], Xa[va2]
ytr, yva = yall[tr2], yall[va2]
_, base31 = fit_predict("lgbm", Xtr, ytr, Xva, yva, Xa[te2], names31)
cal31 = CalibratedClassifierCV(estimator=base31, cv="prefit", method="sigmoid")
cal31.fit(Xva, yva)
pos31 = int(np.where(cal31.classes_ == 1)[0][0])

# ---- load persisted 85-feat model (the one Task4 uses) ----
bundle = joblib.load(os.path.join(HERE, "feature_store", "p3_refit_models",
                                    "full_random_random_all_lgbm.joblib"))
cal85, names85 = bundle["model"], bundle["feature_names"]
pos85 = int(np.where(cal85.classes_ == 1)[0][0])
assert len(names85) == 85, f"expected 85 features, got {len(names85)}"
assert set(names31).issubset(set(names85)), "31-feat arm must be a subset of the 85-feat schema"

# ---- RAID rows, complete-case on the 85-FEATURE schema (Task4's exact rule) ----
rows = [json.loads(l) for l in open(D("p5_raid_features.jsonl"))]
print(f"[intersect] {len(rows)} RAID rows total", flush=True)
y_full = np.array([0 if r["Label"] == "human" else 1 for r in rows])
X85_full = np.array([[r[n] for n in names85] for r in rows], dtype=float)
nan85 = np.isnan(X85_full).any(axis=1)
print(f"[intersect] complete-case on 85-feat schema: {(~nan85).sum()} / {len(rows)} rows "
      f"(Task4's exact population)", flush=True)

# restrict to this intersection; also require 31-feat completeness on the SAME rows
# (any row complete on 85 features is complete on the 31-feat subset by construction)
X85 = X85_full[~nan85]
y = y_full[~nan85]
X31 = np.array([[r[n] for n in names31] for r, keep in zip(rows, ~nan85) if keep], dtype=float)
assert not np.isnan(X31).any(), "31-feat subset of a complete-85-feat row must itself be complete"
print(f"[intersect] final identical population for both models: n={len(y)}", flush=True)

p85 = cal85.predict_proba(X85)[:, pos85]
p31 = cal31.predict_proba(X31)[:, pos31]

# fixed GAGLE-calibrated thresholds, reused exactly as Task4 does (not re-derived on RAID)
tab_thr = json.load(open(D("p4_task5_tabular_thresholds.json")))
t1_85, t5_85 = tab_thr["t1_fpr1pct"], tab_thr["t5_fpr5pct"]
# for the 31-feat model, use ITS OWN val-derived thresholds (same convention, different model)
p_va31_raw = base31.predict_proba(Xva)[:, int(np.where(base31.classes_ == 1)[0][0])]
fpr, tpr, thresh = roc_curve(yva, p_va31_raw)
idx1 = np.where(fpr <= 0.01)[0]
idx5 = np.where(fpr <= 0.05)[0]
t1_31 = thresh[idx1[-1]] if len(idx1) else None
t5_31 = thresh[idx5[-1]] if len(idx5) else None

def report(name, y, p, t1, t5):
    a = roc_auc_score(y, p)
    a_pt, a_lo, a_hi, a_k = bootstrap_ci(auroc_fn, y, p, stratified=True)
    tpr1 = float(np.mean(p[y == 1] >= t1))
    tpr5 = float(np.mean(p[y == 1] >= t5))
    print(f"[intersect] {name:30s} n={len(y):6d} AUROC={a:.4f} [{a_lo:.4f},{a_hi:.4f}] "
          f"TPR@1%={tpr1:.4f} TPR@5%={tpr5:.4f}", flush=True)
    return {"n": int(len(y)), "auroc": float(a), "auroc_ci": [a_lo, a_hi],
            "tpr_1pct": tpr1, "tpr_5pct": tpr5}

print()
r85 = report("all (85-feat, persisted model)", y, p85, t1_85, t5_85)
r31 = report("all_minus_fractal (31-feat, same rows)", y, p31, t1_31, t5_31)

# paired significance: DeLong or bootstrap on the AUROC difference, same population/order
diffs = []
rng = np.random.RandomState(20260830)
n = len(y)
pos_idx = np.where(y == 1)[0]
neg_idx = np.where(y == 0)[0]
for _ in range(2000):
    bi = np.concatenate([rng.choice(pos_idx, len(pos_idx), replace=True),
                          rng.choice(neg_idx, len(neg_idx), replace=True)])
    try:
        d = roc_auc_score(y[bi], p85[bi]) - roc_auc_score(y[bi], p31[bi])
        diffs.append(d)
    except ValueError:
        continue
diffs = np.array(diffs)
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"\n[intersect] paired bootstrap AUROC(85-feat) - AUROC(31-feat): "
      f"mean={diffs.mean():.4f}  95% CI=[{lo:.4f},{hi:.4f}]  "
      f"excludes 0: {lo > 0 or hi < 0}", flush=True)

json.dump({
    "population": "complete-case on 85-feature schema (Task4's exact rule), identical rows for both models",
    "all_85feat": r85,
    "all_minus_fractal_31feat_same_rows": r31,
    "paired_bootstrap_auroc_diff_85_minus_31": {"mean": float(diffs.mean()), "ci95": [float(lo), float(hi)]},
}, open(D("fractalscope_check_raid_intersection.json"), "w"), indent=2)
print("\n[intersect] DONE -- written to data/fractalscope_check_raid_intersection.json")
