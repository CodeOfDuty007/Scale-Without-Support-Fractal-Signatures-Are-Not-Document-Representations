#!/usr/bin/env python3
"""Direct test of the FractalScope outline's central hypothesis: does
removing fractal features (all_minus_fractal, 31 feat) cost robustness
under (a) P4's real adversarial attacks and (b) GAGLE->RAID cross-corpus
transfer, relative to the full 85-feature model? Uses ONLY already-computed
feature stores (v2_edited.parquet, v3_raid.parquet) -- no new generation or
GPU scoring. Trains one new model (all_minus_fractal, config=full,
random/random) via the frozen fit_predict/D-11 pipeline, exactly matching
p3_refit_and_persist.py's method, just for the one arm it never covered.
"""
from fractalsig import paths as _paths  # noqa: E402
import json, os, sys
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve

HERE = _paths.WORKSPACE_STR
from fractalsig.detection.p3_train_eval import load_matrix, build_folds, config_mask, arm_features, fit_predict
D = lambda *p: os.path.join(HERE, "data", *p)
FS = lambda *p: os.path.join(HERE, "feature_store", *p)

print("[test] loading GAGLE matrix + training all_minus_fractal (config=full, random/random)...", flush=True)
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

names = arm_features("all_minus_fractal", groups)
print(f"[test] all_minus_fractal: {len(names)} features", flush=True)
Xa = df[names].to_numpy(float)
Xtr, Xva, Xte = Xa[tr2], Xa[va2], Xa[te2]
ytr, yva, yte = yall[tr2], yall[va2], yall[te2]

p_te, base = fit_predict("lgbm", Xtr, ytr, Xva, yva, Xte, names)
cal = CalibratedClassifierCV(estimator=base, cv="prefit", method="sigmoid")
cal.fit(Xva, yva)

# raw-probability thresholds from the val fold, same convention as
# p4_task5_tabular_features.py / p5_task4_gagle_to_raid_transfer.py
p_va_raw = base.predict_proba(Xva)[:, int(np.where(base.classes_ == 1)[0][0])]
fpr, tpr, thresh = roc_curve(yva, p_va_raw)
idx1 = np.where(fpr <= 0.01)[0]
idx5 = np.where(fpr <= 0.05)[0]
t1 = thresh[idx1[-1]] if len(idx1) else None
t5 = thresh[idx5[-1]] if len(idx5) else None
print(f"[test] val-derived thresholds: t1={t1:.4f} t5={t5:.4f}", flush=True)

from sklearn.metrics import roc_auc_score
auc_clean = roc_auc_score(yte, p_te)
print(f"[test] sanity: full/random/random all_minus_fractal test AUROC={auc_clean:.4f} "
      f"(expect ~0.9984, matching the ledgered grid cell)", flush=True)

def score_raw(X):
    return base.predict_proba(X)[:, int(np.where(base.classes_ == 1)[0][0])]

# ---------------------------------------------------------------- P4 battery
print("\n[test] === P4 robustness battery, all_minus_fractal vs tabular(85-feat) ===", flush=True)
v2 = pd.read_parquet(FS("v2_edited.parquet"))
Xp4 = v2[names].to_numpy(float)
p4_raw = score_raw(Xp4)
v2 = v2.assign(_p=p4_raw)

by_type = {}
for variant, sub in v2.groupby("variant"):
    n = len(sub)
    r1 = float((sub["_p"] >= t1).mean())
    r5 = float((sub["_p"] >= t5).mean())
    by_type[variant] = {"n": n, "recall_1pct": r1, "recall_5pct": r5}
    print(f"  {variant:24s} n={n:6d}  recall@1%={r1:.4f}  recall@5%={r5:.4f}", flush=True)

# pre-attack recall: reuse full/random/random test fold at the SAME val-derived thresholds
p_te_raw = score_raw(Xte)
pre1 = float(((p_te_raw >= t1) & (yte == 1)).sum() / (yte == 1).sum())
pre5 = float(((p_te_raw >= t5) & (yte == 1)).sum() / (yte == 1).sum())
print(f"  {'PRE-ATTACK (clean test fold)':24s} n={int((yte==1).sum()):6d}  recall@1%={pre1:.4f}  recall@5%={pre5:.4f}", flush=True)

json.dump({"arm": "all_minus_fractal", "n_features": len(names),
           "sanity_auroc_full_random_random": float(auc_clean),
           "pre_attack": {"recall_1pct": pre1, "recall_5pct": pre5},
           "by_type": by_type},
          open(D("fractalscope_check_p4_all_minus_fractal.json"), "w"), indent=2)

# ---------------------------------------------------------------- RAID transfer
print("\n[test] === GAGLE->RAID transfer, all_minus_fractal ===", flush=True)
v3 = pd.read_parquet(FS("v3_raid.parquet"))
missing = [c for c in names if c not in v3.columns]
if missing:
    print(f"[test] WARNING: {len(missing)} all_minus_fractal features missing from v3_raid.parquet: {missing}", flush=True)
present = [c for c in names if c in v3.columns]
sub = v3.dropna(subset=present)
print(f"[test] RAID complete-case on {len(present)}/{len(names)} features: {len(sub)}/{len(v3)} rows", flush=True)
Xraid = sub[present].to_numpy(float)
p_raid = score_raw(Xraid)
y_raid = (sub["Label"] != "human").astype(int).to_numpy()

raid_auc = roc_auc_score(y_raid, p_raid)
# re-derive threshold on RAID's OWN human population, same convention as task4
fpr_r, tpr_r, thresh_r = roc_curve(y_raid, p_raid)
idx1r = np.where(fpr_r <= 0.01)[0]; idx5r = np.where(fpr_r <= 0.05)[0]
rt1 = thresh_r[idx1r[-1]] if len(idx1r) else None
rt5 = thresh_r[idx5r[-1]] if len(idx5r) else None
r_tpr1 = float(tpr_r[idx1r[-1]]) if len(idx1r) else None
r_tpr5 = float(tpr_r[idx5r[-1]]) if len(idx5r) else None
print(f"[test] RAID-native-threshold: AUROC={raid_auc:.4f} TPR@1%={r_tpr1:.4f} TPR@5%={r_tpr5:.4f} n={len(sub)}", flush=True)

json.dump({"arm": "all_minus_fractal", "n_features_used": len(present),
           "n_features_expected": len(names), "missing_features": missing,
           "n_test": len(sub), "auroc": float(raid_auc),
           "raid_native_tpr1": r_tpr1, "raid_native_tpr5": r_tpr5},
          open(D("fractalscope_check_raid_all_minus_fractal.json"), "w"), indent=2)
print("\n[test] DONE", flush=True)
