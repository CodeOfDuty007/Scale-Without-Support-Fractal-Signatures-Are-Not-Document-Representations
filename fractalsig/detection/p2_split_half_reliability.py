#!/usr/bin/env python3
"""
P2 -- split-half reliability of the fractal features, and the feature whitelist.

METHOD
  Each document's token-score sequence is split into two halves by INTERLEAVED
  odd/even token positions -- not first-half/second-half. A contiguous split
  would conflate "this feature is unreliable" with "documents differ
  stylistically between their opening and their close", which is a different
  claim entirely. Interleaving makes the two halves exchangeable draws from the
  same document.

  Each candidate feature is computed independently on each half, then correlated
  across the corpus, then Spearman-Brown corrected to estimate full-length
  reliability:  r_full = 2*r_half / (1 + r_half).

  Whitelist threshold: rho >= 0.60 (plan documents' pre-registered bar).

SCALE
  Halves are 200 tokens (documents are clipped to 400). The full sweep reaches
  tau=320, which is meaningless on a 200-point series, so the halves use the
  project's own WINDOW_SCALE = [8,16,32,48,64,96] -- already validated for
  length-200 series in the P1 CPU pass. This is a deliberate reuse, not a new
  choice.

WINDOWED FEATURES
  WINDOW_SIZE=200 with a 200-token half yields exactly ONE window, so drift is
  undefined and var is degenerate. Windowed features are therefore evaluated at
  a reduced window (100/50) so they have >=3 windows. Their reliability is thus
  measured for the STABILITY-PROFILE CONCEPT, not for the exact production
  column -- stated explicitly in the output, because the plan's pre-registered
  fallback is "stability-profile features instead of a raw point estimate" and
  we need to know whether that fallback is itself reliable.

BIAS AND RELIABILITY ARE DIFFERENT PROPERTIES
  Per-doc S is known biased at ~400 tokens (Holder_PerDoc_Unreliable). A
  feature can be consistently biased yet perfectly reliable (halves agree; the
  bias is recalibratable), or unbiased on average yet unreliable (halves
  disagree; no stable signal). Both are reported per feature, never conflated.
"""
from fractalsig import paths as _paths  # noqa: E402
import collections
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy import stats

HERE = _paths.WORKSPACE_STR
from fractalsig.features.fractal_math import generate_trajectory, calc_H, calc_S  # noqa: E402
from fractalsig.pipeline.build_p1_feature_store_cpu import (  # noqa: E402
    make_windows, window_summary, WINDOW_SCALE)

SHARDS = os.environ.get("GAGLE_MANIFEST_DIR", _paths.MANIFEST_SHARD_DIR_STR)
FS = os.path.join(HERE, "feature_store", "v1_gagle.parquet")
OUT_WL = os.path.join(HERE, "feature_store", "fractal_feature_whitelist.json")
OUT_REP = os.path.join(HERE, "feature_store", "p2_reliability_report.json")

RHO_MIN = 0.60
HALF_WIN, HALF_STRIDE = 100, 50
SCORERS = [("gemma", "gagle_gemma_sliced_scores"), ("mistral", "gagle_mistral_sliced_scores")]


def half_features(arr):
    """Features for ONE half (a 200-point score sequence)."""
    traj, incr = generate_trajectory(arr)
    S, S_r2 = calc_S(traj, scale_param=WINDOW_SCALE)
    H, H_r2 = calc_H(incr, scale_param=WINDOW_SCALE)
    out = {"S": float(S), "H": float(H), "S_R2": float(S_r2), "H_R2": float(H_r2)}
    for name, series in (("S", traj), ("H", incr)):
        vals = []
        for w in make_windows(series, window_size=HALF_WIN, stride=HALF_STRIDE):
            if name == "S":
                v, _ = calc_S(w, scale_param=WINDOW_SCALE[:4])
            else:
                v, _ = calc_H(w, scale_param=WINDOW_SCALE[:4])
            vals.append(float(v))
        ws = window_summary(vals)
        for k in ("mean", "var", "drift"):
            out[f"{name}_window_{k}"] = float(ws[k])
    return out


def main():
    files = sorted(glob.glob(os.path.join(SHARDS, "*.parquet")))
    print(f"[info] {len(files)} shards; halves use scale {WINDOW_SCALE}, "
          f"windows {HALF_WIN}/{HALF_STRIDE}", flush=True)
    acc = collections.defaultdict(lambda: {"a": [], "b": []})
    n_docs = 0
    for n, f in enumerate(files, 1):
        cols = ["id"] + [c for _, c in SCORERS]
        t = pq.read_table(f, columns=[c for c in cols])
        for r in t.to_pylist():
            for sname, scol in SCORERS:
                arr = r.get(scol)
                if not arr or len(arr) < 8:
                    continue
                a = np.asarray(arr[0::2], dtype=np.float64)   # odd positions
                b = np.asarray(arr[1::2], dtype=np.float64)   # even positions
                if len(a) < 8 or len(b) < 8:
                    continue
                try:
                    fa, fb = half_features(a), half_features(b)
                except Exception:
                    continue
                for k in fa:
                    if fa[k] == fa[k] and fb[k] == fb[k]:
                        acc[f"{sname}_{k}"]["a"].append(fa[k])
                        acc[f"{sname}_{k}"]["b"].append(fb[k])
            n_docs += 1
        if n % 100 == 0:
            print(f"[info]   {n}/{len(files)} shards, {n_docs:,} docs", flush=True)

    print(f"\n[info] {n_docs:,} documents processed\n", flush=True)
    rows = []
    for feat, d in sorted(acc.items()):
        a, b = np.asarray(d["a"]), np.asarray(d["b"])
        if len(a) < 100 or a.std() == 0 or b.std() == 0:
            continue
        r_half = float(np.corrcoef(a, b)[0, 1])
        rho_sp = float(stats.spearmanr(a, b).statistic)
        sb = 2 * r_half / (1 + r_half) if r_half > -1 else float("nan")
        rows.append({"feature": feat, "n": int(len(a)), "r_half": r_half,
                     "spearman_half": rho_sp, "spearman_brown": float(sb),
                     "mean_half": float(np.mean(np.concatenate([a, b]))),
                     "passes": bool(sb >= RHO_MIN)})

    print("=" * 96)
    print(f"P2 SPLIT-HALF RELIABILITY  (interleaved odd/even; Spearman-Brown; "
          f"threshold rho >= {RHO_MIN})")
    print("=" * 96)
    print(f"{'feature':<28}{'n':>9}{'r half':>9}{'rho SB':>9}{'spearman':>10}"
          f"{'mean':>10}  verdict")
    print("-" * 96)
    for r in sorted(rows, key=lambda x: -x["spearman_brown"]):
        print(f"{r['feature']:<28}{r['n']:>9,}{r['r_half']:>9.3f}"
              f"{r['spearman_brown']:>9.3f}{r['spearman_half']:>10.3f}"
              f"{r['mean_half']:>10.3f}  {'KEEP' if r['passes'] else 'drop'}")

    keep = [r["feature"] for r in rows if r["passes"]]
    drop = [r["feature"] for r in rows if not r["passes"]]
    print("-" * 96)
    print(f"  KEEP: {len(keep)}   DROP: {len(drop)}")
    print("\n  NOTE -- bias and reliability are different properties. Per-doc S is")
    print("  known BIASED at these lengths (Holder_PerDoc_Unreliable); that is not")
    print("  what this table measures. A feature can be biased and still reliable")
    print("  (halves agree, bias recalibratable) or unbiased and unreliable.")
    print("  NOTE -- windowed features use a 100/50 window on 200-token halves, so")
    print("  they measure the STABILITY-PROFILE CONCEPT, not the production column")
    print("  (which uses 200/100 on 400 tokens).")

    fallback = [r["feature"] for r in rows
                if r["passes"] and "_window_" in r["feature"]]
    json.dump({"threshold": RHO_MIN, "method": "interleaved odd/even split-half, "
               "Spearman-Brown corrected", "n_documents": n_docs,
               "whitelist": keep, "rejected": drop,
               "stability_profile_fallback_available": fallback,
               "features": rows}, open(OUT_WL, "w"), indent=2)
    json.dump({"features": rows}, open(OUT_REP, "w"), indent=2)
    print(f"\n[written] {OUT_WL}")
    print(f"[written] {OUT_REP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
