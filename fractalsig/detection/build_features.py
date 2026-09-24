#!/usr/bin/env python3
"""
P3 feature builder (D-13) -- full corpus, complete declared feature set.

INPUT SOURCE, determined not assumed
  The residual features need the INDIVIDUAL per-scale fit points, not the pooled
  point estimate. Checked all three candidates before choosing:
    - feature_store/v1_gagle.parquet : 0 of 88 columns carry per-scale data
    - data/p1_feature_store_cpu.jsonl: NO array-valued fields at all -- the P1
      spec's explicit disk-budget guardrail keeps only aggregated scalars
    - gpu_manifest_shards/*.parquet  : DOES carry gagle_{gemma,mistral}_sliced_
      scores, 400 values per document
  So the shards are the only viable source. They were written BY the P1 CPU
  pass, so they are P1's population by construction rather than a re-selection
  -- which matters, because re-deriving the population is exactly what made
  reconcile_p1_against_p0.py v1 unusable.

ESTIMATOR
  Everything comes from the F-4-fixed fractalsig/features/fractal_math.py calc_S / calc_H, via
  their return_fit= parameter. The per-scale points are NOT recomputed here.
  This project has had three separate defects caused by a second copy of shared
  logic drifting from the original (D-07's dedup key, the v1 cap generator,
  reconcile v1's human key); a fourth copy of the fit would be the same mistake.

  On a ~400-token document calc_S's sampling branch never fires at any scale,
  so per-document S is exact with no RNG involvement whatsoever.

D-13: the feature set is DECLARED IN ADVANCE (see FEATURE_MANIFEST) and every
feature is computed and reported, including ones expected to perform poorly.
P2 found 0/20 fractal features reach rho>=0.60 reliability; the residuals are
computed anyway because a direct empirical result beats an inference from a
proxy test, and because having the columns preserves the ability to choose
subsets later. Selective reporting after the fact would read as p-hacking.
"""
from fractalsig import paths as _paths  # noqa: E402
import argparse
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

HERE = _paths.WORKSPACE_STR
from fractalsig.features.fractal_math import generate_trajectory, calc_S, calc_H  # noqa: E402
from fractalsig.pipeline.build_p1_feature_store_cpu import (  # noqa: E402
    make_windows, window_summary, WINDOW_SCALE)

DATA = os.path.join(HERE, "data")
SHARDS = os.environ.get("GAGLE_MANIFEST_DIR", _paths.MANIFEST_SHARD_DIR_STR)
OUT = os.path.join(DATA, "p3_features.jsonl")
CKPT = os.path.join(DATA, "p3_features_checkpoint.txt")
MANIFEST = os.path.join(HERE, "feature_store", "p3_feature_manifest.json")

RESID_SCALES = [8, 16, 32, 48, 64]
SCORERS = [("Gemma", "gagle_gemma_sliced_scores"),
           ("Mistral", "gagle_mistral_sliced_scores")]
META = ["id", "split", "domain", "Label", "prompt_type", "temp"]
# Deliberately 1.96, NOT the exact 1.959963984540054. build_p1_feature_store_cpu
# uses a flat 1.96 for its CI half-widths, and a smoke test caught the more
# precise constant producing a 2.1e-05 divergence from the stored columns.
# Matching the existing artifact matters more than a fourth-decimal improvement;
# two CI conventions coexisting in one project is how drift starts.
Z95 = 1.96


def declared_features():
    """The complete D-13 feature set, declared before the run."""
    feats = {}
    for p in ("Gemma", "Mistral"):
        feats[f"{p}_S"] = "Holder exponent, point estimate (full scale sweep)"
        feats[f"{p}_H"] = "Hurst exponent, point estimate (full scale sweep)"
        feats[f"{p}_S_R2"] = "log-log fit quality for S"
        feats[f"{p}_H_R2"] = "log-log fit quality for H"
        feats[f"{p}_S_CI_halfwidth"] = "1.96 * linregress std_err on S"
        feats[f"{p}_H_CI_halfwidth"] = "1.96 * linregress std_err on H"
        for t in RESID_SCALES:
            feats[f"{p}_s_res_{t}"] = f"S log-log residual at scale tau={t}"
            feats[f"{p}_h_res_{t}"] = f"H log-log residual at scale tau={t}"
        for e in ("S", "H"):
            for k in ("mean", "var", "min", "max", "drift"):
                feats[f"{p}_{e}_window_{k}"] = (
                    f"{e} {k} across intra-document windows "
                    f"(WINDOW_SCALE, project-validated)")
    feats["Scorer_Divergence_S"] = "|S_gemma - S_mistral|"
    feats["Scorer_Divergence_H"] = "|H_gemma - H_mistral|"
    return feats


def resid_at(fit, scales, wanted):
    """Residual at each requested scale; NaN where that scale dropped out of
    the fit (p0 <= 0 there). Never silently substituted with a neighbour."""
    lookup = {int(round(s)): r for s, r in zip(scales, fit["resid"])}
    return {t: float(lookup.get(t, float("nan"))) for t in wanted}


def features_for(arr, prefix):
    out = {}
    traj, incr = generate_trajectory(arr)
    S, S_r2, S_se, S_fit = calc_S(traj, return_std_err=True, return_fit=True)
    H, H_r2, H_se, H_fit = calc_H(incr, return_std_err=True, return_fit=True)
    out[f"{prefix}_S"] = float(S)
    out[f"{prefix}_H"] = float(H)
    out[f"{prefix}_S_R2"] = float(S_r2)
    out[f"{prefix}_H_R2"] = float(H_r2)
    out[f"{prefix}_S_CI_halfwidth"] = float(Z95 * S_se) if S_se == S_se else float("nan")
    out[f"{prefix}_H_CI_halfwidth"] = float(Z95 * H_se) if H_se == H_se else float("nan")

    s_res = resid_at(S_fit, S_fit.get("scales", []), RESID_SCALES)
    h_scales = [int(round(float(np.exp(x)))) for x in H_fit.get("log_scale", [])]
    h_res = resid_at(H_fit, h_scales, RESID_SCALES)
    for t in RESID_SCALES:
        out[f"{prefix}_s_res_{t}"] = s_res[t]
        out[f"{prefix}_h_res_{t}"] = h_res[t]

    for e, series in (("S", traj), ("H", incr)):
        vals = []
        for w in make_windows(series):
            v = (calc_S(w, scale_param=WINDOW_SCALE)[0] if e == "S"
                 else calc_H(w, scale_param=WINDOW_SCALE)[0])
            vals.append(float(v))
        ws = window_summary(vals)
        for k in ("mean", "var", "min", "max", "drift"):
            out[f"{prefix}_{e}_window_{k}"] = float(ws[k])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-shards", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    feats = declared_features()
    json.dump({"declared_in_advance": True, "decision": "D-13",
               "n_features": len(feats), "resid_scales": RESID_SCALES,
               "window_scale": list(WINDOW_SCALE),
               "estimator": "fractalsig/features/fractal_math.py calc_S/calc_H (F-4 fixed, "
                            "enumeration-based, deterministic)",
               "input_source": "gpu_manifest_shards/*.parquet raw sliced arrays",
               "features": feats}, open(MANIFEST, "w"), indent=2)
    print(f"[manifest] {len(feats)} features declared -> {MANIFEST}", flush=True)

    done = set()
    if os.path.exists(CKPT):
        done = {l.strip() for l in open(CKPT) if l.strip()}
    files = sorted(glob.glob(os.path.join(SHARDS, "*.parquet")))
    todo = [f for f in files if f not in done]
    if args.limit_shards:
        todo = todo[:args.limit_shards]
    print(f"[info] {len(files)} shards, {len(done)} already done, {len(todo)} to do",
          flush=True)

    n_rows = 0
    with open(OUT, "a") as fout:
        for i, f in enumerate(todo, 1):
            cols = ["id", "split", "domain", "Label", "prompt_type", "temp"] + \
                   [c for _, c in SCORERS]
            avail = set(pq.read_schema(f).names)
            t = pq.read_table(f, columns=[c for c in cols if c in avail])
            for r in t.to_pylist():
                row = {k: r.get(k) for k in META}
                for prefix, col in SCORERS:
                    arr = r.get(col)
                    if arr and len(arr) >= max(WINDOW_SCALE):
                        try:
                            row.update(features_for(arr, prefix))
                        except Exception as e:
                            row[f"{prefix}_ERROR"] = str(e)[:120]
                for e in ("S", "H"):
                    g, m = row.get(f"Gemma_{e}"), row.get(f"Mistral_{e}")
                    row[f"Scorer_Divergence_{e}"] = (
                        float(abs(g - m)) if isinstance(g, float) and isinstance(m, float)
                        and g == g and m == m else float("nan"))
                fout.write(json.dumps(row) + "\n")
                n_rows += 1
            fout.flush()
            os.fsync(fout.fileno())
            with open(CKPT, "a") as ck:
                ck.write(f + "\n")
            if i % 50 == 0:
                print(f"[info]   {i}/{len(todo)} shards, {n_rows:,} rows", flush=True)

    print(f"\n[done] {n_rows:,} rows -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
