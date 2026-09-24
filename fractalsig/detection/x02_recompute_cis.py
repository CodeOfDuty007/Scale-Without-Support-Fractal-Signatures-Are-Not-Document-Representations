#!/usr/bin/env python3
"""
X-02: recompute S/H confidence intervals and the per-domain human
normalisation constants, post-X-01 and post-F-4.

SCOPE (design decision, 2026-08-23): ALL S-based CIs, human AND LLM.
  X-02 was originally scoped human-only, on the premise that human N was
  inflated 2.34x (11,696 -> 5,000) so human CIs were ~1.53x too narrow. F-4
  changed the premise: EVERY S value in the corpus was re-derived, so the LLM
  side's S CIs are equally stale. Recomputing only the human side would leave
  human and LLM CIs on two different vintages and make any cross-population
  comparison incoherent.

WHAT IS COMPUTED
  1. Per-doc mean S/H per (domain, Label) per scorer, with a percentile
     bootstrap CI. This is the quantity src/visualizer.py actually consumes:
     it normalises by `df[df.Type=='Human'].groupby('domain')[['S','H']].mean()`,
     so these means ARE the log-ratio denominators.
  2. The same quantity on the PRE-X-01 store, so the CI-width change is
     measured rather than asserted. The ledger's "~1.53x too narrow" is a
     sqrt(11696/5000) prediction; this checks it against the data.
  3. Pooled corpus-level S/H per cell with a 10-fold outer bootstrap over
     documents, matching the internal verification notes's
     methodology. Pooled S is the only valid corpus-level S estimator (per-doc
     S is biased -- Holder_PerDoc_Unreliable), so its CI needs the expensive
     re-pooling, not a resample of scalars.

NaN POLICY (design decision): per-column masking, no row dropping. Each cell's
n is therefore the count of NON-NaN values for THAT column, reported per row.
"""
from fractalsig import paths as _paths  # noqa: E402
import collections
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

HERE = _paths.WORKSPACE_STR
from fractalsig.features.fractal_math import generate_trajectory, calc_H, calc_S  # noqa: E402

DATA = os.path.join(HERE, "data")
NEW_FS = os.path.join(DATA, "p1_feature_store_cpu.jsonl")
OLD_FS = os.path.join(DATA, "pre_x01_2026-08-23", "p1_feature_store_cpu.jsonl")
SHARDS = os.environ.get("GAGLE_MANIFEST_DIR", _paths.MANIFEST_SHARD_DIR_STR)
OUT = os.path.join(DATA, "x02_confidence_intervals.json")

B_MEAN = 2000     # bootstrap reps for the cheap per-doc-mean CIs
B_POOLED = 10     # outer bootstrap folds for pooled S/H (Test-4's methodology)
SEED = 20260823


def load_percell(path, cols):
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    ids = collections.defaultdict(list)
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            k = (d["domain"], d["Label"])
            ids[k].append(d["id"])
            for c in cols:
                v = d.get(c)
                out[k][c].append(float(v) if isinstance(v, (int, float)) and v == v
                                 else np.nan)
    return out, ids


def boot_mean_ci(vals, rng, B=B_MEAN):
    v = np.asarray([x for x in vals if x == x], dtype=float)
    if len(v) < 2:
        return None
    idx = rng.integers(0, len(v), size=(B, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"n": int(len(v)), "mean": float(v.mean()),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "ci_width": float(hi - lo)}


def main():
    rng = np.random.default_rng(SEED)
    cols = ["Gemma_S", "Gemma_H", "Mistral_S", "Mistral_H"]
    new, new_ids = load_percell(NEW_FS, cols)
    print(f"[info] new store: {len(new)} cells")

    results = {"per_doc_mean_ci": {}, "normalisation_constants": {},
               "pooled_ci": {}, "old_vs_new_human": {}}

    print("\n" + "=" * 96)
    print("1. PER-DOC MEAN S/H WITH BOOTSTRAP CI  (the quantity visualizer.py normalises by)")
    print("=" * 96)
    print(f"{'cell':<30}{'col':<12}{'n':>8}{'mean':>9}{'ci_lo':>9}{'ci_hi':>9}{'width':>9}")
    print("-" * 96)
    for k in sorted(new):
        for c in cols:
            r = boot_mean_ci(new[k][c], rng)
            if r is None:
                continue
            results["per_doc_mean_ci"][f"{k[0]}|{k[1]}|{c}"] = r
            print(f"{k[0]+'|'+k[1]:<30}{c:<12}{r['n']:>8,}{r['mean']:>9.4f}"
                  f"{r['ci_lo']:>9.4f}{r['ci_hi']:>9.4f}{r['ci_width']:>9.4f}")

    # normalisation constants = per-domain HUMAN per-doc means
    print("\n" + "=" * 96)
    print("2. PER-DOMAIN HUMAN NORMALISATION CONSTANTS (log-ratio denominators)")
    print("=" * 96)
    for k in sorted(new):
        if k[1] != "Human":
            continue
        ent = {}
        for c in cols:
            v = np.asarray([x for x in new[k][c] if x == x], dtype=float)
            if len(v):
                ent[c] = float(v.mean())
        results["normalisation_constants"][k[0]] = ent
        print(f"  {k[0]:<22} " + "  ".join(f"{c}={ent[c]:.4f}" for c in sorted(ent)))

    # old vs new human CI widths -- verify the 1.53x prediction
    if os.path.exists(OLD_FS):
        old, _ = load_percell(OLD_FS, cols)
        print("\n" + "=" * 96)
        print("3. HUMAN CI WIDTH: PRE-X-01 vs POST-X-01   (ledger predicts ~1.53x widening)")
        print("=" * 96)
        print(f"{'domain':<22}{'col':<12}{'n old':>8}{'n new':>8}"
              f"{'width old':>11}{'width new':>11}{'ratio':>8}{'sqrt(N/n)':>11}")
        print("-" * 96)
        for k in sorted(new):
            if k[1] != "Human":
                continue
            for c in cols:
                a = boot_mean_ci(old.get(k, {}).get(c, []), rng)
                b = boot_mean_ci(new[k][c], rng)
                if not a or not b or a["ci_width"] == 0:
                    continue
                ratio = b["ci_width"] / a["ci_width"]
                pred = (a["n"] / b["n"]) ** 0.5
                results["old_vs_new_human"][f"{k[0]}|{c}"] = {
                    "n_old": a["n"], "n_new": b["n"],
                    "width_old": a["ci_width"], "width_new": b["ci_width"],
                    "ratio": ratio, "sqrt_n_ratio": pred}
                print(f"{k[0]:<22}{c:<12}{a['n']:>8,}{b['n']:>8,}"
                      f"{a['ci_width']:>11.5f}{b['ci_width']:>11.5f}"
                      f"{ratio:>8.2f}{pred:>11.2f}")

    # pooled corpus-level CIs (expensive)
    print("\n" + "=" * 96)
    print(f"4. POOLED CORPUS-LEVEL S/H, {B_POOLED}-FOLD OUTER BOOTSTRAP (gemma-2b)")
    print("=" * 96, flush=True)
    flat = {i for v in new_ids.values() for i in v}
    arrays = {}
    for f in sorted(glob.glob(os.path.join(SHARDS, "*.parquet"))):
        t = pq.read_table(f, columns=["id", "gagle_gemma_sliced_scores"])
        for r in t.to_pylist():
            if r["id"] in flat and r["gagle_gemma_sliced_scores"]:
                arrays[r["id"]] = r["gagle_gemma_sliced_scores"]
    print(f"[info] {len(arrays):,} gemma arrays loaded", flush=True)
    print(f"{'cell':<30}{'n':>8}{'S mean':>9}{'S lo':>9}{'S hi':>9}"
          f"{'H mean':>9}{'H lo':>9}{'H hi':>9}")
    print("-" * 96)
    for k in sorted(new_ids):
        got = [np.asarray(arrays[i], dtype=np.float64) for i in new_ids[k] if i in arrays]
        if len(got) < 10:
            continue
        Ss, Hs = [], []
        for _ in range(B_POOLED):
            idx = rng.integers(0, len(got), size=len(got))
            samp = [got[j] for j in idx]
            trajs, incrs = zip(*(generate_trajectory(a) for a in samp))
            Ss.append(float(calc_S(list(trajs))[0]))
            Hs.append(float(calc_H(np.concatenate(incrs))[0]))
        Ss, Hs = np.array(Ss), np.array(Hs)
        ent = {"n": len(got),
               "S_mean": float(Ss.mean()), "S_lo": float(np.percentile(Ss, 2.5)),
               "S_hi": float(np.percentile(Ss, 97.5)),
               "H_mean": float(Hs.mean()), "H_lo": float(np.percentile(Hs, 2.5)),
               "H_hi": float(np.percentile(Hs, 97.5))}
        results["pooled_ci"][f"{k[0]}|{k[1]}"] = ent
        print(f"{k[0]+'|'+k[1]:<30}{ent['n']:>8,}{ent['S_mean']:>9.4f}"
              f"{ent['S_lo']:>9.4f}{ent['S_hi']:>9.4f}{ent['H_mean']:>9.4f}"
              f"{ent['H_lo']:>9.4f}{ent['H_hi']:>9.4f}", flush=True)

    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\n[written] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
