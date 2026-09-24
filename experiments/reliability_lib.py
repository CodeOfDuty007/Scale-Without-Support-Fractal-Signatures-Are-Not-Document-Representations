#!/usr/bin/env python3
"""
Shared library for T0 (corrected coefficient + variance decomposition) and
T1 (reliability under six split constructions) of the reliability analysis.

Reuses the project's own estimators unmodified (fractalsig/features/fractal_math.py,
fractalsig/pipeline/build_p1_feature_store_cpu.py's make_windows/window_summary/WINDOW_SCALE)
rather than re-implementing them -- this is a deliberate design choice so
that any T0/T1 result is directly comparable to the frozen P2 baseline in
feature_store/p2_reliability_report.json, computed by the same math.

This module is written fresh under experiments/ rather than editing
the P2 reliability-gate script in place: keep the
original pipeline scripts untouched and put every new artifact under
outputs/ for easy retrieval.

STRATA NOTE: the plan's demeaning strata are written as
(class, generator, domain, temp). In GAGLE, `Label` (Human / gemma_2b /
mistral_7b) already IS the class-generator join -- there is no separate
"generator" column, and Label fully determines class (Human is its own
class). Strata used throughout this module are therefore (Label, domain,
temp), which is the exact information content of (class, generator, domain,
temp) for this corpus. prompt_type is excluded from the demeaning set,
per the plan's explicit instruction (Section 4.2 treats prompt_type as
part of delta, not population structure).
"""
import glob
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from scipy import stats

from fractalsig import paths as _paths  # noqa: E402
from fractalsig.features.fractal_math import generate_trajectory, calc_H, calc_S  # noqa: E402
from fractalsig.pipeline.build_p1_feature_store_cpu import (  # noqa: E402
    make_windows, window_summary, WINDOW_SCALE)

SHARD_DIR = _paths.MANIFEST_SHARD_DIR_STR
SCORERS = [("gemma", "gagle_gemma_sliced_scores"), ("mistral", "gagle_mistral_sliced_scores")]
STRATA_COLS = ["Label", "domain", "temp"]  # see STRATA NOTE above
RHO_MIN = 0.60


# --------------------------------------------------------------------------
# Half-feature computation (byte-for-byte reuse of the frozen P2 method)
# --------------------------------------------------------------------------
def half_features(arr, window_scale=None, skip_windowed=False):
    """Features for ONE half (a ~200-point score sequence). Identical to
    p2_split_half_reliability.py's half_features(), with two additive
    options used by T1: `window_scale` overrides the scale swept for the
    windowed sub-features (T1's matched_sweep construction), and
    `skip_windowed` drops the windowed sub-features entirely (used for the
    permutation construction, where 500 reps over 223k documents makes the
    windowed loop the dominant cost and the plan's headline test only needs
    S/H/S_R2/H_R2)."""
    ws = window_scale if window_scale is not None else WINDOW_SCALE[:4]
    traj, incr = generate_trajectory(arr)
    S, S_r2 = calc_S(traj, scale_param=WINDOW_SCALE)
    H, H_r2 = calc_H(incr, scale_param=WINDOW_SCALE)
    out = {"S": float(S), "H": float(H), "S_R2": float(S_r2), "H_R2": float(H_r2)}
    if skip_windowed:
        return out
    for name, series in (("S", traj), ("H", incr)):
        vals = []
        for w in make_windows(series, window_size=100, stride=50):
            if name == "S":
                v, _ = calc_S(w, scale_param=ws)
            else:
                v, _ = calc_H(w, scale_param=ws)
            vals.append(float(v))
        summ = window_summary(vals)
        for k in ("mean", "var", "drift"):
            out[f"{name}_window_{k}"] = float(summ[k])
    return out


# --------------------------------------------------------------------------
# Split constructions
# --------------------------------------------------------------------------
def split_interleave(arr, rng=None):
    return arr[0::2], arr[1::2]


def split_contiguous(arr, rng=None):
    n = len(arr)
    m = n // 2
    return arr[:m], arr[m:]


def make_split_block(b):
    def _split(arr, rng=None):
        a_idx, b_idx = [], []
        for start in range(0, len(arr), b):
            block = list(range(start, min(start + b, len(arr))))
            if (start // b) % 2 == 0:
                a_idx.extend(block)
            else:
                b_idx.extend(block)
        return arr[a_idx], arr[b_idx]
    _split.__name__ = f"block_{b}"
    return _split


def split_permutation(arr, rng):
    """Pronk-style random balanced split: randomize HALF MEMBERSHIP only,
    preserve each half's own internal chronological order (interleave is the
    deterministic special case of this same idea -- alternating membership,
    order preserved). v2: the first version
    returned arr[idx[:m]] with idx[:m] UNSORTED, which scrambles each half's
    temporal order along with its membership -- R/S and displacement-
    probability scaling both operate on the ORDERED trajectory, so a
    scrambled half is white noise to the estimator regardless of the
    document's real structure. Measured directly (not assumed): the v1 bug
    produced rho_pooled~0.003 (CI crossing zero) for gemma_H/S and mistral_H/S
    against interleave's 0.576-0.626 on the identical corpus -- and tellingly,
    H_R2 (log-log fit quality) stayed non-zero and CI-excluding-zero even
    under the scrambled input, since fit quality partly survives order
    destruction in a way the slope-derived H estimate does not. That
    dissociation is what pinned this down as an order-destruction artifact,
    not a genuine "the quantity isn't there under random splitting" result."""
    n = len(arr)
    idx = rng.permutation(n)
    m = n // 2
    return arr[np.sort(idx[:m])], arr[np.sort(idx[m:])]


def make_split_block_bootstrap(block_len=100):
    """Block-bootstrap halves: MOVING block bootstrap (blocks drawn from
    every valid overlapping start offset 0..n-block_len, not a fixed
    partition into block_len-sized bins) -- standard Kunsch/Politis-Romano
    block bootstrap. Each half is an independent with-replacement resample
    of ceil(n/block_len) such blocks, concatenated and truncated to n.

    v2: the FIRST implementation partitioned
    the ~400-token document into only 4 FIXED non-overlapping 100-token
    blocks, then sampled 4 of those 4 (with replacement) per half. With a
    pool that small, the two halves shared ~1.9 of 4 blocks ON AVERAGE
    (measured directly: only 2.5% of draws shared zero blocks) -- meaning
    the "two halves" were mostly the SAME tokens by construction, which
    inflated the measured split-half correlation to ~0.78 for gemma_H
    (higher than every other construction, including the untouched
    original document) for a purely mechanical reason having nothing to do
    with document-level reliability. Moving-block resampling draws from
    n-block_len+1 (here ~300) possible start offsets instead of 4 fixed
    bins, which cuts the same-block collision rate by roughly two orders of
    magnitude (empirically verified before use, not assumed) and eliminates
    the with-replacement-from-a-tiny-pool inflation this way."""
    def _split(arr, rng):
        n = len(arr)
        if n <= block_len:
            return arr, arr.copy()
        n_starts = n - block_len + 1
        n_blocks_needed = -(-n // block_len)  # ceil

        def _resample():
            starts = rng.integers(0, n_starts, size=n_blocks_needed)
            out = np.concatenate([arr[s:s + block_len] for s in starts])
            return out[:n]
        return _resample(), _resample()
    _split.__name__ = "block_bootstrap"
    return _split


CONSTRUCTIONS = {
    "interleave": split_interleave,
    "contiguous": split_contiguous,
    "block_4": make_split_block(4),
    "block_8": make_split_block(8),
    "block_16": make_split_block(16),
    "block_32": make_split_block(32),
    "block_bootstrap": make_split_block_bootstrap(100),
    # matched_sweep uses the interleave split but a different windowed scale;
    # handled as a variant flag in run_construction(), not a distinct splitter.
}


# --------------------------------------------------------------------------
# Corpus loading (once; reused across every construction)
# --------------------------------------------------------------------------
def load_corpus(shard_dir=SHARD_DIR, verbose=True):
    files = sorted(glob.glob(os.path.join(shard_dir, "*.parquet")))
    cols = ["id", "domain", "Label", "temp", "prompt_type"] + [c for _, c in SCORERS]
    recs = {"id": [], "domain": [], "Label": [], "temp": [], "prompt_type": [],
            "gemma": [], "mistral": []}
    t0 = time.time()
    for i, f in enumerate(files, 1):
        t = pq.read_table(f, columns=cols)
        d = t.to_pydict()
        recs["id"].extend(d["id"])
        recs["domain"].extend(d["domain"])
        recs["Label"].extend(d["Label"])
        recs["temp"].extend(d["temp"])
        recs["prompt_type"].extend(d["prompt_type"])
        recs["gemma"].extend(d["gagle_gemma_sliced_scores"])
        recs["mistral"].extend(d["gagle_mistral_sliced_scores"])
        if verbose and i % 100 == 0:
            print(f"[load] {i}/{len(files)} shards, {len(recs['id']):,} docs, "
                  f"{time.time()-t0:.0f}s", flush=True)
    if verbose:
        print(f"[load] done: {len(files)} shards, {len(recs['id']):,} docs, "
              f"{time.time()-t0:.0f}s", flush=True)
    return recs


# --------------------------------------------------------------------------
# Stratified (within) correlation -- Route 1
# --------------------------------------------------------------------------
def demean_by_stratum(values, strata_keys):
    """Subtract each stratum's sample mean from `values`. strata_keys is a
    list of tuples (Label, domain, temp), same length as values."""
    values = np.asarray(values, dtype=np.float64)
    from collections import defaultdict
    sums = defaultdict(float)
    counts = defaultdict(int)
    for v, k in zip(values, strata_keys):
        sums[k] += v
        counts[k] += 1
    means = {k: sums[k] / counts[k] for k in sums}
    return np.array([v - means[k] for v, k in zip(values, strata_keys)])


def pooled_and_within_stats(a, b, strata_keys, rho_min=RHO_MIN):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = len(a)
    out = {"n": int(n)}
    if n < 100 or a.std() == 0 or b.std() == 0:
        return None
    r_pooled = float(np.corrcoef(a, b)[0, 1])
    sp_pooled = float(stats.spearmanr(a, b).statistic)
    sb_pooled = 2 * r_pooled / (1 + r_pooled) if r_pooled > -1 else float("nan")
    out.update({"r_pooled": r_pooled, "rho_pooled": sb_pooled, "spearman_pooled": sp_pooled})

    a_w = demean_by_stratum(a, strata_keys)
    b_w = demean_by_stratum(b, strata_keys)
    if a_w.std() > 0 and b_w.std() > 0:
        r_within = float(np.corrcoef(a_w, b_w)[0, 1])
        sp_within = float(stats.spearmanr(a_w, b_w).statistic)
        sb_within = 2 * r_within / (1 + r_within) if r_within > -1 else float("nan")
    else:
        r_within = sp_within = sb_within = float("nan")
    out.update({"r_within": r_within, "rho_within": sb_within, "spearman_within": sp_within})
    out["passes_pooled"] = bool(sb_pooled >= rho_min)
    out["passes_within"] = bool(sb_within >= rho_min) if sb_within == sb_within else False
    return out


def fisher_z_ci(r, n, alpha=0.05):
    """Fisher-z CI for a Pearson r (used on the SPEARMAN-BROWN corrected rho,
    matching the original P2 report's own convention of quoting the CI on
    rho, not on the half-correlation r_half)."""
    if not (-1 < r < 1) or n < 4:
        return (float("nan"), float("nan"))
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    lo, hi = z - 1.959963984540054 * se, z + 1.959963984540054 * se
    return float(np.tanh(lo)), float(np.tanh(hi))


def cluster_robust_se_pearson_r(a, b, cluster_ids, n_boot=2000, seed=20260908):
    """Cluster-bootstrap SE of Pearson r, clustered by `cluster_ids` (here:
    seed article, via src/splits.py's group_key()). Resamples CLUSTERS with
    replacement, recomputes r each time; SE = std of the boot distribution.

    EFFICIENT FORM: Pearson r is a function of five sums (n, sum_a, sum_b,
    sum_ab, sum_a2, sum_b2) that are each LINEAR in how many times a cluster
    is resampled. So instead of concatenating full row-index arrays per
    bootstrap replicate (O(n_boot x N), N=corpus size), precompute those five
    sums ONCE per cluster (O(N)), then get every replicate's r from a
    multiplicity-weighted dot product over the (much smaller) K clusters --
    O(n_boot x K). At K~5,000 seed-article clusters and N~200k+ rows this is
    roughly a 40x-per-replicate reduction, and the K x n_boot multiplicity
    matrix lets all replicates be computed in one vectorized matmul rather
    than a Python-level loop."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    cluster_ids = np.asarray(cluster_ids)
    uniq, inv = np.unique(cluster_ids, return_inverse=True)
    K = len(uniq)

    n_c = np.bincount(inv, minlength=K).astype(np.float64)
    sum_a = np.bincount(inv, weights=a, minlength=K)
    sum_b = np.bincount(inv, weights=b, minlength=K)
    sum_ab = np.bincount(inv, weights=a * b, minlength=K)
    sum_a2 = np.bincount(inv, weights=a * a, minlength=K)
    sum_b2 = np.bincount(inv, weights=b * b, minlength=K)

    rng = np.random.default_rng(seed)
    # multiplicities[i, c] = number of times cluster c is drawn in replicate i
    picks = rng.integers(0, K, size=(n_boot, K))
    mult = np.zeros((n_boot, K), dtype=np.float64)
    for i in range(n_boot):
        mult[i] = np.bincount(picks[i], minlength=K)

    N = mult @ n_c
    Sa = mult @ sum_a
    Sb = mult @ sum_b
    Sab = mult @ sum_ab
    Sa2 = mult @ sum_a2
    Sb2 = mult @ sum_b2

    cov = Sab / N - (Sa / N) * (Sb / N)
    var_a = Sa2 / N - (Sa / N) ** 2
    var_b = Sb2 / N - (Sb / N) ** 2
    denom = np.sqrt(np.maximum(var_a, 0) * np.maximum(var_b, 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = np.where(denom > 0, cov / denom, np.nan)
    boots = boots[~np.isnan(boots)]
    return float(np.std(boots)), int(len(boots))
