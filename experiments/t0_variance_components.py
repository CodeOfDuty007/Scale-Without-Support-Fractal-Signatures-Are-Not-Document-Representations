#!/usr/bin/env python3
"""
T0 Route 2 -- variance decomposition + cluster-robust SE, built on top of the
T0/T1 interleave pass output (outputs/stats/t0_t1_interleave_features.parquet
and t0_t1_interleave_stats.json, produced by run_construction.py).

METHOD (closed-form G-theory decomposition, not an iterative mixed model):
For a feature value on a single half h of document d in stratum s = (Label,
domain, temp): value(d,h) = mu_s + delta_d + eta_{d,h}, with eta_a, eta_b iid
noise. Then, on POOLED single-half observations (both halves treated as
separate draws, N = 2 x n_docs):

  Var(mu_s)              = one-way random-effects ANOVA between-stratum
                            variance component (classic unbalanced method-of-
                            moments: (MSB - MSW) / n0).
  Var(delta) + Var(eta)  = MSW from that same one-way ANOVA (the within-
                            stratum variance of a single half-observation).
  Var(delta), Var(eta)   = split MSW using r_within (already computed in
                            Route 1 / the stats json): Corr(half_a-mean_s,
                            half_b-mean_s) = Var(delta)/(Var(delta)+Var(eta))
                            by construction (independent eta_a, eta_b), so
                            Var(delta) = r_within * MSW, Var(eta) = (1 -
                            r_within) * MSW. This reuses Route 1's own
                            already-validated statistic rather than fitting a
                            second, separate model that could disagree with
                            it for no principled reason.
  ICC                     = Var(mu_s) / (Var(mu_s) + MSW)

Var(Label), Var(domain), Var(temp) are each also reported MARGINALLY (one-way
ANOVA on that single factor alone, ignoring the other two) -- these are
descriptive, non-orthogonal contributions, not an adjusted/Type-II
decomposition; documented as such rather than presented as if they summed to
Var(mu_s).

This closed-form route was chosen over an iterative crossed-random-effects
mixed model (e.g. statsmodels MixedLM with vc_formula) specifically because
it has NO convergence risk at n=223,834 documents -- same rationale Route 1
already uses for its own zero-solver headline number. Documented here as a
deliberate substitution, not a shortcut taken silently.

Cluster-robust SE: cluster-bootstrap (2000 reps) of the POOLED Pearson r,
clustered by "seed article" = (domain, dataset_id) via src/splits.py's own
group_key() -- the same grouping the project's frozen train/val/test split
is built on -- rather than by the (Label,domain,temp) stratum.
"""
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

from fractalsig import paths as _paths  # noqa: E402
from fractalsig.detection.splits import group_key  # noqa: E402
from experiments.reliability_lib import cluster_robust_se_pearson_r  # noqa: E402

DATA_DIR = _paths.STATS_DIR_STR


def one_way_anova_vc(values, group_keys):
    """Classic unbalanced one-way random-effects ANOVA variance component.
    Returns (var_between, mswithin, k_groups, n0)."""
    values = np.asarray(values, dtype=np.float64)
    groups = {}
    for v, g in zip(values, group_keys):
        groups.setdefault(g, []).append(v)
    k = len(groups)
    N = len(values)
    grand_mean = values.mean()
    ssb = 0.0
    ssw = 0.0
    sum_ni2 = 0.0
    for g, vals in groups.items():
        vals = np.asarray(vals)
        ni = len(vals)
        sum_ni2 += ni * ni
        gm = vals.mean()
        ssb += ni * (gm - grand_mean) ** 2
        ssw += np.sum((vals - gm) ** 2)
    msb = ssb / (k - 1) if k > 1 else float("nan")
    msw = ssw / (N - k) if N > k else float("nan")
    n0 = (N - sum_ni2 / N) / (k - 1) if k > 1 else float("nan")
    var_between = max(0.0, (msb - msw) / n0) if n0 and n0 == n0 and n0 > 0 else float("nan")
    return var_between, msw, k, n0


def main():
    t0 = time.time()
    pq_path = os.path.join(DATA_DIR, "t0_t1_interleave_features.parquet")
    stats_path = os.path.join(DATA_DIR, "t0_t1_interleave_stats.json")
    table = pq.read_table(pq_path)
    df_stats = json.load(open(stats_path))["features"]

    d = table.to_pydict()
    n_rows = len(d["id"])
    print(f"[t0-route2] loaded {n_rows:,} long-format rows", flush=True)

    # cluster ids via group_key, computed once per unique (id, domain, Label)
    print("[t0-route2] computing seed-article cluster keys...", flush=True)
    id_domain_label = list(zip(d["id"], d["domain"], d["Label"]))
    uniq_triples = list(set(id_domain_label))
    gk_cache = {}
    for (rid, dom, lab) in uniq_triples:
        gk = group_key({"id": rid, "domain": dom, "Label": lab})
        gk = gk if gk is not None else (dom, rid)  # fallback: doc is its own cluster
        gk_cache[(rid, dom, lab)] = f"{gk[0]}||{gk[1]}"  # flatten to a string: a list/array of
        # equal-length tuples silently becomes a 2D numpy array, not an array-of-tuples
    cluster_id_full = [gk_cache[t] for t in id_domain_label]  # plain list of tuples -- NOT np.array
    # (np.array() of same-length tuples silently builds a 2D array, not an array-of-tuples)
    print(f"[t0-route2] {len(set(cluster_id_full))} unique seed-article clusters "
          f"over {n_rows:,} rows (long-format; documents appear once per scorer x feature)", flush=True)

    feats = sorted(set(zip(d["scorer"], d["feature"])))
    out = {}
    scorer_arr = np.asarray(d["scorer"])
    feature_arr = np.asarray(d["feature"])
    label_arr = np.asarray(d["Label"])
    domain_arr = np.asarray(d["domain"])
    temp_arr = np.asarray(d["temp"])
    half_a = np.asarray(d["half_a"], dtype=np.float64)
    half_b = np.asarray(d["half_b"], dtype=np.float64)

    for sname, feat in feats:
        key = f"{sname}_{feat}"
        mask = (scorer_arr == sname) & (feature_arr == feat)
        a, b = half_a[mask], half_b[mask]
        strata = list(zip(label_arr[mask], domain_arr[mask], temp_arr[mask]))
        labels_only = label_arr[mask]
        domains_only = domain_arr[mask]
        temps_only = temp_arr[mask]
        mask_idx = np.flatnonzero(mask)
        cluster_ids = np.array([cluster_id_full[i] for i in mask_idx], dtype=object)
        n = len(a)
        if n < 100:
            continue

        pooled_vals = np.concatenate([a, b])
        pooled_strata = strata + strata
        pooled_label = np.concatenate([labels_only, labels_only])
        pooled_domain = np.concatenate([domains_only, domains_only])
        pooled_temp = np.concatenate([temps_only, temps_only])

        var_mu_stratum, msw, k_strata, n0 = one_way_anova_vc(pooled_vals, pooled_strata)
        var_label, _, k_label, _ = one_way_anova_vc(pooled_vals, pooled_label)
        var_domain, _, k_domain, _ = one_way_anova_vc(pooled_vals, pooled_domain)
        var_temp, _, k_temp, _ = one_way_anova_vc(pooled_vals, pooled_temp)

        r_within = df_stats.get(key, {}).get("r_within", float("nan"))
        if r_within == r_within and msw == msw:
            var_delta = max(0.0, r_within) * msw
            var_eta = max(0.0, 1 - max(0.0, r_within)) * msw
        else:
            var_delta = var_eta = float("nan")
        icc = var_mu_stratum / (var_mu_stratum + msw) if (var_mu_stratum == var_mu_stratum and msw == msw and (var_mu_stratum + msw) > 0) else float("nan")

        se_cluster, n_valid_boot = cluster_robust_se_pearson_r(a, b, cluster_ids, n_boot=2000)
        rho_pooled = df_stats.get(key, {}).get("rho_pooled", float("nan"))
        r_pooled = df_stats.get(key, {}).get("r_pooled", float("nan"))

        out[key] = {
            "n": int(n), "k_strata": int(k_strata),
            "var_mu_stratum_Label_domain_temp": var_mu_stratum,
            "var_Label_marginal": var_label, "var_domain_marginal": var_domain, "var_temp_marginal": var_temp,
            "var_delta_document": var_delta, "var_eta_half_residual": var_eta,
            "msw_within_stratum_total": msw, "icc_between_stratum": icc,
            "r_within_used_for_split": r_within,
            "r_pooled": r_pooled, "rho_pooled_spearman_brown": rho_pooled,
            "se_r_pooled_cluster_robust_seed_article": se_cluster,
            "n_valid_bootstrap_reps": n_valid_boot,
            "n_seed_article_clusters": int(len(set(cluster_ids.tolist()))),
        }
        print(f"[t0-route2] {key}: Var(mu_stratum)={var_mu_stratum:.5f} Var(delta)={var_delta:.5f} "
              f"Var(eta)={var_eta:.5f} ICC={icc:.4f} r_pooled={r_pooled:.4f} "
              f"cluster-SE={se_cluster:.5f}", flush=True)

    out_path = os.path.join(DATA_DIR, "t0_variance_components.json")
    json.dump({"method": "closed-form one-way ANOVA variance components + r_within split "
                          "(see module docstring); cluster-robust SE via seed-article cluster bootstrap",
               "runtime_sec": time.time() - t0, "features": out}, open(out_path, "w"), indent=2)
    print(f"\n[written] {out_path}", flush=True)
    print(f"[t0-route2] TOTAL {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
