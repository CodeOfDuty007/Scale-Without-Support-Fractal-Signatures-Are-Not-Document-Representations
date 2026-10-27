#!/usr/bin/env python3
"""
Consolidation pass: merges every T0/T1 split-construction stats file plus the
permutation distribution into one table (the plan's own "merge T0's table
with T1's" instruction), and composes T2's N* curve from T0's Var(delta) and
T2 Step 1's noise floor.
"""
import glob
import json
import os

from fractalsig import paths as _paths

DATA_DIR = _paths.STATS_DIR_STR
HEADLINE = ["gemma_H", "gemma_S", "mistral_H", "mistral_S"]
DETERMINISTIC = ["interleave", "contiguous", "block_4", "block_8", "block_16", "block_32",
                  "block_bootstrap", "matched_sweep"]
# block_bootstrap is measured and reported, but EXCLUDED from the headline
# cross-construction reliability comparison: with block_len>=96 (the plan's
# own constraint, the largest fit scale) on ~400-token documents, each block
# covers ~25% of the document, so two independently-drawn block-bootstrap
# halves share ~47% of their token positions even under moving-block
# resampling (measured directly, not assumed -- see reliability_lib.py's
# make_split_block_bootstrap docstring). This is a structural property of
# the construction at this document length, not a bug: the two "halves" are
# not independent, so their inflated correlation (~0.78 for gemma_H, above
# EVERY other construction including the untouched document) is not
# comparable to genuine split-half reliability and would corrupt the
# headline test if left in.
EXCLUDE_FROM_HEADLINE = {
    "block_bootstrap": ("block_len>=96 on ~400-token documents structurally forces "
                          "~47% token-position overlap between the two nominally "
                          "independent halves (measured directly) -- not a genuine "
                          "split-half reliability number."),
}


def consolidate_t0_t1():
    table = {q: {} for q in HEADLINE}
    for c in DETERMINISTIC:
        path = os.path.join(DATA_DIR, f"t0_t1_{c}_stats.json")
        if not os.path.exists(path):
            print(f"[consolidate] MISSING: {path} (construction not finished/run)")
            continue
        d = json.load(open(path))
        for q in HEADLINE:
            if q in d["features"]:
                f = d["features"][q]
                table[q][c] = {"rho_pooled": f["rho_pooled"], "rho_within": f["rho_within"], "n": f["n"]}

    perm_path = os.path.join(DATA_DIR, "t1_permutation_stats.json")
    if os.path.exists(perm_path):
        d = json.load(open(perm_path))
        for q in HEADLINE:
            if q in d["summary"]:
                s = d["summary"][q]
                table[q]["permutation"] = {
                    "rho_pooled_median": s["rho_pooled_median"],
                    "rho_pooled_p5": s["rho_pooled_p5"], "rho_pooled_p95": s["rho_pooled_p95"],
                    "rho_within_median": s["rho_within_median"], "n_reps": s["n_reps"],
                }
    else:
        print(f"[consolidate] MISSING: {perm_path} (permutation not finished)")

    print("\n" + "=" * 100)
    print("MERGED T0/T1 TABLE  (rho_pooled | rho_within, by construction)")
    print("=" * 100)
    for q in HEADLINE:
        print(f"\n{q}:")
        for c, v in table[q].items():
            if "rho_pooled_median" in v:
                print(f"  {c:16} pooled_median={v['rho_pooled_median']:.4f} "
                      f"[{v['rho_pooled_p5']:.4f},{v['rho_pooled_p95']:.4f}]  "
                      f"within_median={v['rho_within_median']:.4f}  (n_reps={v['n_reps']})")
            else:
                print(f"  {c:16} pooled={v['rho_pooled']:.4f}  within={v['rho_within']:.4f}  (n={v['n']:,})")

    # headline test: does gemma_S stay in 0.03-0.12 across all VALID constructions?
    # block_bootstrap is excluded (see EXCLUDE_FROM_HEADLINE) because it is not a
    # genuine split-half measurement at this document length; "permutation" and
    # "contiguous" are retained since both are informative (contiguous is expected,
    # and pre-registered, to FAIL low as evidence against drift confounding).
    excluded_here = {c: v for c, v in table["gemma_S"].items() if c in EXCLUDE_FROM_HEADLINE}
    gemma_s_items = [(c, v["rho_pooled"]) for c, v in table["gemma_S"].items()
                      if "rho_pooled" in v and c not in EXCLUDE_FROM_HEADLINE]
    if "permutation" in table["gemma_S"]:
        gemma_s_items.append(("permutation", table["gemma_S"]["permutation"]["rho_pooled_median"]))
    in_band = all(0.03 <= v <= 0.12 for _, v in gemma_s_items) if gemma_s_items else None
    below_band = [(c, v) for c, v in gemma_s_items if v < 0.03]
    above_band = [(c, v) for c, v in gemma_s_items if v > 0.12]
    print(f"\n[headline test] gemma_S rho_pooled stays in [0.03,0.12] across valid constructions "
          f"(excludes {list(EXCLUDE_FROM_HEADLINE)}): {in_band}")
    print(f"    values: {[(c, round(v,4)) for c, v in gemma_s_items]}")
    if below_band:
        print(f"    BELOW band (expected for 'contiguous' -- drift confound, evidence FOR its "
              f"rejection, not a test failure): {[(c, round(v,4)) for c, v in below_band]}")
    if above_band:
        print(f"    ABOVE band: {[(c, round(v,4)) for c, v in above_band]}")

    headline_test = {
        "in_band_0.03_0.12": in_band,
        "values_used": {c: v for c, v in gemma_s_items},
        "excluded": {c: {"rho_pooled": excluded_here[c]["rho_pooled"], "reason": reason}
                     for c, reason in EXCLUDE_FROM_HEADLINE.items() if c in excluded_here},
        "below_band": {c: v for c, v in below_band},
        "above_band": {c: v for c, v in above_band},
        "note": ("'contiguous' falling below the band is pre-registered, expected evidence "
                 "AGAINST that construction (drift confound: gemma_H collapses 0.576->0.261 "
                 "under contiguous), not a headline-test failure in the sense of undermining "
                 "gemma_S's reliability claim."),
    }
    table["_headline_test_gemma_S"] = headline_test

    json.dump(table, open(os.path.join(DATA_DIR, "t0_t1_MERGED_table.json"), "w"), indent=2)
    print(f"\n[written] {os.path.join(DATA_DIR, 't0_t1_MERGED_table.json')}")
    return table


def compose_n_star():
    vc_path = os.path.join(DATA_DIR, "t0_variance_components.json")
    floor_path = os.path.join(DATA_DIR, "t2_step1_noise_floor.json")
    if not (os.path.exists(vc_path) and os.path.exists(floor_path)):
        print("[N*] missing t0_variance_components.json or t2_step1_noise_floor.json, skipping")
        return

    vc = json.load(open(vc_path))["features"]
    floor = json.load(open(floor_path))["results"]

    out = {}
    for q, est_key in (("gemma_S", "S"), ("gemma_H", "H"), ("mistral_S", "S"), ("mistral_H", "H")):
        if q not in vc:
            continue
        var_delta = vc[q]["var_delta_document"]
        # average the noise variance across the calibrated H grid (0.60,0.65,0.70,0.72) at each N
        by_n = {}
        for row in floor:
            if row["H_true"] not in (0.60, 0.65, 0.70, 0.72):
                continue
            n = row["N"]
            sd_key = f"{est_key}_hat_sd"
            by_n.setdefault(n, []).append(row[sd_key] ** 2)
        curve = []
        n_star = None
        for n in sorted(by_n):
            var_eta = sum(by_n[n]) / len(by_n[n])
            # r_N is the correlation between two HALF-length (single) measurements
            # separated by noise Var(eta_N) -- the pre-registered 0.60 bar was set
            # on the SPEARMAN-BROWN-CORRECTED coefficient (full-length reliability),
            # not on this raw r. Apply the same 2r/(1+r) correction used everywhere
            # else in T0/T1 before comparing to 0.60 -- review-identified bug: comparing
            # an uncorrected r to a corrected bar systematically overstates N*.
            r_n = var_delta / (var_delta + var_eta) if (var_delta + var_eta) > 0 else float("nan")
            rho_n = 2 * r_n / (1 + r_n) if r_n == r_n and r_n > -1 else float("nan")
            curve.append({"N": n, "var_eta_N": var_eta, "r_N": r_n, "rho_N_spearman_brown": rho_n})
            if n_star is None and rho_n >= 0.60:
                n_star = n
        out[q] = {"var_delta": var_delta, "curve": curve,
                   "N_star_fBm_floor": n_star,
                   "note": ("N* is the first N_grid point in {100..12800} where the "
                            "Spearman-Brown-corrected rho(N) crosses 0.60 (matching the bar's "
                            "own pre-registration on the corrected coefficient, not raw r); "
                            "true crossing may lie between grid points. "
                            + ("For S, this fBm-arm N* is a lower bound: at N=400 the "
                               "composed reliability overstates measured split-half reliability "
                               "by 3.7–4.0x, so correcting the optimistic floor moves the "
                               "crossing farther out." if est_key == "S" else
                               "For H, this fBm-arm N* is a point estimate under the "
                               "validated fBm noise floor.")),
                   }
        print(f"\n[N*] {q}: Var(delta)={var_delta:.5f}  N*={n_star}")
        for row in out[q]["curve"]:
            print(f"    N={row['N']:6d}  Var(eta_N)={row['var_eta_N']:.5f}  "
                  f"r(N)={row['r_N']:.4f}  rho_SB(N)={row['rho_N_spearman_brown']:.4f}")

    json.dump(out, open(os.path.join(DATA_DIR, "t2_n_star_composition.json"), "w"), indent=2)
    print(f"\n[written] {os.path.join(DATA_DIR, 't2_n_star_composition.json')}")


if __name__ == "__main__":
    consolidate_t0_t1()
    compose_n_star()
