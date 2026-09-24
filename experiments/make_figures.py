#!/usr/bin/env python3
"""
Render the T0-T10 summary figures from the statistics JSON files.

This script computes no statistics; it only plots values already written by the
experiment scripts. By default it reads $FRACTALSIG_STATS_DIR (outputs/stats) and
writes to outputs/figures. To re-render the figures from the statistics shipped
with the repository (no data download, CPU only, < 1 minute):

    python -m experiments.make_figures --stats-dir reference_outputs/stats
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fractalsig import paths as _paths

DATA = _paths.STATS_DIR_STR
FIG = _paths.FIGURES_DIR_STR

RHO_MIN = 0.60


def load(name):
    return json.load(open(os.path.join(DATA, name)))


def savefig(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {path}")


# ---------------------------------------------------------------------------
def fig_t0_t1():
    d = load("t0_t1_MERGED_table.json")
    quantities = ["gemma_H", "gemma_S", "mistral_H", "mistral_S"]
    constructions = ["interleave", "contiguous", "block_4", "block_8", "block_16",
                      "block_32", "matched_sweep", "permutation", "block_bootstrap"]
    excluded = {"block_bootstrap"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    for ax, q in zip(axes.flat, quantities):
        vals, labels, colors = [], [], []
        for c in constructions:
            if c not in d[q]:
                continue
            v = d[q][c]
            rho = v.get("rho_pooled") or v.get("rho_pooled_median")
            vals.append(rho)
            labels.append(c + ("*" if c in excluded else ""))
            colors.append("#c96b6b" if c in excluded else
                           ("#e8a33d" if c == "contiguous" else "#4c78a8"))
        x = np.arange(len(vals))
        ax.bar(x, vals, color=colors)
        ax.axhline(RHO_MIN, color="black", linestyle="--", linewidth=1, label="0.60 bar")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(q)
        ax.set_ylabel("rho_pooled (SB-corrected)")
        ax.set_ylim(0, max(vals) * 1.15)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle("T0/T1: reliability across split constructions\n"
                  "(* = excluded from headline test, see paper appendix; orange = pre-registered drift confound)",
                  fontsize=11)
    fig.tight_layout()
    savefig(fig, "t0_t1_reliability_by_construction.png")


# ---------------------------------------------------------------------------
def fig_t2_n_star():
    d = load("t2_n_star_composition.json")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for q, style in [("gemma_H", "-o"), ("mistral_H", "-s"),
                      ("gemma_S", "--o"), ("mistral_S", "--s")]:
        curve = d[q]["curve"]
        ns = [r["N"] for r in curve]
        rhos = [r["rho_N_spearman_brown"] for r in curve]
        ax.plot(ns, rhos, style, label=f'{q} (N*={d[q]["N_star_fBm_floor"]})')
    ax.axhline(RHO_MIN, color="black", linestyle=":", linewidth=1, label="0.60 bar")
    ax.axvline(400, color="gray", linestyle=":", linewidth=1)
    ax.text(400, 0.02, "GAGLE\n~400 tok", fontsize=8, ha="center", color="gray")
    ax.set_xscale("log")
    ax.set_xlabel("N (tokens, fBm floor)")
    ax.set_ylabel("rho_SB(N)")
    ax.set_title("T2: reliability vs. support (N*) under the fBm noise floor")
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, "t2_n_star_curve.png")


def fig_t2_noise_floor():
    d = load("t2_step1_noise_floor.json")["results"]
    band = [r for r in d if r["H_true"] in (0.60, 0.65, 0.70, 0.72)]
    ns = sorted(set(r["N"] for r in band))
    s_bias = [np.mean([r["S_bias"] for r in band if r["N"] == n]) for n in ns]
    s_sd = [np.mean([r["S_hat_sd"] for r in band if r["N"] == n]) for n in ns]
    h_bias = [np.mean([r["H_bias"] for r in band if r["N"] == n]) for n in ns]
    h_sd = [np.mean([r["H_hat_sd"] for r in band if r["N"] == n]) for n in ns]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].errorbar(ns, s_bias, yerr=s_sd, fmt="-o", capsize=3, label="S", color="#c96b6b")
    axes[0].errorbar(ns, h_bias, yerr=h_sd, fmt="-s", capsize=3, label="H", color="#4c78a8")
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("N (tokens)")
    axes[0].set_ylabel("estimator bias (+/- sd)")
    axes[0].set_title("fBm noise floor: bias/sd vs. N\n(H_true averaged over 0.60-0.72 band)")
    axes[0].legend()

    axes[1].plot(ns, s_sd, "-o", label="S sd", color="#c96b6b")
    axes[1].plot(ns, h_sd, "-s", label="H sd", color="#4c78a8")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("N (tokens)")
    axes[1].set_ylabel("sd (log scale)")
    axes[1].set_title("Noise floor collapse rate")
    axes[1].legend()
    fig.tight_layout()
    savefig(fig, "t2_step1_noise_floor.png")


# ---------------------------------------------------------------------------
def fig_t3():
    d = load("t3_population_control.json")
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = ["T3.1\npopulation-only\n(structure alone)",
              "T3.3\nfractal_only\nbefore demean",
              "T3.3\nfractal_only\nafter demean",
              "T3.4\nfractal_only\nwithin-stratum\n(mean of 30 cells)",
              "T3.3\nall-features\nbefore demean",
              "T3.3\nall-features\nafter demean"]
    vals = [d["t3_1_population_only_control"]["macro_f1"],
            d["t3_3_group_mean_removal"]["fractal_only_before"],
            d["t3_3_group_mean_removal"]["fractal_only_after"],
            d["t3_4_within_stratum"]["distribution_fractal_only"]["mean"],
            d["t3_3_group_mean_removal"]["all_features_before"],
            d["t3_3_group_mean_removal"]["all_features_after_fractal_demeaned"]]
    colors = ["#999999", "#c96b6b", "#c96b6b", "#c96b6b", "#4c78a8", "#4c78a8"]
    x = np.arange(len(vals))
    ax.bar(x, vals, color=colors)
    ax.axhline(0.125, color="black", linestyle="--", linewidth=1, label="8-class chance (0.125)")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylabel("macro-F1 (prompt-type classification)")
    ax.set_title("T3: P3 signal survives population-structure control\n"
                  "(red = fractal-only features, blue = all features, grey = structure alone)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, "t3_population_control.png")


# ---------------------------------------------------------------------------
def fig_t4():
    d = load("t4_alt_estimators_reliability.json")["reliability"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    estimators = ["RS", "DFA", "Whittle"]
    x = np.arange(len(estimators))
    width = 0.35
    for i, scorer in enumerate(["gemma", "mistral"]):
        vals = [d[f"{scorer}_{e}_H"]["rho_pooled"] for e in estimators]
        axes[0].bar(x + (i - 0.5) * width, vals, width, label=scorer)
    axes[0].axhline(RHO_MIN, color="black", linestyle="--", linewidth=1, label="0.60 bar")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(estimators)
    axes[0].set_ylabel("rho_pooled")
    axes[0].set_title("T4: Hurst-exponent reliability by estimator")
    axes[0].legend(fontsize=9)

    cec = load("t4_alt_estimators_reliability.json")["cross_estimator_correlation"]
    pairs = ["RS~DFA", "RS~Whittle", "DFA~Whittle"]
    x2 = np.arange(len(pairs))
    for i, scorer in enumerate(["gemma", "mistral"]):
        vals = [cec[scorer][f"corr_{p.replace('~','_')}"] for p in pairs]
        axes[1].bar(x2 + (i - 0.5) * width, vals, width, label=scorer)
    axes[1].axhline(0.85, color="green", linestyle=":", linewidth=1, label="same-quantity (>0.85)")
    axes[1].axhline(0.60, color="red", linestyle=":", linewidth=1, label="different-quantity (<0.60)")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(pairs)
    axes[1].set_ylabel("doc-level correlation")
    axes[1].set_title("Cross-estimator correlation\n(decides same- vs different-quantity)")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    savefig(fig, "t4_estimator_comparison.png")


# ---------------------------------------------------------------------------
def fig_t5():
    d = load("t5_raid_group_mean.json")
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["all-85\n(reference)", "all-minus-\nfractal31", "REPLACEMENT\n(->group mean)",
              "SUBTRACTION\n(-group mean\n+global mean)"]
    vals = [d["reference_gap_all85_minus_all_minus_fractal31"]["auroc_all85"],
            d["reference_gap_all85_minus_all_minus_fractal31"]["auroc_minus_fractal"],
            d["replacement"]["auroc"],
            d["subtraction"]["auroc"]]
    colors = ["#4c78a8", "#c96b6b", "#54a24b", "#e8a33d"]
    ax.bar(np.arange(4), vals, color=colors)
    ax.set_ylim(0.75, 0.83)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("AUROC (RAID cross-corpus)")
    ax.set_title("T5: RAID group-mean test\n"
                  f"replacement recovers {d['replacement']['fraction_of_gap_recovered']*100:.0f}% of gap; "
                  f"subtraction retains {d['subtraction']['fraction_of_gap_remaining']*100:.0f}%")
    fig.tight_layout()
    savefig(fig, "t5_raid_group_mean.png")


# ---------------------------------------------------------------------------
def fig_t7():
    d = load("t7_matched_length.json")
    bands = ["320-360", "361-400", "401-464"]
    quantities = ["gemma_H", "gemma_S", "mistral_H", "mistral_S"]
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.2
    x = np.arange(len(bands))
    for i, q in enumerate(quantities):
        vals = [d["by_band"][b][q]["rho_pooled"] for b in bands]
        ax.bar(x + (i - 1.5) * width, vals, width, label=q)
    ax.axhline(RHO_MIN, color="black", linestyle="--", linewidth=1, label="0.60 bar")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} tok\n(n={d['by_band'][b]['n_rows_all_features']:,})" for b in bands])
    ax.set_ylabel("rho_pooled")
    ax.set_title("T7: reliability by token-length band\n"
                  "(361-400 dominated by Human's fixed 400-token value)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    savefig(fig, "t7_matched_length.png")


# ---------------------------------------------------------------------------
def fig_t8():
    d = load("t8_deployment_metrics.json")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sweep = d["ppv_by_deployment_llm_prevalence"]
    order = ["llm_share_0.01", "llm_share_0.05", "llm_share_0.10", "llm_share_0.25",
             "llm_share_0.50", "gagle_corpus_composition"]
    xlabels = ["1%", "5%", "10%", "25%", "50%", f"GAGLE\n({sweep['gagle_corpus_composition']['assumed_llm_prevalence']*100:.0f}%)"]
    p1 = [sweep[k]["ppv_at_1pct_fpr_threshold"] for k in order]
    p5 = [sweep[k]["ppv_at_5pct_fpr_threshold"] for k in order]
    x = np.arange(len(order))
    axes[0].plot(x, p1, "-o", label="threshold @ 1% FPR")
    axes[0].plot(x, p5, "-s", label="threshold @ 5% FPR")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(xlabels)
    axes[0].set_xlabel("assumed deployment LLM prevalence")
    axes[0].set_ylabel("PPV")
    axes[0].set_title("T8: PPV vs. deployment prevalence\n(y=1 is LLM; GAGLE's own composition is an outlier reference)")
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(0, 1.02)

    metrics = ["AUROC\n(full)", "partial AUROC\n(FPR<=1%)", "partial AUROC\n(FPR<=5%)",
               "TPR\n@1% FPR", "TPR\n@5% FPR"]
    mvals = [d["auroc_full"], d["partial_auroc_fpr_le_1pct"], d["partial_auroc_fpr_le_5pct"],
             d["tpr_at_1pct_fpr"], d["tpr_at_5pct_fpr"]]
    axes[1].bar(np.arange(len(metrics)), mvals, color="#4c78a8")
    axes[1].set_xticks(np.arange(len(metrics)))
    axes[1].set_xticklabels(metrics, fontsize=8)
    axes[1].set_ylim(0.9, 1.005)
    axes[1].set_title(f"Deployment metrics (ECE={d['expected_calibration_error_15bins']:.4f})")
    fig.tight_layout()
    savefig(fig, "t8_deployment_metrics.png")


# ---------------------------------------------------------------------------
def fig_t9():
    d = load("t9_raid_complete_case.json")["auroc_by_group_complete_case"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    dom = d["domain"]
    doms = sorted(dom, key=lambda k: dom[k]["auroc"] if isinstance(dom[k], dict) and "auroc" in dom[k] else list(dom[k].values())[0])
    dom_vals = [dom[k]["auroc"] if "auroc" in dom[k] else list(dom[k].values())[0] for k in doms]
    axes[0].barh(np.arange(len(doms)), dom_vals, color="#4c78a8")
    axes[0].set_yticks(np.arange(len(doms)))
    axes[0].set_yticklabels(doms, fontsize=8)
    axes[0].set_xlabel("AUROC")
    axes[0].set_title("T9: RAID complete-case AUROC by domain")
    axes[0].axvline(d["domain"][doms[0]].get("n", 0) and 0.5 or 0.5, color="gray", linewidth=0)

    gen = d["generator_vs_human"]
    gens = sorted(gen, key=lambda k: gen[k]["auroc"])
    gen_vals = [gen[k]["auroc"] for k in gens]
    axes[1].barh(np.arange(len(gens)), gen_vals, color="#e8a33d")
    axes[1].set_yticks(np.arange(len(gens)))
    axes[1].set_yticklabels(gens, fontsize=8)
    axes[1].set_xlabel("AUROC (vs. shared human pool)")
    axes[1].set_title("T9: RAID complete-case AUROC by generator")
    fig.tight_layout()
    savefig(fig, "t9_raid_complete_case.png")


# ---------------------------------------------------------------------------
def fig_t10():
    comp = load("t10_pilot_geometry_comparison.json")
    configs = ["w40_s40", "w15_s15", "w10_s10"]
    mean_nw = [comp[c]["mean_n_windows"] for c in configs]
    feat_keys = ["H1_total_persistence", "H1_entropy", "H1_n_bars", "H1_max_persistence"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for key in feat_keys:
        vals = [comp[c]["features"][key]["partial_r_SB"] for c in configs]
        axes[0].plot(mean_nw, vals, "-o", label=key)
    axes[0].axhline(0.576, color="#4c78a8", linestyle="--", linewidth=1, label="gemma_H (0.576)")
    axes[0].axhline(RHO_MIN, color="black", linestyle=":", linewidth=1, label="0.60 bar")
    axes[0].set_xlabel("mean windows/doc (point-cloud size)")
    axes[0].set_ylabel("length-controlled reliability (SB-corrected partial r)")
    axes[0].set_title("T10: reliability vs. point-cloud density\n(all 3 configs at ZERO content overlap, measured)")
    axes[0].legend(fontsize=8)

    overlap_configs = [("w40/s20\n(baseline)", 0.906), ("w40/s8\n(densified, confounded)", 0.963),
                        ("w40/s40", 0.0), ("w15/s15", 0.0), ("w10/s10", 0.0)]
    labels = [c[0] for c in overlap_configs]
    ovals = [c[1] for c in overlap_configs]
    colors = ["#c96b6b", "#c96b6b", "#54a24b", "#54a24b", "#54a24b"]
    axes[1].bar(np.arange(len(labels)), ovals, color=colors)
    axes[1].set_xticks(np.arange(len(labels)))
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("inter-half word-position overlap (Jaccard)")
    axes[1].set_title("Content overlap by geometry\n(red = confounded, green = zero-overlap, valid)")
    fig.tight_layout()
    savefig(fig, "t10_geometry_sweep.png")


# ---------------------------------------------------------------------------
def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats-dir", default=DATA, help="directory holding the t*.json statistics")
    ap.add_argument("--out-dir", default=FIG, help="directory to write PNG figures to")
    return ap.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    DATA = os.path.abspath(_args.stats_dir)
    FIG = os.path.abspath(_args.out_dir)
    if not os.path.isdir(DATA):
        raise SystemExit(f"[figures] statistics directory not found: {DATA}")
    os.makedirs(FIG, exist_ok=True)
    fig_t0_t1()
    fig_t2_n_star()
    fig_t2_noise_floor()
    fig_t3()
    fig_t4()
    fig_t5()
    fig_t7()
    fig_t8()
    fig_t9()
    fig_t10()
    print("[figures] all done")
