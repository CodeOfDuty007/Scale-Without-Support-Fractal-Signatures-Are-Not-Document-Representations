#!/usr/bin/env python3
"""Render data/p3_t1_results.json into results/p3_tables.md (phase-plan P3 output).

Reports EVERY declared feature's importance including zero-gain ones (D-13):
selective reporting after the fact reads as p-hacking at review.
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
import sys

HERE = _paths.WORKSPACE_STR
R = json.load(open(os.path.join(HERE, "data", "p3_t1_results.json")))
OUT = os.path.join(HERE, "results", "p3_tables.md")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

CFG_LABEL = {
    "full": "1 · full corpus",
    "no_short_window": "2 · short-window rows excluded (D-14)",
    "no_nan_signal": "3 · missingness ablation (imputed)",
    "complete_case": "4 · complete-case (no imputation)",
    "clean_both": "5 · clean — short-window excluded AND complete-case",
}


def cell(c, k):
    ci = c.get("ci95", {}).get(k)
    v = c["point"][k]
    if not ci or "lo" not in ci:
        return f"{v:.4f}"
    star = "" if ci["n_valid_folds"] >= R["n_boot"] else f"†{ci['n_valid_folds']}"
    return f"{v:.4f} [{ci['lo']:.4f}, {ci['hi']:.4f}]{star}"


L = []
w = L.append
w("# P3 Task 5 — T1 binary detection results\n")
w(f"Generated {R['generated']} from `data/p3_t1_results.json`. "
  f"Design: `{R['design_doc']}`.\n")
w(f"**Positive class:** {R['positive_class']}. "
  f"Corpus {R['corpus']['n']:,} rows, {R['corpus']['composition']}, "
  f"human prevalence {100*R['corpus']['prevalence_human']:.3f}%.\n")
w("**D-11 order, and it is load-bearing:** class weighting → calibrate on the "
  "validation split at true prevalence (Platt/sigmoid, `cv='prefit'`) → then "
  "report. `macro-F1@0.5` and `ECE` are computed on calibrated probabilities "
  "only; reporting macro-F1@0.5 on an uncalibrated weight-trained model is "
  "exactly what D-11 forbids.\n")
w("CIs are 95% percentile bootstrap, stratified, 2,000 folds, from the frozen "
  "`fractalsig/detection/metrics.py`. `†k` marks an interval resting on fewer than 2,000 usable "
  "folds. `AP_human` is average precision for detecting the **human** class "
  "(no-skill floor 0.0223); LLM-positive AP has a 0.9777 floor and would "
  "discriminate nothing.\n")

w("\n## Excluded from the feature matrix\n")
_GATE_FLAGS = R['excluded_inputs']['exit_gate_flags']
_UNGATED = {"ZWSP_Present_In_Source_Text"}  # documented (added 2026-08-29): this list
# is EXIT_GATE_FLAGS itself, but load_matrix() never loads this one column for GAGLE at
# all, so the P1 exit gate never evaluates or reclassifies it here -- it's in the list
# for RAID's benefit (p5_exit_gate_raid.py), not because GAGLE's gate acted on it.
_GATE_EVALUATED = [f for f in _GATE_FLAGS if f not in _UNGATED]
w(f"- **{len(_GATE_EVALUATED)} provenance flags** the P1 "
  f"exit gate reclassified as metadata: "
  f"`{'`, `'.join(_GATE_EVALUATED)}`"
  + (f" (`EXIT_GATE_FLAGS` also lists {len(_GATE_FLAGS) - len(_GATE_EVALUATED)} more"
     f" — `{'`, `'.join(f for f in _GATE_FLAGS if f in _UNGATED)}` — documented here as"
     f" a non-input, not gated: `load_matrix()` never loads it for GAGLE, so this gate"
     f" never evaluates it. RAID's own `p5_exit_gate_raid.py` explicitly EXCLUDES bool"
     f" dtypes from its feature auto-detection, so it doesn't gate on this column"
     f" either — it stays a provenance flag everywhere, never a candidate feature.)"
     if len(_GATE_FLAGS) != len(_GATE_EVALUATED) else ""))
w(f"- **{len(R['excluded_inputs']['window_length_provenance'])} window-length "
  f"provenance columns** (found 2026-08-24, not covered by the exit gate): "
  f"`{'`, `'.join(R['excluded_inputs']['window_length_provenance'])}`. Human is "
  f"pinned to a single value in all six; `N_Scored != 400` holds on 19,650 rows "
  f"(8.78%) with **Human 0**. The `provenance_only` diagnostic below measures "
  f"what that exclusion was worth.\n")

for scheme in ("random", "lodo", "logo"):
    rows = [c for c in R["cells"] if c["scheme"] == scheme]
    if not rows:
        continue
    w(f"\n## {scheme.upper()} — {'in-distribution' if scheme=='random' else ('leave-one-domain-out' if scheme=='lodo' else 'leave-one-generator-out')}\n")
    for cfg in CFG_LABEL:
        sub = [c for c in rows if c["config"] == cfg]
        if not sub:
            continue
        w(f"\n### Config {CFG_LABEL[cfg]}\n")
        w("| fold | arm | learner | n feat | test n | human % | TPR@1%FPR | TPR@5%FPR | AUROC | AP_human | macro-F1@0.5 | ECE |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for c in sub:
            w(f"| {c['fold']} | `{c['arm']}` | {c['learner']} | {c['n_features']} | "
              f"{c['n']:,} | {100*c['prevalence_human']:.2f} | "
              f"{cell(c,'tpr_at_1pct_fpr')} | {cell(c,'tpr_at_5pct_fpr')} | "
              f"{cell(c,'auroc')} | {cell(c,'ap_human')} | "
              f"{cell(c,'macro_f1_at_0.5_calibrated')} | "
              f"{c['point']['ece_calibrated']:.4f} |")

# ---- controls first: if the harness leaks, nothing below is readable --------
if R.get("controls"):
    w("\n## Harness control — label shuffle (read this before any result)\n")
    w("Ten independent permutations of the **training** labels only, evaluated "
      "against true test labels on **base-model scores with no calibrator** (a "
      "sigmoid with a fitted sign would orient the null model's random "
      "correlation positively every time, and the control could not fail).\n")
    w("**The criterion is one-sided.** Measured on this corpus a label-shuffled "
      "model lands systematically *below* chance — human rows occupy a sparse, "
      "extreme feature region (23.08% have `N_Words_Used` outside the entire LLM "
      "range) and a class-weighted booster on random labels gives that region its "
      "global prior. Leakage inflates performance toward 1.0; it cannot make a "
      "model worse than chance. FAIL iff `mean − 1.96·SEM > 0.5` or ≥9/10 "
      "shuffles exceed 0.5.\n")
    w("| config | scheme | fold | mean AUROC | sd | range | above 0.5 | verdict |")
    w("|---|---|---|---|---|---|---|---|")
    for c in R["controls"]:
        a = c["aucs"]
        w(f"| {c['config']} | {c['scheme']} | {c['fold']} | {c['mean']:.4f} | "
          f"{c['sd']:.4f} | [{min(a):.4f}, {max(a):.4f}] | "
          f"{c['n_above_0.5']}/{c['n_shuffles']} | **{c['verdict']}** |")
    inj = next((c["leak_injection"] for c in R["controls"] if "leak_injection" in c), None)
    if inj:
        w("\n**Positive control for the control** — leakage injected by adding a "
          "fraction of the test rows to training with their TRUE labels. If the "
          "shuffled-label AUROC does not rise above 0.5, this control cannot "
          "detect what it exists to detect.\n")
        w("| leaked fraction of test rows | rows | shuffled-label AUROC |")
        w("|---|---|---|")
        for d in inj:
            w(f"| {d['leak_fraction']:.0%} | {d['n_leaked_rows']:,} | "
              f"{d['auroc']:.4f}{' — **above 0.5, leak detected**' if d['auroc']>0.5 else ''} |")
        w("\n**Detection floor, stated:** a 2% or 5% leak still reads below 0.5 and "
          "**would be missed**. This control catches gross leakage only. The exact "
          "instrument for train/test contamination is the per-fold "
          "`assert_disjoint()` group check, which fires on a single shared group.\n")

# ---- Part B: prevalence disclosure -----------------------------------------
pd_ = R.get("prevalence_disclosure")
if pd_:
    w("\n## Calibration prevalence disclosure (Part B, amending D-11)\n")
    w(f"Corpus human prevalence: **{100*pd_['corpus_human_prevalence']:.3f}%**.\n")
    w(pd_["note"] + "\n")
    w("| config | scheme | fold | train human % | val human % | test human % |")
    w("|---|---|---|---|---|---|")
    for r in pd_["per_fold"]:
        w(f"| {r['config']} | {r['scheme']} | {r['fold']} | {100*r['train']:.3f} | "
          f"{100*r['val']:.3f} | {100*r['test']:.3f} |")

w("\n## `All` vs `All − fractal` — the P3 exit-gate comparison\n")
w("DeLong is the analytic paired test on AUROC; the paired bootstrap covers "
  "TPR@1%FPR, which DeLong does not. They are reported together, and whether "
  "they agree is reported rather than resolved by preferring one.\n")
w("| config | scheme | fold | AUROC all | AUROC all−fractal | DeLong z | DeLong p | ΔTPR@1%FPR [95% CI] | agree? |")
w("|---|---|---|---|---|---|---|---|---|")
for p in [x for x in R["paired"] if x.get("kind","").startswith("within_config")]:
    w(f"| {p['config']} | {p['scheme']} | {p['fold']} | {p['auroc_all']:.4f} | "
      f"{p['auroc_all_minus_fractal']:.4f} | {p['delong_z']:+.3f} | "
      f"{p['delong_p']:.3g} | {p['tpr1_diff']:+.4f} "
      f"[{p['tpr1_diff_lo']:+.4f}, {p['tpr1_diff_hi']:+.4f}] | "
      f"{'yes' if p['delong_and_paired_bootstrap_agree_on_significance'] else '**NO**'} |")

cross = [x for x in R["paired"] if x.get("kind") == "cross_config__vs_full"]
if cross:
    w("\n## Cross-configuration differences — every one with an interval\n")
    w("Each row is config *c* minus config 1 (`full`), **paired on the "
      "intersection of the two test sets**. Pairing matters: config 2 removes "
      "21,357 rows that are 100% LLM and configs 4/5 remove incomplete rows, so "
      "comparing metrics computed on different row sets would absorb the "
      "population change into the 'effect'. A difference is real only if its CI "
      "excludes zero — **'config c ≈ config 1' means the paired CI includes "
      "zero, never 'the point estimates looked close'**.\n")
    w("| config | scheme | fold | arm | shared n | % of full test | ΔAUROC [95% CI] | ΔTPR@1%FPR [95% CI] | ΔAP_human [95% CI] |")
    w("|---|---|---|---|---|---|---|---|---|")
    for x in cross:
        def d(k):
            mark = " **✦**" if x[f"d_{k}_excludes_zero"] else ""
            return f"{x[f'd_{k}']:+.5f} [{x[f'd_{k}_lo']:+.5f}, {x[f'd_{k}_hi']:+.5f}]{mark}"
        w(f"| {x['config']} | {x['scheme']} | {x['fold']} | `{x['arm']}` | "
          f"{x['n_shared_test_rows']:,} | {100*x['frac_of_full_test']:.1f}% | "
          f"{d('auroc')} | {d('tpr_at_1pct_fpr')} | {d('ap_human')} |")
    w("\n**✦** marks a CI that excludes zero.\n")

if R.get("leak_checks"):
    w("\n## Config 3 — did the imputation ablate the missingness signal, or relocate it?\n")
    w("`was_missing` AUROC is predicting *did this row have any NaN* from the "
      "**imputed** matrix. Reading `≈ 0.5 ⇒ success` would be wrong: missingness "
      "is mechanistically caused by the data, so it is partly predictable from "
      "never-missing columns no matter what is imputed. The **floor** column is "
      "that mechanism alone. The **median** column is a positive control that is "
      "*expected* to fail — it is what proves this check can fail.\n")
    w("| scheme | fold | floor (never-missing cols) | random draw | median control | excess over floor |")
    w("|---|---|---|---|---|---|")
    for lc in R["leak_checks"]:
        if "skipped" in lc:
            w(f"| {lc['scheme']} | {lc['fold']} | — | — | — | skipped: {lc['skipped']} |")
            continue
        w(f"| {lc['scheme']} | {lc['fold']} | "
          f"{lc['auroc_floor_never_missing_cols']:.4f} | "
          f"{lc['auroc_random_draw_imputed']:.4f} | "
          f"{lc['auroc_median_imputed_positive_control']:.4f} | "
          f"{lc['excess_over_floor']:+.4f} |")
    w("\nAnd the confound-free version — complete rows only, half masked-and-imputed, "
      "can a model tell the halves apart? Both halves are the same population, so "
      "0.5 means an imputed value is indistinguishable from a real one.\n")
    w("| scheme | fold | random draw | median control | n test |")
    w("|---|---|---|---|---|")
    for lc in R["leak_checks"]:
        idt = lc.get("imputation_detectability", {})
        if "auroc_random_draw" not in idt:
            continue
        w(f"| {lc['scheme']} | {lc['fold']} | {idt['auroc_random_draw']:.4f} | "
          f"{idt['auroc_median_positive_control']:.4f} | {idt['n_test']:,} |")

imp = R.get("importance", {})
if imp.get("lgbm_gain"):
    gain = imp["lgbm_gain"]
    split = imp.get("lgbm_split", {})
    perm = imp.get("permutation_auroc_drop", {})
    perms = imp.get("permutation_auroc_drop_std", {})
    tot = sum(gain.values()) or 1.0
    groups = R["feature_groups"]
    fam = {}
    for g, cols in groups.items():
        for c in cols:
            fam[c] = g
    w("\n## Feature importance — ALL declared features, failures included (D-13)\n")
    w(f"LightGBM `all` arm, config 1, random split, {len(gain)} inputs. Sorted by "
      f"gain. **Every declared feature appears, including the zero-gain ones** — "
      f"reporting only what worked is the selective reporting D-13 exists to "
      f"prevent. Permutation importance is the AUROC drop when that column is "
      f"shuffled on the test fold (5 repeats), an implementation-independent "
      f"cross-check on LightGBM's own gain accounting.\n")
    zero = [k for k, v in gain.items() if v == 0]
    w(f"**{len(zero)} of {len(gain)} features have exactly zero gain** — the "
      f"model never split on them once.\n")
    w("| rank | feature | family | gain | gain % | splits | perm. AUROC drop |")
    w("|---|---|---|---|---|---|---|")
    for i, (k, v) in enumerate(sorted(gain.items(), key=lambda kv: -kv[1]), 1):
        pv = perm.get(k)
        ps = perms.get(k)
        pstr = f"{pv:+.5f} ± {ps:.5f}" if pv is not None else "—"
        w(f"| {i} | `{k}` | {fam.get(k,'?')} | {v:,.1f} | {100*v/tot:.2f}% | "
          f"{split.get(k,0):,} | {pstr} |")

    w("\n### Gain by family\n")
    w("| family | n features | total gain | share | n with zero gain |")
    w("|---|---|---|---|---|")
    for g, cols in groups.items():
        s = sum(gain.get(c, 0.0) for c in cols)
        z = sum(1 for c in cols if gain.get(c, 0.0) == 0)
        w(f"| {g} | {len(cols)} | {s:,.1f} | {100*s/tot:.2f}% | {z} |")

open(OUT, "w").write("\n".join(L) + "\n")
print(f"[written] {OUT}  ({len(L)} lines)")
