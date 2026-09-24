#!/usr/bin/env python3
"""
P1 EXIT GATE v2 -- adds the two checks whose ABSENCE is F-6.

v1 (`p1_exit_gate.py`) is left untouched, per D-10 and the precedent set by
`reconcile_p1_against_p0_v2.py`. v1 is not wrong about what it checked; it is
incomplete about what it checked FOR.

WHAT F-6 IS
  v1 asked, of every feature column: are its NaNs explained, is it non-constant,
  does it have variance, does it carry inf. All four are properties of the column
  IN ISOLATION. None of them looks at the column's relationship to the LABEL.

  So v1 passed `Gemma_Feature_Window_N_Scored` as a legitimate feature. That
  column is 400.0 on 5,000/5,000 human rows with standard deviation exactly 0,
  while LLM rows range 319..462. `N_Scored != 400` is therefore a 100%-precision
  "this row is LLM" rule covering 19,650 rows (8.78% of the corpus). It is
  provenance metadata about our own preprocessing, and it was available to P2 and
  P3 as a model input.

  Five more columns share the shape (`Mistral_Feature_Window_N_Scored`, the four
  `*_window_n_windows`). The defect is the gate's DESIGN, not the six instances.

  This matters beyond P3: P5 builds a fresh feature store from RAID through the
  same gate. Without these checks the identical class of defect appears on a
  dataset nobody has audited, where it is much harder to notice.

THE TWO NEW CHECKS
  C1  ONE-WAY CLASS INDICATOR. Does one class occupy a value, value-set, or
      range that the other class never occupies, covering more than
      ONE_WAY_COVERAGE of that class's rows?
  C2  ZERO-VARIANCE-WITHIN-CLASS. Does either class have standard deviation
      exactly 0 on a column that is not globally constant?
  C3  ONE-WAY MISSINGNESS (added because it is the same defect wearing a
      different hat -- see F-5 and the Gemma Family-B block). Is one class NaN
      on more than NAN_ONE_WAY of its rows while the other class is NEVER NaN?

CAN THESE CHECKS FAIL?
  Yes, and `--controls` proves it rather than asserting it. Two synthetic
  columns are injected:
      NEGATIVE control -- drawn from one distribution for BOTH classes. MUST NOT
                          be flagged. If it is, the checks are over-sensitive and
                          every flag they raise is suspect.
      POSITIVE control -- a literal class indicator. MUST be flagged by C1 and
                          C2. If it is not, the checks are decoration.
  A run in which the negative control is flagged, or the positive control is not,
  is reported as a BROKEN GATE, not as a result.
"""
from fractalsig import paths as _paths  # noqa: E402
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = _paths.WORKSPACE_STR

# ---------------------------------------------------------------------------
# Thresholds. Stated, not implicit -- a referee (and the next session) needs to
# know what "flagged" means numerically.
# ---------------------------------------------------------------------------
ONE_WAY_COVERAGE = 0.01     # >=1% of a class in a region the other never occupies
NAN_ONE_WAY = 0.005         # >=0.5% of a class NaN while the other is never NaN
DISCRETE_MAX_UNIQUE = 50    # at or below this, compare value SETS, not ranges

HUMAN = "Human"


def scan_column(vals, is_human, col):
    """One column, one label split. Returns the evidence, and the verdicts."""
    v = pd.to_numeric(vals, errors="coerce").to_numpy(float)
    h, l = v[is_human], v[~is_human]
    hn, ln = h[~np.isnan(h)], l[~np.isnan(l)]

    out = {"column": col,
           "n_human": int(is_human.sum()), "n_llm": int((~is_human).sum()),
           "human_n_present": int(len(hn)), "llm_n_present": int(len(ln)),
           "human_sd": (float(np.std(hn)) if len(hn) else None),
           "llm_sd": (float(np.std(ln)) if len(ln) else None),
           "human_nunique": int(len(np.unique(hn))) if len(hn) else 0,
           "llm_nunique": int(len(np.unique(ln))) if len(ln) else 0,
           "global_nunique": int(len(np.unique(v[~np.isnan(v)])))}

    # ---- C3: one-way missingness -----------------------------------------
    h_nan = float(np.isnan(h).mean()) if len(h) else 0.0
    l_nan = float(np.isnan(l).mean()) if len(l) else 0.0
    out["human_nan_frac"], out["llm_nan_frac"] = h_nan, l_nan
    out["C3_one_way_missingness"] = bool(
        (l_nan >= NAN_ONE_WAY and h_nan == 0.0) or
        (h_nan >= NAN_ONE_WAY and l_nan == 0.0))
    out["C3_direction"] = ("LLM-only NaN" if (l_nan > 0 and h_nan == 0)
                           else "Human-only NaN" if (h_nan > 0 and l_nan == 0)
                           else "both or neither")

    # ---- C2: zero variance within a class ---------------------------------
    out["C2_zero_variance_within_class"] = bool(
        out["global_nunique"] > 1 and
        ((out["human_sd"] == 0.0 and len(hn) > 1) or
         (out["llm_sd"] == 0.0 and len(ln) > 1)))
    out["C2_which"] = ("Human" if out["human_sd"] == 0.0 and len(hn) > 1
                       else "LLM" if out["llm_sd"] == 0.0 and len(ln) > 1 else None)

    # ---- C1: one-way class indicator --------------------------------------
    # Two regimes, because "a region the other class never occupies" means
    # different things for a 5-valued count and for a continuous exponent.
    cov_l, cov_h, mode = 0.0, 0.0, None
    if len(hn) and len(ln):
        if out["global_nunique"] <= DISCRETE_MAX_UNIQUE:
            mode = "value-set (discrete)"
            hs, ls = set(np.unique(hn).tolist()), set(np.unique(ln).tolist())
            cov_l = float(np.mean([x not in hs for x in ln]))
            cov_h = float(np.mean([x not in ls for x in hn]))
        else:
            mode = "out-of-support (continuous)"
            hlo, hhi = hn.min(), hn.max()
            llo, lhi = ln.min(), ln.max()
            cov_l = float(np.mean((ln < hlo) | (ln > hhi)))
            cov_h = float(np.mean((hn < llo) | (hn > lhi)))
    out["C1_mode"] = mode
    out["C1_llm_coverage_outside_human"] = cov_l
    out["C1_human_coverage_outside_llm"] = cov_h
    out["C1_one_way_indicator"] = bool(cov_l >= ONE_WAY_COVERAGE or
                                       cov_h >= ONE_WAY_COVERAGE)

    out["FLAGGED"] = bool(out["C1_one_way_indicator"] or
                          out["C2_zero_variance_within_class"] or
                          out["C3_one_way_missingness"])
    return out


def load_p3_feature_set():
    """The 85 columns P3 actually feeds a model, PLUS the six F-6 columns, so the
    gate is run over exactly what P3 uses and demonstrably re-finds the six."""
    pass  # import path handled by the fractalsig package
    from fractalsig.detection.p3_train_eval import load_matrix, PROVENANCE_LEAK
    df, groups = load_matrix()
    feats = groups["fractal"] + groups["stat"] + groups["style"]
    return df, feats, list(PROVENANCE_LEAK), groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "feature_store",
                                                  "p1_exit_gate_v2_report.json"))
    ap.add_argument("--controls", action="store_true", default=True)
    args = ap.parse_args()

    df, feats, leak, groups = load_p3_feature_set()
    is_human = (df["Label"] == HUMAN).to_numpy()
    fam = {c: g for g, cols in groups.items() for c in cols}
    for c in leak:
        fam[c] = "provenance (excluded by P3)"

    rng = np.random.default_rng(20260824)
    if args.controls:
        # NEGATIVE: same distribution for both classes -> must NOT be flagged.
        df["__control_negative"] = rng.normal(0, 1, len(df))
        # POSITIVE: a literal class indicator -> must be flagged by C1 and C2.
        df["__control_positive"] = np.where(is_human, 1.0, 0.0)
        fam["__control_negative"] = "CONTROL (negative)"
        fam["__control_positive"] = "CONTROL (positive)"

    cols = feats + leak + (["__control_negative", "__control_positive"]
                           if args.controls else [])

    print("=" * 92)
    print("P1 EXIT GATE v2 -- label-relationship checks (F-6)")
    print("=" * 92)
    print(f"  rows {len(df):,}   human {int(is_human.sum()):,}   "
          f"LLM {int((~is_human).sum()):,}   columns scanned {len(cols)}")
    print(f"  THRESHOLDS -- C1 one-way coverage >= {ONE_WAY_COVERAGE:.1%} of a class; "
          f"C2 within-class sd == 0; C3 one-way NaN >= {NAN_ONE_WAY:.1%}")
    print(f"  C1 compares VALUE SETS at <= {DISCRETE_MAX_UNIQUE} unique values, "
          f"OUT-OF-SUPPORT above it")

    rows = [scan_column(df[c], is_human, c) for c in cols]
    for r in rows:
        r["family"] = fam.get(r["column"], "?")

    # --- control verdict FIRST: if the checks are broken, nothing else counts
    neg = next((r for r in rows if r["column"] == "__control_negative"), None)
    pos = next((r for r in rows if r["column"] == "__control_positive"), None)
    controls_ok = None
    if neg is not None and pos is not None:
        neg_ok = not neg["FLAGGED"]
        pos_ok = pos["C1_one_way_indicator"] and pos["C2_zero_variance_within_class"]
        controls_ok = bool(neg_ok and pos_ok)
        print("\n--- CONTROLS (this is what makes the checks falsifiable) ---")
        print(f"  negative control (same distribution both classes) flagged? "
              f"{neg['FLAGGED']}  -> {'OK' if neg_ok else '*** OVER-SENSITIVE ***'}")
        print(f"  positive control (literal class indicator) flagged by C1 and C2? "
              f"{pos['C1_one_way_indicator'] and pos['C2_zero_variance_within_class']}"
              f"  -> {'OK' if pos_ok else '*** CHECKS ARE DECORATION ***'}")
        if not controls_ok:
            print("\n  [BROKEN GATE] The controls did not behave. Do not read the "
                  "findings below as results.")

    real = [r for r in rows if not r["column"].startswith("__control")]
    flagged = [r for r in real if r["FLAGGED"]]

    print(f"\n--- RESULT: {len(flagged)} of {len(real)} columns FLAGGED ---")
    print(f"{'column':<40}{'family':<26}{'C1':>4}{'C2':>4}{'C3':>4}"
          f"{'LLM out':>9}{'Hum out':>9}  detail")
    for r in sorted(flagged, key=lambda x: (-max(x['C1_llm_coverage_outside_human'],
                                                 x['C1_human_coverage_outside_llm']))):
        det = []
        if r["C2_zero_variance_within_class"]:
            det.append(f"{r['C2_which']} sd=0")
        if r["C3_one_way_missingness"]:
            det.append(f"{r['C3_direction']} "
                       f"(H {r['human_nan_frac']:.3%} / L {r['llm_nan_frac']:.3%})")
        print(f"{r['column']:<40}{r['family']:<26}"
              f"{'Y' if r['C1_one_way_indicator'] else '.':>4}"
              f"{'Y' if r['C2_zero_variance_within_class'] else '.':>4}"
              f"{'Y' if r['C3_one_way_missingness'] else '.':>4}"
              f"{r['C1_llm_coverage_outside_human']:>9.3%}"
              f"{r['C1_human_coverage_outside_llm']:>9.3%}  {'; '.join(det)}")

    # did it re-find the six F-6 columns?
    found = {r["column"] for r in flagged}
    refound = [c for c in leak if c in found]
    print(f"\n  re-found the six F-6 provenance columns: {len(refound)}/6"
          f"{'' if len(refound)==6 else '  *** MISSED: ' + str([c for c in leak if c not in found])}")
    print(f"  NEW columns flagged that F-6 had not already named: "
          f"{len([r for r in flagged if r['column'] not in leak])}")

    verdict = "BROKEN" if controls_ok is False else ("FLAGS" if flagged else "CLEAN")
    payload = {"gate": "p1_exit_gate_v2 (F-6)",
               "thresholds": {"one_way_coverage": ONE_WAY_COVERAGE,
                              "nan_one_way": NAN_ONE_WAY,
                              "discrete_max_unique": DISCRETE_MAX_UNIQUE},
               "n_rows": int(len(df)), "n_human": int(is_human.sum()),
               "controls_behaved": controls_ok,
               "verdict": verdict,
               "n_scanned": len(real), "n_flagged": len(flagged),
               "refound_f6_columns": refound,
               "columns": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"\n[written] {args.out}")
    return 0 if controls_ok is not False else 1


if __name__ == "__main__":
    sys.exit(main())
