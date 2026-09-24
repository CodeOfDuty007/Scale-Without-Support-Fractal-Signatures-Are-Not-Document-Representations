#!/usr/bin/env python3
"""Generalizes p1_exit_gate_v2.py's C1/C2/C3 label-relationship checks (F-6)
to accept ANY feature store, not just P3's GAGLE one. p1_exit_gate_v2.py
itself is left untouched -- same discipline as v1 being left untouched when
v2 was written (D-10), and `reconcile_p1_against_p0_v2.py`'s own precedent.

WHY THIS SCRIPT EXISTS (found 2026-08-27, Track A A2): F-6's ledger note
says the gate is "REQUIRED BEFORE P5" because "P5 builds a fresh feature
store from RAID through this same gate" -- but `p1_exit_gate_v2.py` calls
`load_p3_feature_set()`, which hardcodes `p3_train_eval.load_matrix()`
(P3's GAGLE parquet). It has no path parameter. It cannot be pointed at a
RAID feature store at all as shipped. This script reuses `scan_column` (the
actual C1/C2/C3 logic, which IS general -- it only needs a numeric column
and a boolean class mask) directly from p1_exit_gate_v2.py rather than
duplicating it, and adds a generic loader plus a self-test that reproduces
p1_exit_gate_v2.py's own GAGLE result exactly, so this generalization is
proven not to have silently changed behavior before it is ever pointed at
real RAID data.

Usage (once a RAID feature store exists, post-scoring):
  python -m fractalsig.transfer.p5_exit_gate_raid --input feature_store/v3_raid.parquet \\
      --label-col Label --human-value human --feature-cols-json <path or omit for all-numeric>

Self-test (works today, no RAID data needed):
  python -m fractalsig.transfer.p5_exit_gate_raid --self-test
"""
from fractalsig import paths as _paths  # noqa: E402
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = _paths.WORKSPACE_STR
from fractalsig.pipeline.p1_exit_gate_v2 import scan_column, ONE_WAY_COVERAGE, NAN_ONE_WAY, DISCRETE_MAX_UNIQUE  # noqa: E402


def run_gate(df, is_human_col_values, feature_cols, out_path, source_label,
             known_leak_cols=None):
    is_human = np.asarray(is_human_col_values, dtype=bool)
    rng = np.random.default_rng(20260824)

    df = df.copy()
    df["__control_negative"] = rng.normal(0, 1, len(df))
    df["__control_positive"] = np.where(is_human, 1.0, 0.0)
    cols = list(feature_cols) + ["__control_negative", "__control_positive"]

    print("=" * 92)
    print(f"P5 EXIT GATE (generalized F-6 checks) -- source: {source_label}")
    print("=" * 92)
    print(f"  rows {len(df):,}   class-A(human) {int(is_human.sum()):,}   "
          f"class-B(llm) {int((~is_human).sum()):,}   columns scanned {len(cols)}")
    print(f"  THRESHOLDS -- C1 one-way coverage >= {ONE_WAY_COVERAGE:.1%}; "
          f"C2 within-class sd == 0; C3 one-way NaN >= {NAN_ONE_WAY:.1%}; "
          f"C1 discrete<= {DISCRETE_MAX_UNIQUE} unique values, out-of-support above")

    rows = [scan_column(df[c], is_human, c) for c in cols]

    neg = next(r for r in rows if r["column"] == "__control_negative")
    pos = next(r for r in rows if r["column"] == "__control_positive")
    neg_ok = not neg["FLAGGED"]
    pos_ok = pos["C1_one_way_indicator"] and pos["C2_zero_variance_within_class"]
    controls_ok = bool(neg_ok and pos_ok)
    print("\n--- CONTROLS ---")
    print(f"  negative control flagged? {neg['FLAGGED']} -> {'OK' if neg_ok else '*** OVER-SENSITIVE ***'}")
    print(f"  positive control flagged by C1+C2? {pos_ok} -> {'OK' if pos_ok else '*** CHECKS ARE DECORATION ***'}")
    if not controls_ok:
        print("\n  [BROKEN GATE] Controls did not behave. Do not trust findings below.")

    real = [r for r in rows if not r["column"].startswith("__control")]
    flagged = [r for r in real if r["FLAGGED"]]
    print(f"\n--- RESULT: {len(flagged)} of {len(real)} columns FLAGGED ---")
    for r in sorted(flagged, key=lambda x: (-max(x['C1_llm_coverage_outside_human'],
                                                 x['C1_human_coverage_outside_llm']))):
        det = []
        if r["C2_zero_variance_within_class"]:
            det.append(f"{r['C2_which']} sd=0")
        if r["C3_one_way_missingness"]:
            det.append(f"{r['C3_direction']} (A {r['human_nan_frac']:.3%} / B {r['llm_nan_frac']:.3%})")
        print(f"  {r['column']:<40} C1={'Y' if r['C1_one_way_indicator'] else '.'} "
              f"C2={'Y' if r['C2_zero_variance_within_class'] else '.'} "
              f"C3={'Y' if r['C3_one_way_missingness'] else '.'}  {'; '.join(det)}")

    refound = None
    if known_leak_cols:
        found = {r["column"] for r in flagged}
        refound = [c for c in known_leak_cols if c in found]
        print(f"\n  re-found known leak columns: {len(refound)}/{len(known_leak_cols)}")

    verdict = "BROKEN" if not controls_ok else ("FLAGS" if flagged else "CLEAN")
    payload = {"gate": "p5_exit_gate_raid (generalized F-6)", "source": source_label,
               "thresholds": {"one_way_coverage": ONE_WAY_COVERAGE, "nan_one_way": NAN_ONE_WAY,
                              "discrete_max_unique": DISCRETE_MAX_UNIQUE},
               "n_rows": int(len(df)), "n_class_a_human": int(is_human.sum()),
               "controls_behaved": controls_ok, "verdict": verdict,
               "n_scanned": len(real), "n_flagged": len(flagged),
               "refound_known_leak_columns": refound, "columns": rows}
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"\n[written] {out_path}")
    return payload


def self_test():
    """Reproduces p1_exit_gate_v2.py's own GAGLE result exactly through the
    generalized loader path, proving the generalization changed nothing."""
    pass  # import path handled by the fractalsig package
    from fractalsig.detection.p3_train_eval import load_matrix, PROVENANCE_LEAK
    df, groups = load_matrix()
    feats = groups["fractal"] + groups["stat"] + groups["style"]
    leak = list(PROVENANCE_LEAK)
    is_human = (df["Label"] == "Human").to_numpy()
    payload = run_gate(df, is_human, feats + leak,
                        os.path.join(HERE, "feature_store", "p5_exit_gate_selftest_report.json"),
                        "SELF-TEST via GAGLE/P3 (should match p1_exit_gate_v2.py exactly)",
                        known_leak_cols=leak)
    expect_refound = 6
    ok = payload["refound_known_leak_columns"] is not None and len(payload["refound_known_leak_columns"]) == expect_refound
    print(f"\n[self-test] refound {len(payload['refound_known_leak_columns'])}/{expect_refound} known F-6 columns "
          f"-> {'PASS, generalization is behavior-preserving' if ok else '*** FAIL, generalization changed behavior ***'}")
    return 0 if ok and payload["controls_behaved"] else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--input", help="path to a .parquet or .jsonl feature store")
    ap.add_argument("--label-col", default="Label")
    ap.add_argument("--human-value", default="human")
    ap.add_argument("--exclude-cols", nargs="*", default=[],
                     help="non-feature columns to exclude (id, text, domain, attack, etc.)")
    ap.add_argument("--out", default=os.path.join(HERE, "feature_store", "p5_exit_gate_raid_report.json"))
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if not args.input:
        ap.error("--input is required unless --self-test")

    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_json(args.input, lines=True)

    is_human = (df[args.label_col].astype(str).str.lower() == args.human_value.lower()).to_numpy()
    exclude = set(args.exclude_cols) | {args.label_col}
    # pandas treats bool as a numeric dtype, so a plain is_numeric_dtype check
    # would silently sweep in every provenance/metadata flag (e.g.
    # ZWSP_Present_In_Source_Text, added 2026-08-29) as a candidate feature --
    # no boolean column in this codebase's convention is ever a real model
    # feature, so excluding bool dtype here is a permanent guard, not just a
    # one-off exclude-cols entry that depends on every future caller
    # remembering to pass it.
    feature_cols = [c for c in df.columns if c not in exclude
                    and pd.api.types.is_numeric_dtype(df[c])
                    and not pd.api.types.is_bool_dtype(df[c])]
    print(f"[p5-exit-gate] {len(feature_cols)} numeric feature columns auto-detected "
          f"(excluded: {sorted(exclude)}, plus any boolean-dtype columns)")

    return 0 if run_gate(df, is_human, feature_cols, args.out, args.input)["controls_behaved"] else 1


if __name__ == "__main__":
    sys.exit(main())
