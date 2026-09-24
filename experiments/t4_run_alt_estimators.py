#!/usr/bin/env python3
"""
T4 -- apply DFA and local-Whittle to the SAME interleaved-split-half
reliability test as T0/T1, on the real GAGLE corpus, ALSO computing R/S H
(fractal_math.calc_H, the paper's own estimator) on the identical halves so
all three are directly comparable per document.

v2, rewritten after internal review of the v1 result (rho=0.63-0.68 for
DFA/Whittle, clearing the 0.60 bar that R/S's 0.576/0.546 misses). The
the review objection: reliability = Var(delta)/(Var(delta)+Var(eta)), and the
v1 fBm validation showed Whittle/DFA are NOISIER than R/S at N=200 (larger
Var(eta)), so a HIGHER reliability requires an even larger Var(delta) --
whose content is unverified. Three concrete risks, each checked here before
the finding is trusted:

  1. Local Whittle's `minimize_scalar(bounds=(-0.49,1.49))` censors. Two
     halves that both saturate at the same bound are a PERFECTLY correlated
     pair by construction, not by any real document-level signal. This run
     records a saturation flag per half (H within 0.02 of either bound) and
     reports reliability BOTH with and without saturated pairs.
  2. Var(delta)/Var(eta) are decomposed for DFA_H/Whittle_H the same way
     T0 Route 2 did for R/S H (closed-form one-way ANOVA + r_within split),
     so the claim can be localized rather than left as a bare ratio.
  3. Per-document R/S-H vs DFA-H vs Whittle-H correlation (pooled and
     within-stratum) is reported: if the "more reliable" estimators barely
     correlate with R/S's own H, "the estimand is recoverable" is not quite
     the right sentence -- a DIFFERENT estimand is recoverable, and that
     needs to be named explicitly.

Saves the full per-document long-format table (not just aggregate stats)
specifically so these diagnostics can be run without re-scoring the corpus.
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.reliability_lib import load_corpus, split_interleave, SCORERS, pooled_and_within_stats, fisher_z_ci  # noqa: E402
from experiments.t4_estimators import dfa, local_whittle  # noqa: E402

from fractalsig import paths as _paths  # noqa: E402
from fractalsig.runtime import default_workers  # noqa: E402
from fractalsig.features.fractal_math import generate_trajectory, calc_H  # noqa: E402

OUT_DIR = _paths.STATS_DIR_STR
DFA_SCALE = [8, 16, 32, 48, 64, 96]
WHITTLE_BOUND_LO, WHITTLE_BOUND_HI = -0.49, 1.49
SATURATION_MARGIN = 0.02

_G = {}


def _is_saturated(h):
    return (h == h) and (h <= WHITTLE_BOUND_LO + 0.5 + SATURATION_MARGIN or
                          h >= WHITTLE_BOUND_HI + 0.5 - SATURATION_MARGIN)
    # h returned is d+0.5; d bounds (-0.49,1.49) -> h bounds (0.01,1.99)


def _process_range(args):
    start, end = args
    recs = _G["recs"]
    rows = []
    for i in range(start, end):
        for sname, _ in SCORERS:
            arr = recs[sname][i]
            if arr is None or len(arr) < 8:
                continue
            arr = np.asarray(arr, dtype=np.float64)
            a, b = split_interleave(arr)
            if len(a) < 8 or len(b) < 8:
                continue
            try:
                _, incr_a = generate_trajectory(a)
                _, incr_b = generate_trajectory(b)
                rs_a, _ = calc_H(incr_a, scale_param=DFA_SCALE)
                rs_b, _ = calc_H(incr_b, scale_param=DFA_SCALE)
                dfa_a, dfa_r2_a, _ = dfa(incr_a, DFA_SCALE)
                dfa_b, dfa_r2_b, _ = dfa(incr_b, DFA_SCALE)
                wh_a, m_a = local_whittle(incr_a)
                wh_b, m_b = local_whittle(incr_b)
            except Exception:
                continue
            rows.append((recs["id"][i], sname, recs["Label"][i], recs["domain"][i], recs["temp"][i],
                         rs_a, rs_b, dfa_a, dfa_b, wh_a, wh_b,
                         _is_saturated(wh_a), _is_saturated(wh_b)))
    return rows


def main():
    t0 = time.time()
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else default_workers()
    _G["recs"] = load_corpus()
    n_docs = len(_G["recs"]["id"])

    chunk = 2000
    ranges = [(s, min(s + chunk, n_docs)) for s in range(0, n_docs, chunk)]
    all_rows = []
    with mp.get_context("fork").Pool(processes=workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(_process_range, ranges), 1):
            all_rows.extend(rows)
            if i % 20 == 0:
                print(f"[t4] {i}/{len(ranges)} chunks, {len(all_rows):,} rows, "
                      f"{time.time()-t0:.0f}s", flush=True)
    print(f"[t4] feature computation done: {len(all_rows):,} rows in {time.time()-t0:.0f}s", flush=True)

    # ---- save full long-format table for downstream diagnostics ----
    cols = {"id": [], "scorer": [], "Label": [], "domain": [], "temp": [],
            "RS_H_a": [], "RS_H_b": [], "DFA_H_a": [], "DFA_H_b": [],
            "Whittle_H_a": [], "Whittle_H_b": [], "Whittle_sat_a": [], "Whittle_sat_b": []}
    for r in all_rows:
        (doc_id, sname, label, domain, temp, rs_a, rs_b, dfa_a, dfa_b, wh_a, wh_b, sat_a, sat_b) = r
        cols["id"].append(doc_id); cols["scorer"].append(sname)
        cols["Label"].append(label); cols["domain"].append(domain); cols["temp"].append(temp)
        cols["RS_H_a"].append(rs_a); cols["RS_H_b"].append(rs_b)
        cols["DFA_H_a"].append(dfa_a); cols["DFA_H_b"].append(dfa_b)
        cols["Whittle_H_a"].append(wh_a); cols["Whittle_H_b"].append(wh_b)
        cols["Whittle_sat_a"].append(bool(sat_a)); cols["Whittle_sat_b"].append(bool(sat_b))
    pq.write_table(pa.table(cols), os.path.join(OUT_DIR, "t4_alt_estimators_longformat.parquet"))
    print(f"[t4] wrote long-format parquet, {len(cols['id']):,} rows", flush=True)

    def reliability(a_key, b_key, sname, exclude_saturated=False):
        idx = [i for i, s in enumerate(cols["scorer"]) if s == sname]
        if exclude_saturated:
            idx = [i for i in idx if not (cols.get("Whittle_sat_a", [False]*len(cols["id"]))[i] or
                                             cols.get("Whittle_sat_b", [False]*len(cols["id"]))[i])]
        a = [cols[a_key][i] for i in idx]
        b = [cols[b_key][i] for i in idx]
        strata = [(cols["Label"][i], cols["domain"][i], cols["temp"][i]) for i in idx]
        mask = [x == x and y == y for x, y in zip(a, b)]
        a = [x for x, m in zip(a, mask) if m]
        b = [x for x, m in zip(b, mask) if m]
        strata = [s for s, m in zip(strata, mask) if m]
        return pooled_and_within_stats(a, b, strata)

    out = {"estimators": ["RS (calc_H, DFA_SCALE)", "DFA (order-1)", "local Whittle"],
           "dfa_scale": DFA_SCALE, "n_docs_loaded": n_docs, "reliability": {},
           "saturation": {}, "cross_estimator_correlation": {}}

    for sname in ("gemma", "mistral"):
        sat_a = [cols["Whittle_sat_a"][i] for i, s in enumerate(cols["scorer"]) if s == sname]
        sat_b = [cols["Whittle_sat_b"][i] for i, s in enumerate(cols["scorer"]) if s == sname]
        n_total = len(sat_a)
        n_either_sat = sum(1 for x, y in zip(sat_a, sat_b) if x or y)
        out["saturation"][sname] = {"n_total": n_total, "n_either_half_saturated": n_either_sat,
                                      "frac_either_saturated": n_either_sat / n_total if n_total else float("nan")}
        print(f"[t4] {sname}: Whittle saturation (either half within {SATURATION_MARGIN} of "
              f"bound [{WHITTLE_BOUND_LO+0.5},{WHITTLE_BOUND_HI+0.5}]): "
              f"{n_either_sat}/{n_total} = {n_either_sat/n_total:.4f}", flush=True)

        for est, (ak, bk) in (("RS_H", ("RS_H_a", "RS_H_b")), ("DFA_H", ("DFA_H_a", "DFA_H_b")),
                                ("Whittle_H", ("Whittle_H_a", "Whittle_H_b"))):
            key = f"{sname}_{est}"
            s_all = reliability(ak, bk, sname, exclude_saturated=False)
            out["reliability"][key] = s_all
            print(f"[t4] {key} (all pairs): n={s_all['n']:,} rho_pooled={s_all['rho_pooled']:.4f} "
                  f"rho_within={s_all['rho_within']:.4f}  passes>=0.60={s_all['passes_pooled']}", flush=True)
            if est == "Whittle_H":
                s_excl = reliability(ak, bk, sname, exclude_saturated=True)
                out["reliability"][f"{key}_excl_saturated"] = s_excl
                print(f"[t4] {key} EXCLUDING saturated pairs: n={s_excl['n']:,} "
                      f"rho_pooled={s_excl['rho_pooled']:.4f} rho_within={s_excl['rho_within']:.4f}  "
                      f"passes>=0.60={s_excl['passes_pooled']}", flush=True)

        # ---- cross-estimator correlation (pooled value = mean of the two halves) ----
        idx = [i for i, s in enumerate(cols["scorer"]) if s == sname]
        rs_doc = np.array([(cols["RS_H_a"][i] + cols["RS_H_b"][i]) / 2 for i in idx])
        dfa_doc = np.array([(cols["DFA_H_a"][i] + cols["DFA_H_b"][i]) / 2 for i in idx])
        wh_doc = np.array([(cols["Whittle_H_a"][i] + cols["Whittle_H_b"][i]) / 2 for i in idx])
        valid = ~(np.isnan(rs_doc) | np.isnan(dfa_doc) | np.isnan(wh_doc))
        rs_doc, dfa_doc, wh_doc = rs_doc[valid], dfa_doc[valid], wh_doc[valid]
        cc = {
            "n": int(valid.sum()),
            "corr_RS_DFA": float(np.corrcoef(rs_doc, dfa_doc)[0, 1]),
            "corr_RS_Whittle": float(np.corrcoef(rs_doc, wh_doc)[0, 1]),
            "corr_DFA_Whittle": float(np.corrcoef(dfa_doc, wh_doc)[0, 1]),
        }
        out["cross_estimator_correlation"][sname] = cc
        print(f"[t4] {sname} cross-estimator corr (doc-level mean-of-halves): "
              f"RS~DFA={cc['corr_RS_DFA']:.4f}  RS~Whittle={cc['corr_RS_Whittle']:.4f}  "
              f"DFA~Whittle={cc['corr_DFA_Whittle']:.4f}", flush=True)

    out_path = os.path.join(OUT_DIR, "t4_alt_estimators_reliability.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n[written] {out_path}", flush=True)
    print(f"[t4] TOTAL {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
