#!/usr/bin/env python3
"""P4 Family D recompute, post-gate (2026-08-27, design decision after the
7-test validity gate). Supersedes p4_compute_family_d.py's output, which
predates the gate and is provisional.

Changes from v1:
  - `human_revision` EXCLUDED ENTIRELY (89.2% near-identity, cosine>0.98 --
    not a real edit for the large majority of the type; design decision:
    drop, not regenerate).
  - Defect-A filter: any edited variant with semantic_cosine < 0.5 is
    excluded from scoring (prompt-template echo / topic hallucination --
    invisible to below_min_tokens/is_empty/repetition, only cosine catches
    it).
  - Two SCOPES computed and reported side by side, per design decision
    (do not pre-emptively restrict to the common subset -- Test 3's own
    addendum found no major-attrition type has a dramatic attack-quality
    selection effect, so score both and let agreement/disagreement decide
    which is primary):
      "full"   -- every surviving, cosine-valid doc for that type, whatever
                  the population size (differs per type).
      "common" -- restricted to the 5-type common-subset id intersection
                  (paraphrase_light, paraphrase_heavy, style_transfer,
                  sentence_reorder, typo_noise -- human_revision dropped,
                  summarize_then_expand excluded from the intersection by
                  design, see below), n=2,355 (Test 4, recomputed here for
                  exact reproducibility rather than hardcoded).
  - `summarize_then_expand` is NEVER in the "common" scope (50% attrition /
    13.1% below the cosine floor -- worst on both axes; scored "full" only,
    with the scope limitation stated explicitly: no robustness claim is
    made for summarization attacks specifically, only descriptive Family D
    numbers for whatever survives).

Estimator methodology unchanged from v1 / reconcile_p1_against_p0.py's own
precedent: H and Delta-log-PPL per-document; S pooled per (variant, domain,
scorer) cell only (per-doc S unreliable, rho=0.05-0.06, P2 finding).
"""
from fractalsig import paths as _paths  # noqa: E402
import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = _paths.WORKSPACE_STR
from fractalsig.features.fractal_math import generate_trajectory, calc_H, calc_S  # noqa: E402

COMMON_SUBSET_TYPES = ["paraphrase_light", "paraphrase_heavy", "style_transfer",
                        "sentence_reorder", "typo_noise"]
SEPARATE_TYPES = ["summarize_then_expand"]  # scored "full" only, never in "common"
ALL_SCORED_TYPES = COMMON_SUBSET_TYPES + SEPARATE_TYPES  # human_revision excluded
SCORERS = ["gemma", "mistral"]
MIN_CELL_DOCS = 10
COSINE_FLOOR = 0.5


def load_rescore(batch, scorer):
    path = os.path.join(HERE, "data", f"p4_rescore_batch{batch}_{scorer}.jsonl")
    by_id_variant = {}
    for line in open(path):
        r = json.loads(line)
        by_id_variant[(r["id"], r["variant"])] = r
    return by_id_variant


def gen_survives_and_valid(v):
    """Generation-level survival AND passes the Defect-A cosine floor."""
    if v is None or v.get("below_min_tokens") or v.get("is_empty"):
        return False
    c = v.get("semantic_cosine")
    if c is not None and c < COSINE_FLOOR:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 2])
    args = ap.parse_args()

    docs_meta = {}
    edits_by_id = {}
    for batch in args.batches:
        for l in open(os.path.join(HERE, "data", f"p4_edits_batch{batch}.jsonl")):
            rec = json.loads(l)
            docs_meta[rec["id"]] = rec
            edits_by_id[rec["id"]] = rec

    # Common-subset id intersection: gen-survives-and-cosine-valid across all
    # 5 COMMON_SUBSET_TYPES (independent of scorer -- Test 4 verified 0
    # per-scorer disagreement among generation-survivors, both scorers).
    common_ids = None
    for t in COMMON_SUBSET_TYPES:
        ids_t = {doc_id for doc_id, rec in edits_by_id.items()
                  if gen_survives_and_valid(rec.get("variants", {}).get(t))}
        common_ids = ids_t if common_ids is None else (common_ids & ids_t)
    print(f"[family_d_v2] common-subset intersection (5 types, cosine-floor applied): "
          f"{len(common_ids)} docs", flush=True)

    for scope_name, scope_ids in [("full", None), ("common", common_ids)]:
        per_doc_rows = []
        cell_pool = collections.defaultdict(lambda: {"clean": [], "edited": [], "ids": []})

        for batch in args.batches:
            for scorer in SCORERS:
                rescore = load_rescore(batch, scorer)
                n_paired, n_excluded_cosine, n_excluded_scope = 0, 0, 0
                for doc_id, meta in docs_meta.items():
                    if scope_ids is not None and doc_id not in scope_ids:
                        n_excluded_scope += 1
                        continue
                    clean_key = (doc_id, "clean")
                    if clean_key not in rescore:
                        continue
                    clean_rec = rescore[clean_key]
                    if clean_rec.get("skipped_below_min_tokens") or not clean_rec["scores"]:
                        continue
                    clean_scores = clean_rec["scores"]
                    clean_H, _ = calc_H(generate_trajectory(clean_scores)[1])
                    clean_logppl = float(np.mean(clean_scores))
                    clean_traj = generate_trajectory(clean_scores)[0]

                    for vtype in ALL_SCORED_TYPES:
                        if scope_name == "common" and vtype in SEPARATE_TYPES:
                            continue  # summarize_then_expand never in "common"
                        v = meta.get("variants", {}).get(vtype)
                        if not gen_survives_and_valid(v):
                            if v is not None and v.get("semantic_cosine") is not None \
                                    and v["semantic_cosine"] < COSINE_FLOOR:
                                n_excluded_cosine += 1
                            continue
                        ed_key = (doc_id, vtype)
                        if ed_key not in rescore:
                            continue
                        ed_rec = rescore[ed_key]
                        if ed_rec.get("skipped_below_min_tokens") or not ed_rec["scores"]:
                            continue
                        ed_scores = ed_rec["scores"]
                        ed_H, _ = calc_H(generate_trajectory(ed_scores)[1])
                        ed_logppl = float(np.mean(ed_scores))
                        ed_traj = generate_trajectory(ed_scores)[0]

                        per_doc_rows.append({
                            "id": doc_id, "batch": batch, "domain": meta["domain"],
                            "generator": meta["generator"], "variant": vtype, "scorer": scorer,
                            "scope": scope_name,
                            "clean_H": float(clean_H), "edited_H": float(ed_H),
                            "delta_H": float(ed_H - clean_H),
                            "clean_logppl": clean_logppl, "edited_logppl": ed_logppl,
                            "delta_logppl": ed_logppl - clean_logppl,
                        })
                        n_paired += 1
                        cell_key = (vtype, meta["domain"], scorer)
                        cell_pool[cell_key]["clean"].append(clean_traj)
                        cell_pool[cell_key]["edited"].append(ed_traj)
                        cell_pool[cell_key]["ids"].append(doc_id)

                print(f"[family_d_v2] scope={scope_name} batch={batch} scorer={scorer}: "
                      f"{n_paired} paired rows, {n_excluded_cosine} excluded (cosine<{COSINE_FLOOR}), "
                      f"{n_excluded_scope} excluded (out of scope)", flush=True)

        per_doc_path = os.path.join(HERE, "data", f"p4_family_d_v2_{scope_name}_per_doc.jsonl")
        with open(per_doc_path, "w") as f:
            for row in per_doc_rows:
                f.write(json.dumps(row) + "\n")
        print(f"[family_d_v2] scope={scope_name}: wrote {len(per_doc_rows)} per-doc rows to {per_doc_path}",
              flush=True)

        cell_rows = []
        for (vtype, domain, scorer), pool in sorted(cell_pool.items()):
            n = len(pool["ids"])
            if n < MIN_CELL_DOCS:
                print(f"[family_d_v2] scope={scope_name} SKIPPED cell {vtype}/{domain}/{scorer}: n={n} < {MIN_CELL_DOCS}",
                      flush=True)
                continue
            S_clean, S_clean_r2 = calc_S(pool["clean"])
            S_edited, S_edited_r2 = calc_S(pool["edited"])
            cell_rows.append({
                "scope": scope_name, "variant": vtype, "domain": domain, "scorer": scorer, "n_docs": n,
                "S_clean": float(S_clean), "S_clean_r2": float(S_clean_r2),
                "S_edited": float(S_edited), "S_edited_r2": float(S_edited_r2),
                "delta_S": float(S_edited - S_clean),
            })

        cell_path = os.path.join(HERE, "data", f"p4_family_d_v2_{scope_name}_cells.jsonl")
        with open(cell_path, "w") as f:
            for row in cell_rows:
                f.write(json.dumps(row) + "\n")
        print(f"[family_d_v2] scope={scope_name}: wrote {len(cell_rows)} pooled-cell rows to {cell_path}",
              flush=True)


if __name__ == "__main__":
    main()
