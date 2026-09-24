#!/usr/bin/env python3
"""P4 task 4 (2026-08-27): Family D -- Delta H, Delta S, Delta log-PPL
between each edited variant and the clean snippet it was generated from.

Estimator choice matches the project's own established precedent exactly
(internal review, 2026-08-27) -- `reconcile_p1_against_p0.py`'s
`compute_pooled_SH` already solved "how do you compute H vs S from raw
score arrays" once, and this reuses that call shape rather than re-deriving
it:
  - H: per-document. calc_H(generate_trajectory(scores)[1]) (the INCREMENT
    array). Supported by P2's split-half reliability (rho=0.55-0.58 for H).
  - S: NEVER per-document. `fractal_math.py`'s own `get_allMetric` flags
    per-doc S as `Holder_PerDoc_Unreliable: True`, and P2 measured why:
    rho=0.05-0.06 per-document reliability, essentially noise with a
    domain-dependent mean. Differencing two such noisy estimates roughly
    doubles the noise variance against a signal already swamped -- but
    averaged over thousands of documents produces a tight-looking,
    spurious CI that would pass every check in this project's harness
    without being real. S is instead pooled per (variant, domain, scorer)
    CELL: gather every paired document's own z-scored trajectory
    (generate_trajectory(scores)[0]) into a list, run
    calc_S(list_of_trajectories) ONCE per cell (list-mode, exactly
    `compute_pooled_SH`'s pattern) for the clean side and the edited side
    separately, then difference the two cell-level S values. Reported
    per-cell, with n_docs, never per-row.
  - Delta log-PPL: per-document. mean(scores) is a first moment, not a
    scale-sweep fit -- doesn't inherit S's problem.

PAIRED RESTRICTION (internal review): a document only contributes to any
Delta if BOTH its clean score array and that specific edited variant's
score array survived the MIN_TOKENS window for that scorer (tokenizers
differ, so the surviving id set differs between Gemma and Mistral --
computed separately per scorer, not assumed shared).
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

ALL_TYPES = [
    "paraphrase_light", "paraphrase_heavy", "human_revision",
    "style_transfer", "summarize_then_expand", "sentence_reorder", "typo_noise",
]
SCORERS = ["gemma", "mistral"]
MIN_CELL_DOCS = 10  # matches compute_pooled_SH's own floor


def load_rescore(batch, scorer):
    path = os.path.join(HERE, "data", f"p4_rescore_batch{batch}_{scorer}.jsonl")
    by_id_variant = {}
    for line in open(path):
        r = json.loads(line)
        by_id_variant[(r["id"], r["variant"])] = r
    return by_id_variant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 2])
    args = ap.parse_args()

    per_doc_rows = []  # Delta H, Delta log-PPL, one row per (id, variant, scorer)
    cell_pool = collections.defaultdict(lambda: {"clean": [], "edited": [], "ids": []})
    # cell_pool key: (variant, domain, scorer)

    for batch in args.batches:
        docs_meta = {json.loads(l)["id"]: json.loads(l)
                     for l in open(os.path.join(HERE, "data", f"p4_edits_batch{batch}.jsonl"))}
        for scorer in SCORERS:
            rescore = load_rescore(batch, scorer)
            n_paired = 0
            n_unpaired_clean_missing = 0
            n_unpaired_edited_skipped = 0
            for doc_id, meta in docs_meta.items():
                clean_key = (doc_id, "clean")
                if clean_key not in rescore:
                    n_unpaired_clean_missing += 1
                    continue
                clean_rec = rescore[clean_key]
                if clean_rec.get("skipped_below_min_tokens") or not clean_rec["scores"]:
                    n_unpaired_clean_missing += 1
                    continue
                clean_scores = clean_rec["scores"]
                clean_H, _ = calc_H(generate_trajectory(clean_scores)[1])
                clean_logppl = float(np.mean(clean_scores))
                clean_traj = generate_trajectory(clean_scores)[0]

                for vtype in ALL_TYPES:
                    ed_key = (doc_id, vtype)
                    if ed_key not in rescore:
                        continue
                    ed_rec = rescore[ed_key]
                    if ed_rec.get("skipped_below_min_tokens") or not ed_rec["scores"]:
                        n_unpaired_edited_skipped += 1
                        continue
                    ed_scores = ed_rec["scores"]
                    ed_H, _ = calc_H(generate_trajectory(ed_scores)[1])
                    ed_logppl = float(np.mean(ed_scores))
                    ed_traj = generate_trajectory(ed_scores)[0]

                    per_doc_rows.append({
                        "id": doc_id, "batch": batch, "domain": meta["domain"],
                        "generator": meta["generator"], "variant": vtype, "scorer": scorer,
                        "clean_H": float(clean_H), "edited_H": float(ed_H),
                        "delta_H": float(ed_H - clean_H),
                        "clean_logppl": clean_logppl, "edited_logppl": ed_logppl,
                        "delta_logppl": ed_logppl - clean_logppl,
                        "clean_n_scored": clean_rec["n_scored"], "edited_n_scored": ed_rec["n_scored"],
                    })
                    n_paired += 1

                    cell_key = (vtype, meta["domain"], scorer)
                    cell_pool[cell_key]["clean"].append(clean_traj)
                    cell_pool[cell_key]["edited"].append(ed_traj)
                    cell_pool[cell_key]["ids"].append(doc_id)

            print(f"[family_d] batch {batch} scorer={scorer}: {n_paired} paired rows, "
                  f"{n_unpaired_clean_missing} docs with no usable clean baseline, "
                  f"{n_unpaired_edited_skipped} edited variants skipped (below MIN_TOKENS)",
                  flush=True)

    per_doc_path = os.path.join(HERE, "data", "p4_family_d_per_doc.jsonl")
    with open(per_doc_path, "w") as f:
        for row in per_doc_rows:
            f.write(json.dumps(row) + "\n")
    print(f"[family_d] wrote {len(per_doc_rows)} per-doc rows (Delta H, Delta log-PPL only, "
          f"S is NOT a per-row column here -- see cell-level output) to {per_doc_path}", flush=True)

    cell_rows = []
    for (vtype, domain, scorer), pool in sorted(cell_pool.items()):
        n = len(pool["ids"])
        if n < MIN_CELL_DOCS:
            print(f"[family_d] SKIPPED cell {vtype}/{domain}/{scorer}: n={n} < {MIN_CELL_DOCS}", flush=True)
            continue
        S_clean, S_clean_r2 = calc_S(pool["clean"])
        S_edited, S_edited_r2 = calc_S(pool["edited"])
        cell_rows.append({
            "variant": vtype, "domain": domain, "scorer": scorer, "n_docs": n,
            "S_clean": float(S_clean), "S_clean_r2": float(S_clean_r2),
            "S_edited": float(S_edited), "S_edited_r2": float(S_edited_r2),
            "delta_S": float(S_edited - S_clean),
        })
        print(f"[family_d] cell {vtype:22} {domain:18} {scorer:8} n={n:5}  "
              f"S_clean={S_clean:+.4f}  S_edited={S_edited:+.4f}  delta_S={S_edited-S_clean:+.4f}",
              flush=True)

    cell_path = os.path.join(HERE, "data", "p4_family_d_cells.jsonl")
    with open(cell_path, "w") as f:
        for row in cell_rows:
            f.write(json.dumps(row) + "\n")
    print(f"[family_d] wrote {len(cell_rows)} pooled-cell rows (Delta S, the ONLY grain S is "
          f"reported at) to {cell_path}", flush=True)


if __name__ == "__main__":
    main()
