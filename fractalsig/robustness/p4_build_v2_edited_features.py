#!/usr/bin/env python3
"""P0-P5 audit continuation, step 3 (2026-08-29/30): build `feature_store/
v2_edited.parquet` -- P4's other named exit-gate deliverable
(`Extension1_PhaseWise_Execution_Plan.md:123`, "Families A-D on edited
data"), never produced at P4's original close.

Checked before writing this (internal review): does Family B (GLTR rank
buckets, entropy) require a new GPU pass, or is it already recoverable from
the existing rescore files? `p4_rescore_edits.py` (the script that produced
`data/p4_rescore_batch{1,2}_{gemma,mistral}.jsonl`) reuses `score_batch`
from `fractalsig/pipeline/build_p1_feature_store_gpu.py`, which computes AND PERSISTS
`ranks`/`entropies` per token (verified directly against the file schema,
not assumed from the docstring) -- so Family B is fully recoverable, no new
GPU scoring needed. This closes as originally scoped: CPU-only.

Reuses `build_feature_row` from `p4_task5_tabular_features.py` AS-IS -- the
SAME function that already assembles Family A (fractal, via `features_for`),
Family B (via `family_b_feature_summary`), Family C (via `family_c_style`),
and embedding dispersion for this exact rescore-file schema, because it's
what P4 task 5's own `tabular` detector runs on this population already
(`data/p4_task5_tabular_scores.jsonl` -- but that file only persists the
detector's calibrated probability, not the underlying feature vector, so
this script's contribution is producing and KEEPING the feature vector
itself). No new feature code, matching `p5_raid_build_features.py`'s
identical reuse pattern for RAID.

Family D deltas (`data/p4_family_d_v2_full_per_doc.jsonl`, scope="full",
already computed) are added as extra columns per scorer
(`{Scorer}_Delta_H`, `{Scorer}_Delta_LogPPL`) -- the "D" in "Families A-D."

Population: `data/p4_task5_population.jsonl` (39,810 rows, `human_revision`
already excluded, semantic_cosine<0.5 already filtered -- the same
already-quality-gated population `p4_edited_vs_clean_classifier.py` used).
One row per (id, variant) = one edited document; the clean side is NOT
duplicated here (it has no `variant`/attack identity of its own and is
already covered by `v1_gagle.parquet`'s own GAGLE rows for the same base
documents) -- this store is specifically the EDITED-side feature vectors
Extension1_PhaseWise_Execution_Plan.md's `v2_edited.parquet` calls for.
"""
from fractalsig import paths as _paths  # noqa: E402
import argparse
import json
import os
import sys
import collections

import numpy as np
import pandas as pd

HERE = _paths.WORKSPACE_STR
D = lambda *p: os.path.join(HERE, "data", *p)
FS = lambda *p: os.path.join(HERE, "feature_store", *p)

from fractalsig.robustness.p4_task5_tabular_features import build_feature_row  # noqa: E402
from fractalsig.pipeline.build_p1_feature_store_gpu import load_models  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="smoke-test: only featurize this many population rows")
    args = ap.parse_args()

    print("[v2-edited] loading embedder...", flush=True)
    _, device, _, _, _, embedder = load_models(scorers=())
    print(f"[v2-edited] embedder loaded on device={device}", flush=True)

    print("[v2-edited] loading rescore files...", flush=True)
    rescore = {"gemma": {}, "mistral": {}}
    for scorer in ("gemma", "mistral"):
        for batch in (1, 2):
            for l in open(D(f"p4_rescore_batch{batch}_{scorer}.jsonl")):
                r = json.loads(l)
                rescore[scorer][(r["id"], r["variant"])] = r
    print(f"[v2-edited] {len(rescore['gemma'])} gemma, {len(rescore['mistral'])} mistral rescore records", flush=True)

    print("[v2-edited] loading Family D deltas...", flush=True)
    family_d = {}
    for l in open(D("p4_family_d_v2_full_per_doc.jsonl")):
        r = json.loads(l)
        family_d[(r["id"], r["variant"], r["scorer"])] = r
    print(f"[v2-edited] {len(family_d)} Family D per-doc records", flush=True)

    pop = [json.loads(l) for l in open(D("p4_task5_population.jsonl"))]
    if args.limit:
        pop = pop[:args.limit]
    print(f"[v2-edited] {len(pop)} population rows to featurize", flush=True)
    assert not any(r["variant"] == "human_revision" for r in pop), \
        "human_revision must stay excluded (89.2% near-identity, not a real edit)"

    out_path = D("p4_v2_edited_features.jsonl")
    out = open(out_path, "w")
    n_ok, n_missing_rescore, n_missing_familyd, n_err = 0, 0, 0, 0
    for i, row in enumerate(pop):
        key = (row["id"], row["variant"])
        grec = rescore["gemma"].get(key)
        mrec = rescore["mistral"].get(key)
        if grec is None or mrec is None or grec.get("skipped_below_min_tokens") or mrec.get("skipped_below_min_tokens"):
            n_missing_rescore += 1
            continue
        gd = family_d.get((row["id"], row["variant"], "gemma"))
        md = family_d.get((row["id"], row["variant"], "mistral"))
        if gd is None or md is None:
            n_missing_familyd += 1
            continue
        try:
            feats = build_feature_row(row["text"], grec, mrec, embedder, device)
            out.write(json.dumps({
                "id": row["id"], "variant": row["variant"], "domain": row["domain"],
                "generator": row["generator"],
                **feats,
                "Gemma_Delta_H": gd["delta_H"], "Gemma_Delta_LogPPL": gd["delta_logppl"],
                "Mistral_Delta_H": md["delta_H"], "Mistral_Delta_LogPPL": md["delta_logppl"],
            }) + "\n")
            n_ok += 1
        except Exception as e:
            n_err += 1
            if n_err <= 5:
                print(f"[v2-edited] ERROR row {row['id']}/{row['variant']}: {e}", flush=True)
        if (i + 1) % 2000 == 0:
            out.flush()
            print(f"[v2-edited]   {i+1}/{len(pop)}  ok={n_ok} missing_rescore={n_missing_rescore} "
                  f"missing_familyd={n_missing_familyd} err={n_err}", flush=True)

    out.close()
    print(f"[v2-edited] DONE  ok={n_ok} missing_rescore={n_missing_rescore} "
          f"missing_familyd={n_missing_familyd} err={n_err}  -> {out_path}", flush=True)

    print("[v2-edited] converting to parquet...", flush=True)
    rows = [json.loads(l) for l in open(out_path)]
    df = pd.DataFrame(rows)
    print(f"[v2-edited] shape: {df.shape}", flush=True)
    df.to_parquet(FS("v2_edited.parquet"), index=False)
    df2 = pd.read_parquet(FS("v2_edited.parquet"))
    assert df2.shape == df.shape and list(df2.columns) == list(df.columns), "round-trip mismatch"
    print(f"[v2-edited] written and verified: feature_store/v2_edited.parquet, shape={df2.shape}", flush=True)


if __name__ == "__main__":
    main()
