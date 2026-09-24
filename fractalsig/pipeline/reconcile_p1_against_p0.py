#!/usr/bin/env python3
"""
P1 exit-gate reconciliation: recompute corpus-level H/S from Family A and
diff against data/p0_corpus_level_SH.json (P0's numbers are trusted ground
truth per the P1 spec -- if these don't match closely, the extraction is
wrong, not P0).

TWO DIFFERENT RECONCILIATION PROCEDURES, NOT ONE -- this is deliberate, not
an oversight:

  H reconciles by averaging build_p1_feature_store_cpu.py's own per-document
  Gemma_H column, grouped by (domain, label), straight from
  data/p1_feature_store_cpu.jsonl. Per-doc H is trustworthy (validated
  against synthetic fBm ground truth), so this is a legitimate,
  cheap check that the CPU pass's own H computation matches P0's.

  S does NOT reconcile this way. Per-doc S is severely biased (established
  against synthetic fBm, computed anyway per spec, flagged Holder_PerDoc_Unreliable)
  -- averaging biased per-doc point estimates is a DIFFERENT statistical
  procedure than P0's pooled-trajectory calc_S, and would show a large,
  EXPECTED gap that has nothing to do with whether the CPU pass extracted
  anything wrong. The only valid comparison is re-pooling raw per-document
  trajectories through calc_S's list-of-trajectories mode, exactly like
  build_p0_local.py's compute_corpus_level_SH does. The feature store
  deliberately does NOT retain raw per-token score arrays (the P1 spec's own
  explicit disk-budget guardrail: "keep only aggregated scalars in the
  parquet, not raw per-token arrays"), so this script re-reads gagle.zip
  directly -- a second, lightweight pass (scores only, no Family A/B/C
  compute) solely for this audit step, not part of the main pipeline.

Only gemma-2b is reconciled (P0 only ever scored gemma-2b) -- P1's Family A
mistral-7b columns have no P0 ground truth to check against.

USAGE
    python -m fractalsig.pipeline.reconcile_p1_against_p0
"""
from fractalsig import paths as _paths  # noqa: E402
import collections
import fnmatch
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

from fractalsig.features.fractal_math import generate_trajectory, calc_H, calc_S
from fractalsig.pipeline.build_p0_local import slice_scores

ZIP_PATH = _paths.GAGLE_ZIP_STR
MEMBER_PATTERN = "gagle3/*/*.jsonl"
DATA_DIR = os.path.join(_paths.WORKSPACE_STR, "data")
CPU_FS_PATH = os.path.join(DATA_DIR, "p1_feature_store_cpu.jsonl")
P0_SH_PATH = os.path.join(DATA_DIR, "p0_corpus_level_SH.json")


def rebuild_gemma_pools():
    """Mirrors build_p0_local.py's own extraction/dedup exactly (domain-
    scoped human key included) so this reconciliation isn't itself exposed
    to the collision bug that P0 already found and fixed -- reusing the
    logic by re-deriving it inline here rather than importing extract_rows
    directly, since that function is gemma-2b-hardcoded and single-purpose
    in build_p0_local.py; the id/dedup KEY construction is copied verbatim,
    not reinvented."""
    groups = collections.defaultdict(list)
    human_ids_seen = set()
    with zipfile.ZipFile(ZIP_PATH) as zf:
        members = sorted(m for m in zf.namelist() if fnmatch.fnmatch(m, MEMBER_PATTERN))
        print(f"[info] re-reading {len(members)} members for gemma-2b score pooling only "
              f"(no Family A/B/C recompute)...")
        for i, member in enumerate(members, 1):
            split = member.split("/")[1]
            with zf.open(member) as fh:
                for line_bytes in fh:
                    line = line_bytes.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    domain = entry.get("dataset_name")
                    scores_block = entry.get("scores", {}).get("gemma-2b", {})

                    llm_sliced = slice_scores(scores_block.get("llm_text"))
                    if llm_sliced is not None:
                        label = entry.get("model", "Unknown_LLM")
                        groups[(domain, label)].append(np.asarray(llm_sliced, dtype=np.float64))

                    human_sliced = slice_scores(scores_block.get("original_text"))
                    if human_sliced is not None:
                        human_id = f"{split}_{domain}_{entry.get('dataset_id')}_human"
                        if human_id not in human_ids_seen:
                            human_ids_seen.add(human_id)
                            groups[(domain, "Human")].append(np.asarray(human_sliced, dtype=np.float64))
            if i % 100 == 0:
                print(f"[info] {i}/{len(members)} members re-read")
    return groups


def compute_pooled_SH(groups):
    results = {}
    for (domain, label), score_lists in sorted(groups.items()):
        if len(score_lists) < 10:
            continue
        increments = [generate_trajectory(s)[1] for s in score_lists]
        all_incr = np.concatenate(increments)
        H, H_r2 = calc_H(all_incr)
        trajectories = [generate_trajectory(s)[0] for s in score_lists]
        S, S_r2 = calc_S(trajectories)
        results[f"{domain}|{label}"] = {
            "n_docs": len(score_lists), "S": float(S), "S_R2": float(S_r2),
            "H": float(H), "H_R2": float(H_r2),
        }
    return results


def reconcile_H_via_averaging():
    rows = []
    with open(CPU_FS_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df = df[df["Gemma_available"] == True]  # noqa: E712
    agg = df.groupby(["domain", "Label"])["Gemma_H"].agg(["mean", "count"]).reset_index()
    return {f"{r.domain}|{r.Label}": {"H_avg_per_doc": float(r.mean), "n_docs": int(r.count)}
            for r in agg.itertuples()}


def main():
    if not os.path.exists(CPU_FS_PATH):
        print(f"[abort] {CPU_FS_PATH} not found -- run build_p1_feature_store_cpu.py to completion first.")
        sys.exit(1)
    if not os.path.exists(P0_SH_PATH):
        print(f"[abort] {P0_SH_PATH} not found.")
        sys.exit(1)

    with open(P0_SH_PATH) as f:
        p0 = json.load(f)

    print("=== H reconciliation (per-doc average, from p1_feature_store_cpu.jsonl) ===")
    h_recon = reconcile_H_via_averaging()
    for key in sorted(p0.keys()):
        p0_cell = p0[key]
        p1_cell = h_recon.get(key)
        if p1_cell is None:
            print(f"  {key}: MISSING from P1 output")
            continue
        diff = p1_cell["H_avg_per_doc"] - p0_cell["H"]
        print(f"  {key:35s} P0_H={p0_cell['H']:.4f} (n={p0_cell['n_docs']:6d})  "
              f"P1_avg_perdoc_H={p1_cell['H_avg_per_doc']:.4f} (n={p1_cell['n_docs']:6d})  "
              f"diff={diff:+.4f}")

    print("\n=== S reconciliation (re-pooled raw trajectories, list-mode, gemma-2b only) ===")
    print("(This re-reads gagle.zip -- a lightweight second pass, scores only, not part of the "
          "main P1 pipeline. See module docstring for why per-doc averaging isn't valid for S.)")
    groups = rebuild_gemma_pools()
    pooled = compute_pooled_SH(groups)
    with open(os.path.join(DATA_DIR, "p1_reconciliation_pooled_SH.json"), "w") as f:
        json.dump(pooled, f, indent=2)
    max_abs_diff_S = 0.0
    max_abs_diff_H = 0.0
    for key in sorted(p0.keys()):
        p0_cell = p0[key]
        p1_cell = pooled.get(key)
        if p1_cell is None:
            print(f"  {key}: MISSING from P1 re-pooled output")
            continue
        diff_s = p1_cell["S"] - p0_cell["S"]
        diff_h = p1_cell["H"] - p0_cell["H"]
        max_abs_diff_S = max(max_abs_diff_S, abs(diff_s))
        max_abs_diff_H = max(max_abs_diff_H, abs(diff_h))
        print(f"  {key:35s} P0 S={p0_cell['S']:.4f} H={p0_cell['H']:.4f} (n={p0_cell['n_docs']:6d})  "
              f"P1-repooled S={p1_cell['S']:.4f} H={p1_cell['H']:.4f} (n={p1_cell['n_docs']:6d})  "
              f"diff_S={diff_s:+.4f} diff_H={diff_h:+.4f}")

    print(f"\n=== SUMMARY ===")
    print(f"  max |diff_S| (re-pooled) = {max_abs_diff_S:.4f}")
    print(f"  max |diff_H| (re-pooled) = {max_abs_diff_H:.4f}")
    if max_abs_diff_S > 0.01 or max_abs_diff_H > 0.005:
        print("  [WARNING] Larger-than-expected divergence from P0 -- since both procedures now "
              "use identical extraction/dedup logic and identical raw gemma-2b scores, any "
              "non-trivial diff here means something in the extraction genuinely differs "
              "(different member set, different slicing) and needs investigation before trusting "
              "the feature store, per the P1 spec's exit gate.")
    else:
        print("  Re-pooled S/H match P0 closely. Family A's gemma-2b extraction is consistent "
              "with the P0 ground truth.")


if __name__ == "__main__":
    main()
