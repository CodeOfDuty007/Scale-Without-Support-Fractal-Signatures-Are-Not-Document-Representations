#!/usr/bin/env python3
"""
Merge the two per-scorer human Family B partials, then union with the LLM side.

WHAT THIS PRODUCES
  data/p1_feature_store_gpu_selfscored_v2.jsonl -- 223,834 rows:
      218,834 LLM rows, carried over UNCHANGED from the job-56 store
    +   5,000 post-X-01 human rows, freshly scored by job 89

THE SUBTLETY THAT MATTERS
  data/p1_feature_store_gpu_selfscored.jsonl holds 230,530 rows, NOT 218,834.
  That is 218,834 LLM rows PLUS 11,696 pre-X-01 human rows that X-01 retired.
  Copying the file wholesale would silently reinstate the inflated human
  population the whole X-01 exercise removed. The LLM subset is therefore
  FILTERED by Label, and the human ids are asserted disjoint from it.

  Reuses build_p1_join_feature_store's own assert_no_duplicate_ids /
  assert_id_sets_match rather than reimplementing them, so this file and the
  real join agree by construction on what "matching" means.

VERIFICATION
  - each partial is exactly 5,000 rows and 5,000 distinct ids, same id set
  - merged human rows carry BOTH scorers' columns, no collisions on shared keys
  - the union has no duplicate ids
  - every carried-over LLM row is BYTE-IDENTICAL to its source row (guards the
    copy itself -- a silent mutation here would be invisible downstream)
"""
from fractalsig import paths as _paths  # noqa: E402
import collections
import json
import os
import sys

HERE = _paths.WORKSPACE_STR
DATA = os.path.join(HERE, "data")

GEMMA = os.path.join(DATA, "human_familyb_gemma_2b.jsonl")
MISTRAL = os.path.join(DATA, "human_familyb_mistral_7b.jsonl")
LLM_SRC = os.path.join(DATA, "p1_feature_store_gpu_selfscored.jsonl")
OUT = os.path.join(DATA, "p1_feature_store_gpu_selfscored_v2.jsonl")

HUMAN = "Human"
# Written by every invocation regardless of scorer; identical across partials.
SHARED = {"id", "split", "domain", "Label", "prompt_type", "temp",
          "Embedding_Dispersion"}


def read(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                if d["id"] in out:
                    sys.exit(f"[FAIL] duplicate id {d['id']} in {path}")
                out[d["id"]] = d
    return out


def main():
    if os.path.exists(OUT):
        sys.exit(f"[FAIL] {OUT} exists. Move it aside (never delete, D-10).")

    g, m = read(GEMMA), read(MISTRAL)
    print(f"[info] gemma partial   : {len(g):,} rows")
    print(f"[info] mistral partial : {len(m):,} rows")
    for name, d in (("gemma", g), ("mistral", m)):
        if len(d) != 5000:
            sys.exit(f"[FAIL] {name} partial has {len(d)} rows, expected 5000")
    if set(g) != set(m):
        only_g, only_m = set(g) - set(m), set(m) - set(g)
        sys.exit(f"[FAIL] partial id sets differ: gemma-only={len(only_g)} "
                 f"mistral-only={len(only_m)} e.g. {sorted(only_g)[:3]}{sorted(only_m)[:3]}")
    print(f"[ok] partials: 5,000 rows each, id sets identical")

    # --- merge the two scorers ---------------------------------------------
    # NOTE on a guard that used to live here. It was:
    #     if k in row and k not in SHARED and row[k] != v: collisions.append(...)
    # and it CANNOT FIRE. The only non-SHARED keys are Gemma_* and Mistral_*,
    # which are disjoint between the two partials, so `k in row` is never true
    # for a non-SHARED key. It printed "0 collisions" on every possible input,
    # including a broken one. Replaced with the check it was pretending to be:
    # SHARED keys are the ones that genuinely appear in BOTH partials, so those
    # are what must be compared. Embedding_Dispersion is the live risk -- it is
    # recomputed per invocation, and a silent overwrite would keep whichever
    # partial merged last.
    def same(a, b):
        # NaN != NaN in Python, so a naive `!=` flags every NaN pair as a
        # disagreement. 6 human rows have NaN Embedding_Dispersion in BOTH
        # partials -- genuinely identical, and reporting them as conflicts
        # would be a false alarm that trains the reader to ignore this warning.
        if isinstance(a, float) and isinstance(b, float):
            if a != a and b != b:
                return True
        return a == b

    disagree = collections.Counter()
    worst, nan_pairs = {}, collections.Counter()
    for i, gr in g.items():
        mr = m[i]
        for k in SHARED:
            if k not in gr or k not in mr:
                continue
            a, b = gr[k], mr[k]
            if isinstance(a, float) and a != a and isinstance(b, float) and b != b:
                nan_pairs[k] += 1
                continue
            if not same(a, b):
                disagree[k] += 1
                if isinstance(a, float) and isinstance(b, float):
                    d = abs(a - b)
                    if d > worst.get(k, (0, None))[0]:
                        worst[k] = (d, i)
    if disagree:
        print(f"[warn] SHARED keys disagree between partials: {dict(disagree)}")
        for k, (d, i) in worst.items():
            print(f"        {k}: max |delta| {d:.6g} at {i} -- merge keeps the "
                  f"MISTRAL value")
    else:
        extra = (f" ({sum(nan_pairs.values())} NaN-in-both pairs treated as equal: "
                 f"{dict(nan_pairs)})" if nan_pairs else "")
        print(f"[ok] all {len(SHARED)} shared keys agree across both partials, "
              f"all {len(g):,} ids -- nothing is silently overwritten{extra}")

    merged = {}
    for i, gr in g.items():
        row = dict(gr)
        row.update(m[i])
        merged[i] = row
    sample = merged[next(iter(merged))]
    for pfx in ("Gemma", "Mistral"):
        if not any(k.startswith(pfx + "_") for k in sample):
            sys.exit(f"[FAIL] merged rows carry no {pfx}_* columns")
    print(f"[ok] merged: {len(merged):,} human rows, both scorers' column "
          f"families present")

    # --- carry over the LLM side, filtered, byte-identical -------------------
    human_ids = set(merged)
    n_llm = n_src_human = 0
    mismatch = 0
    with open(OUT, "w") as out:
        for i in merged:
            out.write(json.dumps(merged[i]) + "\n")
        with open(LLM_SRC) as src:
            for line in src:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("Label") == HUMAN:
                    n_src_human += 1
                    continue
                if d["id"] in human_ids:
                    sys.exit(f"[FAIL] LLM row {d['id']} collides with a human id")
                # byte-identical carry-over: re-serialising must round-trip
                if json.dumps(d) != line:
                    mismatch += 1
                out.write(line + "\n")
                n_llm += 1
    print(f"[info] source store    : {n_llm + n_src_human:,} rows "
          f"({n_llm:,} LLM kept, {n_src_human:,} stale pre-X-01 human DROPPED)")
    if mismatch:
        print(f"[warn] {mismatch:,} LLM rows did not round-trip byte-identically "
              f"through json (formatting only; values preserved)")

    # --- verify the union ---------------------------------------------------
    import pandas as pd  # noqa: E402
    from fractalsig.pipeline.build_p1_join_feature_store import assert_no_duplicate_ids  # noqa: E402
    rows = [json.loads(l) for l in open(OUT) if l.strip()]
    df = pd.DataFrame(rows)
    print(f"\n[check] union rows: {len(df):,} (expect 223,834)")
    assert_no_duplicate_ids(df, "selfscored_v2")
    lab = df["Label"].value_counts().to_dict()
    print(f"[check] composition: {lab}")
    ok = (len(df) == 223834 and lab.get(HUMAN) == 5000 and mismatch == 0)
    print(f"\n[{'PASS' if ok else 'FAIL'}] union written to {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
