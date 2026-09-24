#!/usr/bin/env python3
"""P5 track A1 (2026-08-27): stratified RAID sample, single
CPU/IO pass over the 4,211,865-row ingested corpus (`ingest_raid.py`'s
output). Full-scale RAID is DEFERRED (see ledger D-05 amendment), priced at
~6.5 GPU-days -- this sample is scored instead, sized to schedule against
the Sep 11 deadline while remaining a comparable-scale dataset to GAGLE
(223,834 rows).

Stratification priority, per design decision:
  1. generator family (P5 task 6's new 11-class label, plus human as its
     own class for the binary GAGLE->RAID transfer task) -- BALANCED, not
     proportional, since a starved generator cell breaks task 6 outright.
  2. domain (8 RAID domains) -- balanced within each generator budget.
  3. RAID's own attack type (9 of 12 in scope) -- balanced within each
     (generator, domain) cell; trivial here since the corpus itself is
     already perfectly attack-balanced (verified: every (generator,attack)
     cell is exactly total/9).

Budget split: 50% human (60,000) / 50% LLM (60,000, 11 generators x 5,455
each) -- serves BOTH consumers of this one sample: the binary human-vs-LLM
transfer task wants a 50/50 class balance, task 6's 11-way generator
classifier wants its 11 classes balanced against each other (human is not
one of task 6's classes). Verified before running (see ledger): every
realized (generator, domain, attack) cell in the full corpus has 800+
times the requested per-cell count -- no starvation risk anywhere in this
design, unlike a naive proportional sample which would still be fine here
(RAID's attack dimension is perfectly balanced) but would NOT be fine on
generator (6 generators have 481,356 rows, 5 have 240,678, human has only
120,339 -- proportional sampling would under-represent every one of the
6 chat/base-model generators relative to a hypothetical needed balance,
and human would already be the smallest class before any attack subsetting).

Single-pass reservoir sampling (Algorithm R) per (generator, domain,
attack) cell -- exact target counts, no second pass, no more than 120k
rows resident in memory at once (attack.RESERVOIR total). Reads
data/raid_ingested.jsonl exactly once.
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
import random
from collections import defaultdict

HERE = _paths.WORKSPACE_STR
D = lambda *p: os.path.join(HERE, "data", *p)

GENERATORS = ["chatgpt", "cohere", "cohere-chat", "gpt2", "gpt3", "gpt4",
              "llama-chat", "mistral", "mistral-chat", "mpt", "mpt-chat"]
DOMAINS = ["abstracts", "books", "news", "poetry", "recipes", "reddit", "reviews", "wiki"]
ATTACKS = ["none", "whitespace", "upper_lower", "perplexity_misspelling", "paraphrase",
           "homoglyph", "article_deletion", "alternative_spelling", "zero_width_space"]

HUMAN_BUDGET = 60_000
LLM_BUDGET_PER_GENERATOR = 5_455
SEED = 20260827


def build_cell_targets():
    targets = {}
    for d in DOMAINS:
        base = HUMAN_BUDGET // (len(DOMAINS) * len(ATTACKS))
        rem = HUMAN_BUDGET - base * len(DOMAINS) * len(ATTACKS)
        for a in ATTACKS:
            targets[("human", d, a)] = base
    # distribute the human remainder across the first `rem` cells deterministically
    human_cells = [("human", d, a) for d in DOMAINS for a in ATTACKS]
    for i in range(rem):
        targets[human_cells[i]] += 1

    for g in GENERATORS:
        base = LLM_BUDGET_PER_GENERATOR // (len(DOMAINS) * len(ATTACKS))
        rem = LLM_BUDGET_PER_GENERATOR - base * len(DOMAINS) * len(ATTACKS)
        gen_cells = [(g, d, a) for d in DOMAINS for a in ATTACKS]
        for c in gen_cells:
            targets[c] = base
        for i in range(rem):
            targets[gen_cells[i]] += 1
    return targets


def main():
    rng = random.Random(SEED)
    targets = build_cell_targets()
    reservoirs = {cell: [] for cell in targets}
    seen = defaultdict(int)

    print(f"[p5-sample] {len(targets)} cells, total target = {sum(targets.values())}", flush=True)

    n_read = 0
    with open(D("raid_ingested.jsonl")) as f:
        for line in f:
            n_read += 1
            if n_read % 1_000_000 == 0:
                print(f"[p5-sample]   read {n_read}", flush=True)
            r = json.loads(line)
            cell = (r["Label"], r["domain"], r["attack"])
            k = targets.get(cell, 0)
            if k == 0:
                continue
            seen[cell] += 1
            res = reservoirs[cell]
            if len(res) < k:
                res.append(r)
            else:
                j = rng.randint(0, seen[cell] - 1)
                if j < k:
                    res[j] = r

    print(f"[p5-sample] read {n_read} rows total", flush=True)

    out_rows = []
    for cell, res in reservoirs.items():
        out_rows.extend(res)
    rng.shuffle(out_rows)

    with open(D("p5_raid_sample.jsonl"), "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    print(f"[p5-sample] wrote {len(out_rows)} rows to data/p5_raid_sample.jsonl", flush=True)

    # realized cell-count report
    realized = defaultdict(int)
    for r in out_rows:
        realized[r["Label"]] += 1
    print(f"\n{'generator':16}{'target':>10}{'realized':>10}")
    gen_targets = {"human": HUMAN_BUDGET}
    for g in GENERATORS:
        gen_targets[g] = LLM_BUDGET_PER_GENERATOR
    for g in ["human"] + GENERATORS:
        print(f"{g:16}{gen_targets[g]:10}{realized.get(g,0):10}")

    short_cells = {str(c): {"target": t, "available": seen.get(c, 0)}
                   for c, t in targets.items() if seen.get(c, 0) < t}
    if short_cells:
        print(f"\n[p5-sample] WARNING: {len(short_cells)} cells had FEWER available rows than target:")
        for c, v in short_cells.items():
            print(f"  {c}: target={v['target']} available={v['available']}")
    else:
        print("\n[p5-sample] every cell met its target exactly, no shortfall")

    with open(D("p5_raid_sample_report.json"), "w") as f:
        json.dump({"n_total": len(out_rows), "cell_targets": {str(k): v for k, v in targets.items()},
                    "cell_available": {str(k): v for k, v in seen.items()},
                    "generator_realized": dict(realized), "short_cells": short_cells}, f, indent=2)
    print("written to data/p5_raid_sample_report.json")


if __name__ == "__main__":
    main()
