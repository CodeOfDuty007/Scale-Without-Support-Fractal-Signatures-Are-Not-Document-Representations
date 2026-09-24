#!/usr/bin/env python3
"""P4 task 5: build the single shared scoring population every detector reads,
so all five detectors are compared on an identical document set (per gate
finding: five detectors deriving their own populations independently produces
five denominators, the same class of problem the gate spent its length
fixing).

Constraints (all established during the 2026-08-27 validity gate):
  - human_revision EXCLUDED ENTIRELY (89.2% near-identity, design decision:
    drop, not regenerate)
  - semantic_cosine >= 0.5 (Defect A: prompt-echo / topic hallucination)
  - generation-survived: not below_min_tokens, not is_empty
  - summarize_then_expand flagged `separate_bucket: true` -- present in the
    population (own detector numbers are still worth reporting) but excluded
    from the primary battery/common-subset comparison
  - complete-case only: a row only enters the population if BOTH scorers'
    rescore records exist and are not skipped_below_min_tokens (this is what
    "no NaN-carrying row reaches the tabular detector" cashes out to, since
    Family A/B for that row are derived from the rescore arrays)

Output: data/p4_task5_population.jsonl, one row per (id, variant), with the
edited text and both scorers' rescore keys resolved so every downstream
detector script can `for row in population: text = row["text"]`.
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os

HERE = _paths.WORKSPACE_STR
D = lambda *p: os.path.join(HERE, *p)

SCORED_TYPES = ["paraphrase_light", "paraphrase_heavy", "style_transfer",
                "sentence_reorder", "typo_noise", "summarize_then_expand"]
SEPARATE_BUCKET = {"summarize_then_expand"}
COSINE_FLOOR = 0.5
SCORERS = ["gemma", "mistral"]


def load_rescore(batch, scorer):
    path = D("data", f"p4_rescore_batch{batch}_{scorer}.jsonl")
    by_key = {}
    for line in open(path):
        r = json.loads(line)
        by_key[(r["id"], r["variant"])] = r
    return by_key


def main():
    out = open(D("data", "p4_task5_population.jsonl"), "w")
    n_by_type = {t: 0 for t in SCORED_TYPES}
    n_by_type_sep = {t: 0 for t in SCORED_TYPES}

    for batch in (1, 2):
        edits = {json.loads(l)["id"]: json.loads(l)
                  for l in open(D("data", f"p4_edits_batch{batch}.jsonl"))}
        rescore = {s: load_rescore(batch, s) for s in SCORERS}

        for doc_id, rec in edits.items():
            # clean baseline must survive both scorers too (Family D consumes it)
            clean_ok = all(
                (doc_id, "clean") in rescore[s]
                and not rescore[s][(doc_id, "clean")].get("skipped_below_min_tokens")
                and rescore[s][(doc_id, "clean")]["scores"]
                for s in SCORERS)
            if not clean_ok:
                continue

            for vtype in SCORED_TYPES:
                v = rec.get("variants", {}).get(vtype)
                if v is None or v.get("below_min_tokens") or v.get("is_empty"):
                    continue
                c = v.get("semantic_cosine")
                if c is not None and c < COSINE_FLOOR:
                    continue
                both_scorers_ok = all(
                    (doc_id, vtype) in rescore[s]
                    and not rescore[s][(doc_id, vtype)].get("skipped_below_min_tokens")
                    and rescore[s][(doc_id, vtype)]["scores"]
                    for s in SCORERS)
                if not both_scorers_ok:
                    continue

                row = {
                    "id": doc_id, "batch": batch, "domain": rec["domain"],
                    "generator": rec["generator"], "variant": vtype,
                    "text": v["text"], "semantic_cosine": c,
                    "separate_bucket": vtype in SEPARATE_BUCKET,
                    "clean_text": None,  # filled by consumers from p4_sampled_docs.jsonl if needed
                }
                out.write(json.dumps(row) + "\n")
                n_by_type[vtype] += 1
                if vtype in SEPARATE_BUCKET:
                    n_by_type_sep[vtype] += 1

    out.close()
    print("[population] rows per type:")
    for t in SCORED_TYPES:
        tag = " (separate bucket)" if t in SEPARATE_BUCKET else ""
        print(f"  {t:22} {n_by_type[t]:6}{tag}")


if __name__ == "__main__":
    main()
