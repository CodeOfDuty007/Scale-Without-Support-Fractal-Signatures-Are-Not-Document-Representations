#!/usr/bin/env python3
"""RAID ingest (2026-08-26): parses `data/raid_raw/train.csv` into the same
row-schema shape GAGLE's `build_p0_local.py`/`extract_rows` established
(`id`, `domain`, `Label`/generator, `text`) plus RAID's own `attack` column
carried through unmodified, so downstream code (scoring, Family A-C,
robustness eval) doesn't need RAID-specific branches.

Scope (locked 2026-08-26, `Extension1_P4P5_Detailed_Plan.md` Sec1/Sec5
decision 5, full reasoning + real counts there): `train.csv` only, 9 of 12
`attack` categories. USE: `none` (clean, needed by P5's GAGLE->RAID transfer
task and P4 task 7's GAGLE->RAID model-family-shift), `paraphrase`, and 7
noise/formatting attacks (`whitespace`, `upper_lower`, `perplexity_misspelling`,
`homoglyph`, `article_deletion`, `alternative_spelling`, `zero_width_space`).
DISCARD: `synonym`, `number`, `insert_paragraphs` (no mapping to any planned
task), all of `extra.csv` (same structure, decoding-parameter variants of
the same generators, not needed) and all of `test.csv` (checked directly --
`id, generation` only, no labels, RAID's own blind leaderboard test set,
unusable here).

`train.csv` is 5,615,820 rows -- read in chunks, not all at once.
"""
from fractalsig import paths as _paths  # noqa: E402
import argparse
import json
import os

import pandas as pd

HERE = _paths.WORKSPACE_STR
TRAIN_CSV = os.path.join(HERE, "data", "raid_raw", "train.csv")
OUT_PATH = os.path.join(HERE, "data", "raid_ingested.jsonl")

USE_ATTACKS = {
    "none", "paraphrase",
    "whitespace", "upper_lower", "perplexity_misspelling", "homoglyph",
    "article_deletion", "alternative_spelling", "zero_width_space",
}
DISCARD_ATTACKS = {"synonym", "number", "insert_paragraphs"}

CHUNK_SIZE = 200_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit_rows", type=int, default=None,
                     help="stop after reading this many raw CSV rows (smoke testing)")
    args = ap.parse_args()

    assert USE_ATTACKS.isdisjoint(DISCARD_ATTACKS)

    n_read = 0
    n_kept = 0
    n_by_attack = {}
    n_by_model = {}
    n_by_domain = {}
    n_missing_text = 0

    out_f = open(OUT_PATH, "w")
    reader = pd.read_csv(
        TRAIN_CSV,
        usecols=["id", "model", "domain", "attack", "generation"],
        chunksize=CHUNK_SIZE,
    )
    for chunk in reader:
        if args.limit_rows is not None and n_read >= args.limit_rows:
            break
        n_read += len(chunk)
        keep = chunk[chunk["attack"].isin(USE_ATTACKS)]
        for row in keep.itertuples(index=False):
            text = row.generation
            if not isinstance(text, str) or not text.strip():
                n_missing_text += 1
                continue
            out_f.write(json.dumps({
                "id": f"raid_{row.id}",
                "domain": row.domain,
                "Label": row.model,       # generator label, GAGLE-schema convention
                "attack": row.attack,
                "text": text,
                "source": "raid",
            }) + "\n")
            n_kept += 1
            n_by_attack[row.attack] = n_by_attack.get(row.attack, 0) + 1
            n_by_model[row.model] = n_by_model.get(row.model, 0) + 1
            n_by_domain[row.domain] = n_by_domain.get(row.domain, 0) + 1
        print(f"[ingest_raid] read {n_read} raw rows, kept {n_kept} so far...", flush=True)
        if args.limit_rows is not None and n_read >= args.limit_rows:
            break
    out_f.close()

    print(f"\n=== INGEST SUMMARY ===", flush=True)
    print(f"Raw rows read: {n_read}", flush=True)
    print(f"Rows kept (attack in USE_ATTACKS, non-empty text): {n_kept}", flush=True)
    print(f"Rows dropped (empty/non-string text): {n_missing_text}", flush=True)
    print(f"By attack: {n_by_attack}", flush=True)
    print(f"By model: {n_by_model}", flush=True)
    print(f"By domain: {n_by_domain}", flush=True)
    print(f"Written to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
