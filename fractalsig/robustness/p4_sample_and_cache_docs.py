#!/usr/bin/env python3
"""P4 attack-cascade sampler (2026-08-26).

Draws the 10,000-document sample for P4's attack cascade (design decision,
2026-08-26: 10,000 total, split into two gated batches of 5,000), stratified
domain(5) x generator(2) = 10 strata, 1,000/stratum (500 batch1 + 500
batch2), sampled without replacement from the `random`-scheme TEST
partition of `feature_store/v1_gagle.parquet` -- the split the random-scheme
detector was never trained on, so post-attack re-evaluation against it is
not contaminated by the detector having seen these documents clean.

Every stratum's test-partition pool was checked before committing to this
size (2026-08-26): minimum pool across all 10 strata is 3,975
(newsroom/mistral_7b), comfortably above the 1,000/stratum draw.

Reads gagle.zip exactly ONCE (the governing constraint build_p0_local.py
established) to pull raw text for the sampled ids only -- both batches are
sampled and cached together here so batch 2 needs no second zip scan; only
whether batch 2's GENERATION proceeds is gated on batch 1's wall-clock.
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
import random
import sys
import zipfile

import pandas as pd

HERE = _paths.WORKSPACE_STR

from fractalsig.detection.splits import group_key  # noqa: E402
from fractalsig.pipeline.build_p1_feature_store_cpu import extract_docs  # noqa: E402

ZIP_PATH = os.environ.get("GAGLE_ZIP_PATH", _paths.GAGLE_ZIP_STR)
MEMBER_PATTERN_PREFIX = "gagle3/"
SEED = 20260826
PER_STRATUM_PER_BATCH = 500
MIN_WORDS = 100  # matches p4_pilot_generate_vs_score_timing.py's own floor
OUT_PATH = os.path.join(HERE, "data", "p4_sampled_docs.jsonl")


def build_sample_frame():
    pq = pd.read_parquet(
        os.path.join(HERE, "feature_store", "v1_gagle.parquet"),
        columns=["id", "domain", "Label"],
    )
    pq = pq[pq["Label"] != "Human"].copy()

    splits = json.load(open(os.path.join(HERE, "data", "splits_v1.json")))
    assign = splits["random"]

    groups = [group_key({"id": i, "domain": d, "Label": l})
              for i, d, l in zip(pq.id, pq.domain, pq.Label)]
    pq["group"] = [f"{g[0]}|{g[1]}" for g in groups]
    pq["split"] = pq["group"].map(assign)
    test = pq[pq["split"] == "test"].copy()

    rng = random.Random(SEED)
    rows = []
    pool_report = {}
    for domain in sorted(test["domain"].unique()):
        for label in sorted(test["Label"].unique()):
            pool = test[(test["domain"] == domain) & (test["Label"] == label)]["id"].tolist()
            pool_report[f"{domain}/{label}"] = len(pool)
            need = 2 * PER_STRATUM_PER_BATCH
            if len(pool) < need:
                raise SystemExit(
                    f"[FATAL] stratum {domain}/{label} has only {len(pool)} "
                    f"test-partition rows, need {need}")
            rng.shuffle(pool)
            drawn = pool[:need]
            batch1_ids = set(drawn[:PER_STRATUM_PER_BATCH])
            for doc_id in drawn:
                rows.append({
                    "id": doc_id, "domain": domain, "generator": label,
                    "batch": 1 if doc_id in batch1_ids else 2,
                })
    print(f"[sample] pool sizes per stratum: {pool_report}", flush=True)
    print(f"[sample] drew {len(rows)} ids total "
          f"({PER_STRATUM_PER_BATCH}/stratum/batch x 10 strata x 2 batches)", flush=True)
    return pd.DataFrame(rows)


def collect_texts(target_ids):
    """One pass over gagle.zip, matching extract_docs()'s own id construction."""
    zf = zipfile.ZipFile(ZIP_PATH)
    members = sorted(m for m in zf.namelist()
                      if m.startswith(MEMBER_PATTERN_PREFIX) and m.endswith(".jsonl"))
    found = {}
    remaining = set(target_ids)
    n_short = 0
    for mi, member in enumerate(members):
        if not remaining:
            break
        split = member.split("/")[1]
        with zf.open(member) as fh:
            for line_bytes in fh:
                if not remaining:
                    break
                line = line_bytes.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for doc in extract_docs(entry, split):
                    doc_id = doc["id"]
                    if doc_id not in remaining:
                        continue
                    text = doc.get("text")
                    if not text or len(text.split()) < MIN_WORDS:
                        n_short += 1
                        remaining.discard(doc_id)
                        continue
                    found[doc_id] = text
                    remaining.discard(doc_id)
        if mi % 100 == 0:
            print(f"[sample] scanned {mi+1}/{len(members)} members, "
                  f"{len(found)} found, {len(remaining)} remaining", flush=True)
    if remaining:
        print(f"[sample] WARNING: {len(remaining)} target ids never found "
              f"in gagle.zip (below MIN_WORDS or absent): "
              f"{list(remaining)[:10]}...", flush=True)
    print(f"[sample] {n_short} sampled ids dropped for <{MIN_WORDS} words "
          f"in raw text (distinct from build_p0_local's MIN_TOKENS=320 "
          f"token-level gate)", flush=True)
    return found


def main():
    frame = build_sample_frame()
    print(f"[sample] scanning gagle.zip for {len(frame)} raw texts "
          f"(single pass, per the governing read-once constraint)...", flush=True)
    texts = collect_texts(set(frame["id"]))

    frame = frame[frame["id"].isin(texts)].copy()
    frame["n_words_available"] = frame["id"].map(lambda i: len(texts[i].split()))

    with open(OUT_PATH, "w") as f:
        for row in frame.itertuples(index=False):
            f.write(json.dumps({
                "id": row.id, "domain": row.domain, "generator": row.generator,
                "batch": row.batch, "text": texts[row.id],
            }) + "\n")

    b1 = (frame["batch"] == 1).sum()
    b2 = (frame["batch"] == 2).sum()
    print(f"[sample] wrote {len(frame)} docs to {OUT_PATH} "
          f"(batch1={b1}, batch2={b2})", flush=True)
    if b1 < 10 * PER_STRATUM_PER_BATCH or b2 < 10 * PER_STRATUM_PER_BATCH:
        print(f"[sample] WARNING: short of the {10*PER_STRATUM_PER_BATCH}/batch "
              f"target after MIN_WORDS drops -- some ids were sampled but had "
              f"raw text under {MIN_WORDS} words. Re-draw needed if this "
              f"matters for the 8h-gate arithmetic.", flush=True)


if __name__ == "__main__":
    main()
