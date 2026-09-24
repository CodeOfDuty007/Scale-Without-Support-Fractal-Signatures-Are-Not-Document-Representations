"""One-pass extraction of {id: text} for every document in the manifest
(matches the current feature-store population exactly, verified earlier:
223,834 rows). Written once so RoBERTa's 8 per-fold jobs don't each re-read
721 parquet shards independently."""
from fractalsig import paths as _paths  # noqa: E402
import glob
import json
import os

import pyarrow.parquet as pq

MANIFEST_DIR = os.environ.get("GAGLE_MANIFEST_DIR",
                              _paths.MANIFEST_SHARD_DIR_STR)
OUT_PATH = os.path.join(_paths.WORKSPACE_STR, "data", "all_doc_texts.jsonl")


def main():
    shard_files = sorted(glob.glob(os.path.join(MANIFEST_DIR, "*.parquet")))
    print(f"[extract] {len(shard_files)} shards")
    n = 0
    with open(OUT_PATH, "w") as f:
        for si, shard in enumerate(shard_files, 1):
            table = pq.read_table(shard, columns=["id", "text"])
            for row in table.to_pylist():
                f.write(json.dumps({"id": row["id"], "text": row["text"]}) + "\n")
                n += 1
            if si % 100 == 0 or si == len(shard_files):
                print(f"[extract] {si}/{len(shard_files)} shards, {n} docs", flush=True)
    print(f"[extract] done: {n} docs -> {OUT_PATH}")


if __name__ == "__main__":
    main()
