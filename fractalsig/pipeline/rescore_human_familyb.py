#!/usr/bin/env python3
"""
Driver: re-score Family B for the 5,000 post-X-01 human rows, ONE scorer per
invocation.

WHY A DRIVER AND NOT AN EDIT
  build_p1_feature_store_gpu_selfscored.py is the script that produced job 56's
  230,530-row production store. It is both-models-resident (X-17), which on this
  cluster's real 23.62 GiB slice (E-18) means ~19.5 GB of weights before any
  activation -- that is what job 56's 4,339 allocator OOM warnings were.
  Rather than edit a proven production path, this driver rebinds the module's
  output/checkpoint/manifest globals and calls its own main() once per scorer.
  main() reads those names as module globals, so rebinding them here redirects
  the run without touching the file.

  Each scorer runs in its OWN process (one invocation = one --scorer), so the
  "models never co-resident" property is guaranteed by process isolation rather
  than by trusting a del/empty_cache to have worked. VRAM is printed at entry
  and exit so the trace is auditable the same way X-03's was.

SCOPE
  Reads ONLY gpu_manifest_shards/human_dedup_x01.parquet, via a scratch
  manifest dir containing just that shard. Never the 720-shard corpus.

  Writes data/human_familyb_<scorer>.jsonl. Does NOT touch
  data/p1_feature_store_gpu_selfscored.jsonl.

USAGE
  python -m fractalsig.pipeline.rescore_human_familyb --scorer gemma-2b   --manifest-dir DIR
  python -m fractalsig.pipeline.rescore_human_familyb --scorer mistral-7b --manifest-dir DIR
"""
from fractalsig import paths as _paths  # noqa: E402
import argparse
import os
import sys

HERE = _paths.WORKSPACE_STR
DATA = os.path.join(HERE, "data")


def vram(tag):
    try:
        import torch
        if not torch.cuda.is_available():
            print(f"[vram] {tag}: no CUDA")
            return
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"[vram] {tag:<26} free {free_b / 2**30:6.2f} GiB / "
              f"total {total_b / 2**30:6.2f} GiB", flush=True)
    except Exception as e:  # never let instrumentation kill the run
        print(f"[vram] {tag}: unavailable ({e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", required=True, choices=["gemma-2b", "mistral-7b"])
    ap.add_argument("--manifest-dir", required=True,
                    help="dir containing ONLY human_dedup_x01.parquet")
    args = ap.parse_args()

    shards = [f for f in os.listdir(args.manifest_dir) if f.endswith(".parquet")]
    if shards != ["human_dedup_x01.parquet"]:
        print(f"[FATAL] manifest dir must contain exactly human_dedup_x01.parquet, "
              f"found {shards}", flush=True)
        return 1

    # Set BEFORE importing: build_p1_feature_store_gpu reads GAGLE_MANIFEST_DIR
    # at import time into MANIFEST_DIR.
    os.environ["GAGLE_MANIFEST_DIR"] = args.manifest_dir
    pass  # import path handled by the fractalsig package
    pass  # import path handled by the fractalsig package
    import fractalsig.pipeline.build_p1_feature_store_gpu_selfscored as M

    tag = args.scorer.replace("-", "_")
    M.MANIFEST_DIR = args.manifest_dir
    M.OUT_PATH = os.path.join(DATA, f"human_familyb_{tag}.jsonl")
    M.SHARD_CKPT_PATH = os.path.join(DATA, f"human_familyb_{tag}_checkpoint.txt")
    M.MANIFEST_PATH = os.path.join(DATA, f"human_familyb_{tag}_manifest.json")

    # Refuse to append onto a previous partial -- this file is an append-mode
    # output, so a rerun without clearing would silently double the rows.
    if os.path.exists(M.OUT_PATH):
        print(f"[FATAL] {M.OUT_PATH} already exists. Move it aside "
              f"(never delete, D-10) before re-running.", flush=True)
        return 1

    print(f"[info] scorer       : {args.scorer}")
    print(f"[info] manifest dir : {M.MANIFEST_DIR}")
    print(f"[info] out          : {M.OUT_PATH}", flush=True)
    vram("before load")
    M.main(scorers=(args.scorer,))
    vram("after run")

    n = sum(1 for _ in open(M.OUT_PATH))
    print(f"[check] rows written: {n}  (expect 5000)", flush=True)
    return 0 if n == 5000 else 1


if __name__ == "__main__":
    sys.exit(main())
